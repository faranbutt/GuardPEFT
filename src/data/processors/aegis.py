import os
import re
from collections import deque

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from ...core.config import HF_TOKEN, SEED
from ...core.judge import run_judge
from ...core.model import load_model_and_tokenizer
from ...core.utils import get_device, print_stats, set_seed


class AegisProcessor:
    def __init__(self, target_rows=2000, checkpoint_n=50, recent_window=30):
        self.target_rows = target_rows
        self.checkpoint_n = checkpoint_n
        self.recent_window = recent_window
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.yes_id = None
        self.no_id = None

    def load_data(self):
        print("\n[Aegis] Loading Aegis dataset …")
        dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
        df = dataset["train"].to_pandas()

        df = df.dropna(subset=["violated_categories"])
        df = df.replace("", np.nan).dropna(subset=["violated_categories"])
        df = df[
            ["prompt", "response", "violated_categories", "response_label"]
        ].dropna()

        df["violated_categories"] = df["violated_categories"].str.split(", ")
        df_exploded = df.explode("violated_categories")

        counts = df_exploded["violated_categories"].value_counts()
        rare = counts[counts < 50].index
        df_exploded["violated_categories"] = df_exploded["violated_categories"].replace(
            rare, "Other"
        )
        df_final = df_exploded[
            ~df_exploded["violated_categories"].isin(
                ["Manipulation", "Immoral/Unethical", "Illegal Activity"]
            )
        ].copy()

        # Balance by label
        df_balanced = self._balance(df_final, "response_label", cap_size=7000)

        # Even category distribution
        categories = df_balanced["violated_categories"].unique()
        target_per_cat = 5000 // len(categories)

        df_top = df_balanced.groupby("violated_categories", group_keys=False).apply(
            lambda g: self._get_best(g, target_per_cat)
        )
        if len(df_top) < 5000:
            remaining = df_balanced.drop(df_top.index).sample(
                n=5000 - len(df_top), random_state=SEED
            )
            df_top = pd.concat([df_top, remaining])

        df_top = df_top.rename(
            columns={
                "prompt": "user_prompt",
                "response": "assistant_response",
                "violated_categories": "categories",
                "response_label": "label",
            }
        )
        df_top["label"] = df_top["label"].str.upper()
        return df_top

    def _balance(self, df_, label_col, cap_size):
        over = df_.groupby(label_col).filter(lambda x: len(x) > cap_size)
        under = df_.groupby(label_col).filter(lambda x: len(x) <= cap_size)
        return (
            pd.concat(
                [over.groupby(label_col).sample(n=cap_size, random_state=SEED), under]
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )

    def _get_best(self, group, target_per_cat):
        return (
            group
            if len(group) <= target_per_cat
            else group.sample(n=target_per_cat, random_state=SEED)
        )

    def init_model(self):
        print("\n[Aegis] Loading model for generation …")
        self.model, self.tokenizer = load_model_and_tokenizer()
        self.yes_id = self.tokenizer.encode("YES", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("NO", add_special_tokens=False)[0]

    def process(self, df_top):
        if self.model is None:
            self.init_model()

        df_shuffled = df_top.sample(frac=1, random_state=SEED).reset_index(drop=True)
        final_data = []
        stats = {"total": 0, "accepted": 0, "rejected": 0}
        recent_q = deque(maxlen=self.recent_window)

        for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled)):
            if stats["accepted"] >= self.target_rows:
                break

            stats["total"] += 1
            reasoning = self._generate_best(row, recent=recent_q)

            if reasoning:
                out = row.to_dict()
                out["reasoning"] = reasoning
                out["source"] = "aegis"
                final_data.append(out)
                recent_q.append(reasoning)
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1

        return pd.DataFrame(final_data)

    def _generate_best(self, row, attempts=5, recent=None):
        up = str(row["user_prompt"])
        ar = str(row["assistant_response"])
        lab = str(row["label"])
        cat = str(row["categories"])
        best_cand, best_score = None, -999

        for attempt in range(attempts):
            raw = self._generate_raw(up, ar, cat, lab, attempt)
            reasoning = self._extract_and_clean(raw)
            if not reasoning:
                continue
            if recent and self._is_duplicate(reasoning, recent):
                continue

            is_valid, score = self._rule_validate(reasoning, lab, cat)
            if score > best_score:
                best_score, best_cand = score, reasoning
            if not is_valid:
                continue

            if run_judge(ar, lab, reasoning, self.model, self.tokenizer):
                return reasoning

        return best_cand if (best_cand and best_score >= 5) else None

    def _generate_raw(
        self, user_prompt, assistant_response, category, label, attempt=0
    ):
        # Prompts from original file
        GENERATION_SYSTEM = "You are a precise content-safety annotator..."  # Simplified for brevity here, should be full in actual implementation
        # (Using original prompts)
        from .prompts import AEGIS_GENERATION_SYSTEM, AEGIS_GENERATION_USER

        user_text = AEGIS_GENERATION_USER.format(
            user_prompt=user_prompt[:300],
            assistant_response=assistant_response[:400],
            category=category,
            label_upper=label.upper(),
        )
        messages = [
            {"role": "system", "content": AEGIS_GENERATION_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1200
        ).to(self.device)
        temp = [0.55, 0.70, 0.80, 0.65, 0.75][attempt % 5]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=55,
                do_sample=True,
                temperature=temp,
                top_p=0.90,
                repetition_penalty=1.20,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def _extract_and_clean(self, raw: str) -> str | None:
        if not raw:
            return None
        raw = re.sub(r"```[\s\S]*?```", "", raw).strip()
        raw = re.sub(
            r"^\s*(Sentence|Reasoning|Output|Answer|REASONING|The reasoning)[:\-]\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
        for line in raw.splitlines():
            line = line.strip()
            if not line or re.match(
                r"^(user|assistant|category|label|rules|note)[:\s]", line, re.IGNORECASE
            ):
                continue
            if re.match(r"^\[?(SAFE|UNSAFE)\]?$", line, re.IGNORECASE):
                continue
            raw = line
            break
        else:
            return None
        raw = re.split(r"(?<=[.!?])\s+", raw)[0].strip()
        if not re.match(r"^The (assistant|response)", raw, re.IGNORECASE):
            raw = "The assistant " + raw[0].lower() + raw[1:]
        if raw and raw[-1] not in ".!?":
            raw += "."
        return raw

    def _is_duplicate(self, reasoning: str, recent) -> bool:
        def _norm(t):
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()

        nw = set(_norm(reasoning).split())
        for prev in recent:
            pw = set(_norm(prev).split())
            if pw and len(nw & pw) / len(nw | pw) > 0.70:
                return True
        return False

    def _rule_validate(self, reasoning: str, label: str, category: str):
        # Implementation of _rule_validate from original file
        # ... (skipping for brevity, but I will include it in the real file)
        from .utils import aegis_rule_validate

        return aegis_rule_validate(reasoning, label, category)
