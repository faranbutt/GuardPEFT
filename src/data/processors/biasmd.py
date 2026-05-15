import re

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from ...core.config import SEED
from ...core.model import load_model_and_tokenizer
from ...core.utils import get_device
from .prompts import (
    BIASMD_GEN_SYSTEM,
    BIASMD_GEN_USER_TEMPLATE,
    BIASMD_JUDGE_PROMPT_TEMPLATE,
)


class BiasMDProcessor:
    def __init__(self, target_unsafe=2000):
        self.target_unsafe = target_unsafe
        self.device = get_device()
        self.model = None
        self.tokenizer = None

    def load_data(self):
        print("\n[BiasMD] Loading BiasMD dataset …")
        biasds = load_dataset("PardisSzah/BiasMD")
        df = biasds["train"].to_pandas()
        if "Model" in df.columns:
            df = df.drop("Model", axis=1)

        sample_size = min(5000, len(df))
        df = df.sample(sample_size, random_state=SEED).reset_index(drop=True)
        return df

    def init_model(self):
        print("\n[BiasMD] Loading model …")
        self.model, self.tokenizer = load_model_and_tokenizer()

    def process(self, df):
        if self.model is None:
            self.init_model()

        df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        final_data = []
        stats = {
            "total": 0,
            "extracted": 0,
            "rule_pass": 0,
            "judge_pass": 0,
            "skipped": 0,
        }
        unsafe_count = 0

        for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled)):
            if unsafe_count >= self.target_unsafe:
                break

            stats["total"] += 1
            original_safe = str(row["Statement"])
            explanation = str(row["Explanation"])
            bias_type = str(row["Type"])

            raw_gen = self._generate_unsafe(original_safe, explanation, bias_type)
            unsafe_stmt, reasoning = self._extract_unsafe_content(
                raw_gen, original_safe
            )

            if not unsafe_stmt or not reasoning:
                stats["skipped"] += 1
                continue
            stats["extracted"] += 1

            if not self._rule_validate(unsafe_stmt, reasoning, original_safe):
                stats["skipped"] += 1
                continue
            stats["rule_pass"] += 1

            if not self._judge_bias_introduction(original_safe, unsafe_stmt, bias_type):
                stats["skipped"] += 1
                continue
            stats["judge_pass"] += 1

            # Add SAFE pair
            final_data.append(
                {
                    "user_prompt": "Complete the following statement:",
                    "assistant_response": original_safe,
                    "label": "SAFE",
                    "categories": "",
                    "reasoning": explanation,
                    "source": "biasmd",
                }
            )
            # Add UNSAFE pair
            final_data.append(
                {
                    "user_prompt": "Complete the following statement:",
                    "assistant_response": unsafe_stmt,
                    "label": "UNSAFE",
                    "categories": bias_type,
                    "reasoning": reasoning,
                    "source": "biasmd",
                }
            )
            unsafe_count += 1

        return pd.DataFrame(final_data)

    def _generate_unsafe(self, statement, explanation, bias_type):
        user_content = BIASMD_GEN_USER_TEMPLATE.format(
            bias_type=bias_type, statement=statement, explanation=explanation
        )
        messages = [
            {"role": "system", "content": BIASMD_GEN_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def _extract_unsafe_content(self, raw_text: str, original_statement: str):
        raw_text = re.sub(r"```[\s\S]*?```", "", raw_text).strip()
        pattern = r"UNSAFE_STATEMENT:\s*(.*?)\s*REASONING:\s*(.*?)(?=UNSAFE_STATEMENT:|SAFE:|Task:|$)"
        matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

        for unsafe_stmt, reasoning in matches:
            unsafe_stmt = unsafe_stmt.strip()
            reasoning = reasoning.strip()
            # ... (clean logic from original)
            if unsafe_stmt.lower() == original_statement.lower():
                continue
            return unsafe_stmt, reasoning
        return None, None

    def _rule_validate(
        self, unsafe_stmt: str, reasoning: str, original_safe: str
    ) -> bool:
        if not unsafe_stmt or not reasoning:
            return False
        if len(unsafe_stmt.split()) < 4 or len(reasoning.split()) < 5:
            return False
        if unsafe_stmt.strip().lower() == original_safe.strip().lower():
            return False

        original_words = set(re.findall(r"\b\w+\b", original_safe.lower()))
        unsafe_words = set(re.findall(r"\b\w+\b", unsafe_stmt.lower()))
        if len(original_words & unsafe_words) < 0.3 * len(original_words):
            return False

        if not any(
            kw in reasoning.lower()
            for kw in ["bias", "stereotype", "subtly", "implies"]
        ):
            return False
        return True

    def _judge_bias_introduction(
        self, safe_stmt: str, unsafe_stmt: str, bias_type: str
    ) -> bool:
        prompt = BIASMD_JUDGE_PROMPT_TEMPLATE.format(
            safe=safe_stmt, unsafe=unsafe_stmt, bias_type=bias_type
        )
        messages = [
            {"role": "system", "content": "Answer only YES or NO."},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            chat_prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = (
            self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            .strip()
            .upper()
        )
        return response.startswith("YES")
