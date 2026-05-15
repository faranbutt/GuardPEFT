import ast
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from huggingface_hub import login
from src.utils.common import (clean_reasoning, extract_reasoning, get_device, load_model_and_tokenizer, print_stats, run_judge, save_to_hf, set_seed, clean_text, map_label, free_memory)
from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config = load_config("configs/data.yaml")
SEED = config["seed"]
HF_REPO = "faranbutt789/ethics-checkpoints"


TARGET_TOTAL = 2000
OUTPUT_DIR = "data/pku"
CHECKPOINT_CSV = f"{OUTPUT_DIR}/pku_checkpoint.csv"
OUTPUT_CSV = f"{OUTPUT_DIR}/pku_prepared.csv"
HF_CONFIG = "pku"


set_seed()
login_to_hf()
device = get_device()
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {device}")


print("\n[1/5] Loading BeaverTails …")
rows = load_dataset("PKU-Alignment/BeaverTails", split="330k_test")
df = pd.DataFrame(rows)
print(f"  Loaded: {df.shape}")
print(f"  is_safe distribution: {df['is_safe'].value_counts().to_dict()}")

df["label"] = df["is_safe"].apply(lambda x: "SAFE" if bool(x) else "UNSAFE")


if df["cat_dict"].dtype == object and isinstance(df["cat_dict"].iloc[0], str):
    df["cat_dict"] = df["cat_dict"].apply(ast.literal_eval)


def _get_violation_str(d: dict) -> str:
    violations = [k for k, v in d.items() if v]
    return ",".join(violations) if violations else "none"


def _get_primary_cat(d: dict) -> str:
    for k, v in d.items():
        if v:
            return k
    return "none"


df["categories"] = df["cat_dict"].apply(_get_violation_str)
df["primary_cat"] = df["cat_dict"].apply(_get_primary_cat)


df = df.rename(columns={"prompt": "user_prompt", "response": "assistant_response"})
df = df[
    ["user_prompt", "assistant_response", "label", "categories", "primary_cat"]
].dropna()


safe_df = df[df["label"] == "SAFE"].copy()
unsafe_df = df[df["label"] == "UNSAFE"].copy()

n_categories = unsafe_df["primary_cat"].nunique()
n_per_cat = TARGET_TOTAL // (2 * n_categories)
n_unsafe = n_per_cat * n_categories
n_safe = n_unsafe

print(
    f"\n  Categories: {n_categories}  |  Per cat: {n_per_cat}  "
    f"→  {n_unsafe} UNSAFE + {n_safe} SAFE = {n_unsafe + n_safe} total"
)

np.random.seed(SEED)
unsafe_balanced = (
    unsafe_df.groupby("primary_cat", group_keys=False)
    .apply(lambda g: g.sample(n=min(n_per_cat, len(g)), random_state=SEED))
    .reset_index(drop=True)
)
safe_balanced = safe_df.sample(
    n=min(n_safe, len(safe_df)), random_state=SEED
).reset_index(drop=True)

sample_df = (
    pd.concat([unsafe_balanced, safe_balanced], ignore_index=True)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)
print(f"  Sample: {sample_df.shape}  |  {sample_df['label'].value_counts().to_dict()}")


print("\n[2/5] Loading Qwen2.5-7B-Instruct …")
model, tokenizer = load_model_and_tokenizer()


def _generate_reasoning(
    user_prompt: str, assistant_response: str, label: str, categories: str
) -> str:
    prompt = f"""You are a precise content-safety annotator.
Write ONE sentence explaining why this AI response is {label}.

User asked: {user_prompt[:300]}
AI responded: {assistant_response[:400]}
Safety category: {categories}
Label: {label}

RULES:
- Start with "The assistant" or "The response"
- Name the specific harm or safe behavior
- One sentence only, no label words (SAFE/UNSAFE)

Examples:
[SAFE]   The assistant declines to assist with the harmful request and redirects to legal channels.
[UNSAFE] The assistant provides detailed instructions that could facilitate physical harm to others.

REASONING: """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(
        model.device
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.55,
            top_p=0.9,
            repetition_penalty=1.15,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def _clean_pku_reasoning(raw: str) -> str | None:
    if not raw:
        return None
    raw = re.sub(r"```[\s\S]*?```", "", raw).strip()
    raw = re.sub(
        r"^\s*(REASONING|Reasoning|Sentence)[:\-]\s*", "", raw, flags=re.IGNORECASE
    ).strip()

    raw = re.split(r"(?<=[.!?])\s+", raw)[0].strip()
    if not re.match(r"^The (assistant|response)", raw, re.IGNORECASE):
        raw = "The assistant " + raw[0].lower() + raw[1:]
    if raw and raw[-1] not in ".!?":
        raw += "."
    return raw if len(raw.split()) >= 6 else None


SAFE_CUES = [
    "declines",
    "refuses",
    "withholds",
    "avoids",
    "does not",
    "cannot",
    "redirects",
    "appropriately",
    "refrains",
    "correctly",
    "prioritises",
    "prioritizes",
]
UNSAFE_CUES = [
    "provides",
    "supplies",
    "facilitates",
    "enables",
    "instructs",
    "produces",
    "assists",
    "details",
    "explains",
    "promotes",
    "glorifies",
    "gives",
]


def _is_valid(reasoning: str, label: str) -> bool:
    if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
        return False
    r = reasoning.lower()
    if not re.match(r"^the (assistant|response)", reasoning, re.IGNORECASE):
        return False
    if label == "SAFE" and not any(c in r for c in SAFE_CUES):
        return False
    if label == "UNSAFE" and not any(c in r for c in UNSAFE_CUES):
        return False
    return True


print(f"\n[3/5] Generating reasoning …")
final_data = []
stats = {"total": 0, "rule_pass": 0, "judge_pass": 0, "rejected": 0}

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    stats["total"] += 1

    up = str(row["user_prompt"])
    ar = str(row["assistant_response"])
    lab = str(row["label"])
    cat = str(row["categories"])

    raw = _generate_reasoning(up, ar, lab, cat)
    reasoning = _clean_pku_reasoning(raw)

    if not reasoning or not _is_valid(reasoning, lab):
        stats["rejected"] += 1
        continue
    stats["rule_pass"] += 1

    if not run_judge(ar, lab, reasoning, model, tokenizer, device):
        stats["rejected"] += 1
        continue
    stats["judge_pass"] += 1

    final_data.append(
        {
            "user_prompt": up,
            "assistant_response": ar,
            "label": lab,
            "categories": cat,
            "reasoning": reasoning,
            "source": "pku",
        }
    )
    print_stats(stats)

    if stats["total"] % 50 == 0:
        pd.DataFrame(final_data).to_csv(CHECKPOINT_CSV, index=False)
        save_to_hf(final_data, config_name=HF_CONFIG)
        print(f"  Checkpoint: {len(final_data)} rows")


out_df = pd.DataFrame(final_data)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(out_df)} rows → {OUTPUT_CSV}")
save_to_hf(final_data, config_name=HF_CONFIG)
