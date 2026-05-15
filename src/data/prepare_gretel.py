import os
import re
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from huggingface_hub import login
from src.utils.common import (clean_reasoning, extract_reasoning, get_device, load_model_and_tokenizer, print_stats, run_judge, save_to_hf, set_seed, clean_text, map_label, free_memory)
from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config = load_config("configs/data.yaml")
SEED = config["seed"]
HF_REPO = "faranbutt789/ethics-checkpoints"


OUTPUT_DIR = "data/gretel"
OUTPUT_CSV = f"{OUTPUT_DIR}/gretel_prepared.csv"
HF_CONFIG = "gretel"


set_seed()
login_to_hf()
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n[1/3] Loading Gretel Safety Alignment dataset …")
ds = load_dataset("gretelai/gretel-safety-alignment-en-v1")
df = ds["train"].to_pandas()
print(f"  Loaded: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

print("\n[2/3] Mapping columns …")

df = df.rename(
    columns={
        "prompt": "user_prompt",
        "response": "assistant_response",
        "judge_safe_response_reasoning": "reasoning",
        "risk_category": "categories",
        "judge_safe_response_score": "score",
    }
)

df["label"] = df["score"].apply(lambda s: "SAFE" if float(s) <= 3 else "UNSAFE")
df = df[
    ["user_prompt", "assistant_response", "label", "categories", "reasoning"]
].dropna()
df["assistant_response"] = df["assistant_response"].astype(str).str.strip()
df["user_prompt"] = df["user_prompt"].astype(str).str.strip()
df["reasoning"] = df["reasoning"].astype(str).str.strip()
df = df[(df["assistant_response"] != "") & (df["user_prompt"] != "")]

print(f"  After clean: {df.shape}")
print(f"  Label distribution: {df['label'].value_counts().to_dict()}")


print("\n[3/3] Validating reasoning …")

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
    "safe",
    "harmless",
    "appropriate",
    "helpful",
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
    "harmful",
    "unsafe",
    "violates",
    "dangerous",
]


def _is_valid(reasoning: str, label: str) -> bool:
    if not reasoning or len(reasoning.split()) < 4:
        return False
    r = reasoning.lower()
    if label == "SAFE" and not any(c in r for c in SAFE_CUES):
        return False
    if label == "UNSAFE" and not any(c in r for c in UNSAFE_CUES):
        return False
    return True


def _norm(t: str) -> set:
    return set(re.sub(r"[^\w\s]", "", t.lower()).split())


seen_reasonings = []


def _is_duplicate(reasoning: str) -> bool:
    nw = _norm(reasoning)
    for prev in seen_reasonings[-200:]:
        pw = _norm(prev)
        if pw and len(nw & pw) / max(len(nw | pw), 1) > 0.75:
            return True
    return False


final_data = []
stats = {"total": 0, "rule_pass": 0, "judge_pass": 0, "rejected": 0}

for _, row in tqdm(df.iterrows(), total=len(df)):
    stats["total"] += 1
    reasoning = clean_reasoning(row["reasoning"]) or row["reasoning"].strip()

    if not _is_valid(reasoning, row["label"]):
        stats["rejected"] += 1
        continue

    if _is_duplicate(reasoning):
        stats["rejected"] += 1
        continue

    stats["judge_pass"] += 1
    seen_reasonings.append(reasoning)

    final_data.append(
        {
            "user_prompt": row["user_prompt"],
            "assistant_response": row["assistant_response"],
            "label": row["label"],
            "categories": row["categories"],
            "reasoning": reasoning,
            "source": "gretel",
        }
    )

print_stats(stats)

out_df = pd.DataFrame(final_data)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(out_df)} rows → {OUTPUT_CSV}")
save_to_hf(final_data, config_name=HF_CONFIG)
