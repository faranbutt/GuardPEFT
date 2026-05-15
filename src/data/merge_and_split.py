import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from huggingface_hub import login
from src.utils.common import (clean_reasoning, extract_reasoning, get_device, load_model_and_tokenizer, print_stats, run_judge, save_to_hf, set_seed, clean_text, map_label, free_memory)
from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config = load_config("configs/data.yaml")
SEED = config["seed"]
HF_REPO = "faranbutt789/ethics-checkpoints"

PREPARED_FILES = {
    "toxigen": config["paths"]["toxigen_prepared"],
    "ethics_commonsense": config["paths"]["ethics_commonsense"],
    "ethics_justice": config["paths"]["ethics_justice"],
    "ethics_virtue": config["paths"]["ethics_virtue"],
    "ethics_deontology": config["paths"]["ethics_deontology"],
    "aegis": config["paths"]["aegis_prepared"],
    "pku": config["paths"]["pku_prepared"],
    "gretel": config["paths"]["gretel_prepared"],
}

REQUIRED_COLS = [
    "user_prompt",
    "assistant_response",
    "label",
    "categories",
    "reasoning",
]
OUTPUT_DIR = config["paths"]["small_dataset_dir"]


login_to_hf()
os.makedirs(OUTPUT_DIR, exist_ok=True)

dfs = []
for source, path in PREPARED_FILES.items():
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — run the corresponding prep script first.")
        continue
    df = pd.read_csv(path)
    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = source
    # Keep only required + source
    available = [c for c in REQUIRED_COLS + ["source"] if c in df.columns]
    df = df[available].dropna(
        subset=["user_prompt", "assistant_response", "label", "reasoning"]
    )
    df["label"] = df["label"].str.upper().str.strip()
    df = df[df["label"].isin(["SAFE", "UNSAFE"])]
    dfs.append(df)
    print(f"  Loaded {len(df):>5} rows from {source}")

if not dfs:
    raise RuntimeError("No prepared CSVs found. Run scripts 01–05 first.")

merged = pd.concat(dfs, ignore_index=True)
print(f"\nMerged: {merged.shape}")
print(f"Label split:\n{merged['label'].value_counts()}")
print(f"Source split:\n{merged['source'].value_counts()}")


before = len(merged)
merged = merged.drop_duplicates(subset=["assistant_response"])
print(f"\nDeduplicated: {before} → {len(merged)} rows")


np.random.seed(SEED)

train_df, temp_df = train_test_split(
    merged, test_size=0.20, random_state=SEED, stratify=merged["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
)

print(f"\nSplits:")
print(f"  train: {len(train_df)}  |  {train_df['label'].value_counts().to_dict()}")
print(f"  val  : {len(val_df)}    |  {val_df['label'].value_counts().to_dict()}")
print(f"  test : {len(test_df)}   |  {test_df['label'].value_counts().to_dict()}")

train_path = config["paths"]["train"]
val_path = config["paths"]["val"]
test_path = config["paths"]["test"]

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"\nSaved:")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {test_path}")


save_to_hf(
    train_df.to_dict("records"),
    repo_name="faranbutt789/ethics-checkpoints",
    config_name="train",
)
save_to_hf(
    val_df.to_dict("records"),
    repo_name="faranbutt789/ethics-checkpoints",
    config_name="val",
)
save_to_hf(
    test_df.to_dict("records"),
    repo_name="faranbutt789/ethics-checkpoints",
    config_name="test",
)

print("\nDone.")
