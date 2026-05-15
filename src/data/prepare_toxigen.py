import os
import re
import sys

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from huggingface_hub import login
from src.utils.common import (clean_reasoning, extract_reasoning, get_device, load_model_and_tokenizer, print_stats, run_judge, save_to_hf, set_seed, clean_text, map_label, free_memory)
from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config = load_config("configs/data.yaml")
SEED = config["seed"]
HF_REPO = "faranbutt789/ethics-checkpoints"


SAMPLE_SIZE = 2000
CHECKPOINT_CSV = "data/toxigen/toxigen_prepared_checkpoint.csv"
OUTPUT_CSV = "data/toxigen/toxigen_prepared.csv"
HF_CONFIG = "toxigen"


set_seed()
login_to_hf()
device = get_device()
print(f"Device: {device}")


print("\n[1/5] Loading ToxiGen dataset …")
toxigen_ds = load_dataset("toxigen/toxigen-data", "annotated")
df = toxigen_ds["train"].to_pandas()

df = df.rename(columns={"text": "assistant_response", "predicted_group": "categories"})


def _map_toxicity(score) -> str:
    return "UNSAFE" if float(score) >= 3 else "SAFE"


df["label"] = df["toxicity_human"].apply(_map_toxicity)
df["categories"] = df["categories"].apply(lambda x: [str(x).lower()])
df = df[["assistant_response", "categories", "label"]].dropna()
df["assistant_response"] = df["assistant_response"].astype(str).str.strip()
df = df[df["assistant_response"] != ""]

# Balance
n = min(df["label"].value_counts()["SAFE"], df["label"].value_counts()["UNSAFE"])
balanced = (
    pd.concat(
        [
            df[df["label"] == "SAFE"].sample(n, random_state=SEED),
            df[df["label"] == "UNSAFE"].sample(n, random_state=SEED),
        ]
    )
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

print(f"Balanced dataset: {balanced['label'].value_counts().to_dict()}")

print("\n[2/5] Loading Qwen2.5-7B-Instruct …")
model, tokenizer = load_model_and_tokenizer()


def _generate_user_prompt(assistant_response: str, label: str, category: str) -> str:
    system = (
        "You are an expert at reverse-engineering user prompts from model outputs. "
        "Given the assistant's response, its safety label, and the topic category, "
        "create a plausible, short user query or instruction that would naturally "
        "lead an AI to produce that exact response.\n"
        "Rules:\n"
        "- Must be different from 'Continue the following text in a natural way'.\n"
        "- Feel like a real user request: a question, a statement to complete, a command.\n"
        "- Keep it concise (one sentence).\n"
        "- Do NOT copy the assistant response.\n\n"
        "Examples:\n"
        "Response: 'the black man is a savage with no regard for life or the law.'\n"
        "Label: UNSAFE | Category: Race-Ethnicity\n"
        "Prompt: What are some negative stereotypes about black people?\n\n"
        "Response: 'mental illnesses can be treated with medication and psychotherapy.'\n"
        "Label: SAFE | Category: Disability\n"
        "Prompt: How are mental health conditions treated?\n"
        "---\n"
        f"Now reverse-engineer the prompt for:\n"
        f"Response: {assistant_response}\n"
        f"Label: {label} | Category: {category}\n"
        "Prompt:"
    )
    inputs = tokenizer(system, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True
        )
    text = (
        tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        .strip()
        .rstrip(":")
        .strip()
    )
    return text if text else "Can you continue this?"


def _generate_reasoning(text: str, label: str) -> str:
    prompt = f"""You are an expert in content safety and toxicity detection.

Here are examples:

Statement: "I think we should have a respectful debate about immigration policies."
Label: SAFE
REASONING: The statement is neutral and promotes civil discussion, containing no offensive or harmful language.

Statement: "All members of group X are stupid and should be deported."
Label: UNSAFE
REASONING: The statement uses a demeaning slur and calls for harmful action against a whole group, which is hate speech.

Statement: "I disagree with your opinion, but I respect your right to say it."
Label: SAFE
REASONING: The statement expresses disagreement respectfully without personal attacks or toxicity.

Now analyze:

Statement: {text}
Label: {label}

RULES:
- One sentence only
- Be specific: mention harmful language, stereotypes, threats, or respectful/neutral tone
- Do NOT repeat the statement
- Do NOT say "this is safe/unsafe"

FORMAT:
REASONING: <one sentence>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, temperature=0.5, top_p=0.9)
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


UNSAFE_KW = [
    "toxic",
    "offensive",
    "harmful",
    "hateful",
    "abusive",
    "stereotype",
    "slur",
    "threat",
    "harassment",
    "demeaning",
]
SAFE_KW = ["neutral", "harmless", "non-offensive", "respectful", "benign", "civil"]


def _is_valid_reasoning(reasoning: str, text: str, label: str) -> bool:
    if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
        return False
    if text[:40].lower() in reasoning.lower():
        return False
    if label == "UNSAFE":
        return any(kw in reasoning.lower() for kw in UNSAFE_KW)
    return any(kw in reasoning.lower() for kw in SAFE_KW)


print(f"\n[3/5] Sampling {SAMPLE_SIZE} rows …")
sample = balanced.sample(SAMPLE_SIZE, random_state=SEED)

print("\n[4/5] Generating user prompts …")
user_prompts = []
for idx, row in tqdm(sample.iterrows(), total=len(sample)):
    cat = (
        row["categories"][0]
        if isinstance(row["categories"], list)
        else str(row["categories"])
    )
    user_prompts.append(
        _generate_user_prompt(row["assistant_response"], row["label"], cat)
    )
sample = sample.copy()
sample["user_prompt"] = user_prompts


print("\n[5/5] Generating reasoning …")
final_data = []
stats = {"total": 0, "rule_pass": 0, "judge_pass": 0, "rejected": 0}

os.makedirs("data/toxigen", exist_ok=True)

for idx, row in tqdm(sample.iterrows(), total=len(sample)):
    stats["total"] += 1

    text = row["assistant_response"]
    label = row["label"]
    categories = row["categories"]
    user_p = row["user_prompt"]

    raw = _generate_reasoning(text, label)
    reasoning = extract_reasoning(raw)
    reasoning = clean_reasoning(reasoning)

    if not reasoning or not _is_valid_reasoning(reasoning, text, label):
        stats["rejected"] += 1
        continue
    stats["rule_pass"] += 1

    if not run_judge(text, label, reasoning, model, tokenizer, device):
        stats["rejected"] += 1
        continue
    stats["judge_pass"] += 1

    final_data.append(
        {
            "user_prompt": user_p,
            "assistant_response": text,
            "label": label,
            "categories": categories,
            "reasoning": reasoning,
            "source": "toxigen",
        }
    )
    print_stats(stats)

    if stats["total"] % 50 == 0:
        pd.DataFrame(final_data).to_csv(CHECKPOINT_CSV, index=False)
        save_to_hf(final_data, config_name=HF_CONFIG)
        print(f"  Checkpoint saved: {len(final_data)} rows")


out_df = pd.DataFrame(final_data)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(out_df)} rows → {OUTPUT_CSV}")
save_to_hf(final_data, config_name=HF_CONFIG)
