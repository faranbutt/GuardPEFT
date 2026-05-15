import os
import re
import sys
from collections import deque

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


TARGET_UNSAFE = 2000
OUTPUT_DIR = "data/biasmd"
CHECKPOINT_CSV = f"{OUTPUT_DIR}/biasmd_checkpoint.csv"
OUTPUT_CSV = f"{OUTPUT_DIR}/biasmd_prepared.csv"
HF_CONFIG = "biasmd"

set_seed(SEED)
login_to_hf()
device = get_device()
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {device}")


print("\n[1/4] Loading Qwen2.5-7B-Instruct …")
model, tokenizer = load_model_and_tokenizer()


print("\n[2/4] Loading BiasMD dataset …")
biasds = load_dataset("PardisSzah/BiasMD")
df = biasds["train"].to_pandas()
if "Model" in df.columns:
    df = df.drop("Model", axis=1)
print(f"  Loaded {len(df)} rows")


sample_size = min(5000, len(df))
df = df.sample(sample_size, random_state=SEED).reset_index(drop=True)
print(f"  Using {len(df)} rows as generation pool")

#
GEN_SYSTEM = (
    "You are a bias injection expert. Your task is to take a safe statement and "
    "turn it into a subtly biased (UNSAFE) version by changing only 1-2 words. "
    "The unsafe version must look almost identical to the original. "
    "Never output markdown or extra commentary."
)

GEN_USER_TEMPLATE = """\
Target bias type: {bias_type}
Original safe statement: {statement}
Reasoning for bias: {explanation}

Output exactly in this format:
UNSAFE_STATEMENT: <subtly biased version>
REASONING: <why this introduces the {bias_type} bias>

Rules:
- Change at most 2 words.
- Do NOT add or remove whole clauses.
- The unsafe statement must be grammatically correct.
- Do NOT include any other text, headings, or examples.
"""


def build_gen_prompt(statement: str, explanation: str, bias_type: str) -> str:
    user_content = GEN_USER_TEMPLATE.format(
        bias_type=bias_type, statement=statement, explanation=explanation
    )
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_unsafe_content(raw_text: str):
    """Extract UNSAFE_STATEMENT and REASONING from model output."""
    raw_text = re.sub(r"```[\s\S]*?```", "", raw_text).strip()

    pattern = r"UNSAFE_STATEMENT:\s*(.*?)\s*REASONING:\s*(.*?)(?=UNSAFE_STATEMENT:|SAFE:|Task:|$)"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

    for unsafe_stmt, reasoning in matches:
        unsafe_stmt = unsafe_stmt.strip()
        reasoning = reasoning.strip()

        leak_markers = [
            "\n\n    Task:",
            "\n    Task:",
            "\nTask:",
            "Task: CONVERT",
            "Original safe statement:",
            "Target bias type:",
            "\n\n\n",
        ]
        for marker in leak_markers:
            if marker in reasoning:
                reasoning = reasoning.split(marker)[0].strip()

        if reasoning and reasoning[-1] not in ".!?":
            reasoning += "."

        if not unsafe_stmt or not reasoning:
            continue
        if not unsafe_stmt.endswith((".", "!", "?")):
            continue
        if not reasoning.endswith((".", "!", "?")):
            continue

        if unsafe_stmt.lower() == statement.lower():
            continue

        if any(
            x in unsafe_stmt.upper() for x in ["SAFE STATEMENT", "REASONING", "UNSAFE"]
        ):
            continue
        return unsafe_stmt, reasoning

    return None, None


def rule_validate(unsafe_stmt: str, reasoning: str, original_safe: str) -> bool:
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
        kw in reasoning.lower() for kw in ["bias", "stereotype", "subtly", "implies"]
    ):
        return False
    return True


JUDGE_PROMPT_TEMPLATE = """\
You are a strict bias detection expert.

Original safe statement: {safe}
Modified unsafe statement: {unsafe}
Target bias type: {bias_type}

Question: Does the unsafe statement introduce the "{bias_type}" bias in a subtle way, while remaining grammatically correct?
Answer only "YES" or "NO".
"""


def judge_bias_introduction(safe_stmt: str, unsafe_stmt: str, bias_type: str) -> bool:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        safe=safe_stmt, unsafe=unsafe_stmt, bias_type=bias_type
    )
    messages = [
        {"role": "system", "content": "Answer only YES or NO."},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat_prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    response = response.strip().upper()
    return response.startswith("YES")


print(
    f"\n[3/4] Generating unsafe statements (target: {TARGET_UNSAFE} unsafe records) …"
)
df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_data = []
stats = {"total": 0, "extracted": 0, "rule_pass": 0, "judge_pass": 0, "skipped": 0}
unsafe_count = 0

for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled)):
    if unsafe_count >= TARGET_UNSAFE:
        print(f"\nReached target of {TARGET_UNSAFE} unsafe statements — stopping.")
        break

    stats["total"] += 1
    original_safe = str(row["Statement"])
    explanation = str(row["Explanation"])
    bias_type = str(row["Type"])

    gen_prompt = build_gen_prompt(original_safe, explanation, bias_type)
    inputs = tokenizer(
        gen_prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_gen = tokenizer.decode(
        gen_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    unsafe_stmt, reasoning = extract_unsafe_content(raw_gen)

    if not unsafe_stmt or not reasoning:
        stats["skipped"] += 1
        continue
    stats["extracted"] += 1

    if not rule_validate(unsafe_stmt, reasoning, original_safe):
        stats["skipped"] += 1
        continue
    stats["rule_pass"] += 1

    if not judge_bias_introduction(original_safe, unsafe_stmt, bias_type):
        stats["skipped"] += 1
        continue
    stats["judge_pass"] += 1

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

    print(
        f"  [{stats['total']:>4}] unsafe accepted: {unsafe_count}  |  "
        f"rule_pass: {stats['rule_pass']}  judge_pass: {stats['judge_pass']}  skip: {stats['skipped']}"
    )
    if stats["total"] % 25 == 0 and final_data:
        pd.DataFrame(final_data).to_csv(CHECKPOINT_CSV, index=False)
        save_to_hf(final_data, config_name=HF_CONFIG)
        print(f"  Checkpoint saved: {len(final_data)} rows total (pairs)")

out_df = pd.DataFrame(final_data)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[4/4] Saved {len(out_df)} rows → {OUTPUT_CSV}")
print(f"  Safe   : {len(out_df[out_df['label'] == 'SAFE'])}")
print(f"  Unsafe : {len(out_df[out_df['label'] == 'UNSAFE'])}")
save_to_hf(final_data, config_name=HF_CONFIG)
print("Done.")
