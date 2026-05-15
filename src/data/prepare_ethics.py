import os
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm

from huggingface_hub import login
from src.utils.common import (clean_reasoning, extract_reasoning, get_device, load_model_and_tokenizer, print_stats, run_judge, save_to_hf, set_seed, clean_text, map_label, free_memory)
from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config = load_config("configs/data.yaml")
SEED = config["seed"]
HF_REPO = "faranbutt789/ethics-checkpoints"


BASE_URL = "https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data"
SAMPLE_PER_SPLIT = 1000  # rows sampled from each sub-dataset
OUTPUT_DIR = "data/ethics"


set_seed()
login_to_hf()
device = get_device()
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {device}")
print("\n[Model] Loading Qwen2.5-7B-Instruct …")
model, tokenizer = load_model_and_tokenizer()


# ═══════════════════════════════════════════════════════════════════════════
# SHARED JUDGE INFRA  (re-used by all four sub-datasets)
# ═══════════════════════════════════════════════════════════════════════════


def _generate(prompt: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.5, top_p=0.9
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def _run_loop(
    df: pd.DataFrame,
    generate_fn,
    validate_fn,
    user_prompt_str: str,
    categories_list: list,
    hf_config: str,
    checkpoint_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """Generic generation + validation loop shared by all sub-datasets."""
    final_data = []
    stats = {"total": 0, "rule_pass": 0, "judge_pass": 0, "rejected": 0}

    for _, row in tqdm(df.iterrows(), total=len(df)):
        stats["total"] += 1
        text = row["assistant_response"]
        label = row["label"]

        raw = generate_fn(text, label)
        reasoning = extract_reasoning(raw)
        reasoning = clean_reasoning(reasoning)

        if not reasoning or not validate_fn(reasoning, text, label):
            stats["rejected"] += 1
            continue
        stats["rule_pass"] += 1

        if not run_judge(text, label, reasoning, model, tokenizer, device):
            stats["rejected"] += 1
            continue
        stats["judge_pass"] += 1

        final_data.append(
            {
                "user_prompt": user_prompt_str,
                "assistant_response": text,
                "label": label,
                "categories": categories_list,
                "reasoning": reasoning,
                "source": f"ethics_{hf_config}",
            }
        )
        print_stats(stats)

        if stats["total"] % 50 == 0:
            pd.DataFrame(final_data).to_csv(checkpoint_csv, index=False)
            save_to_hf(final_data, config_name=hf_config)
            print(f"  Checkpoint: {len(final_data)} rows")

    out = pd.DataFrame(final_data)
    out.to_csv(output_csv, index=False)
    print(f"\n  Saved {len(out)} rows → {output_csv}")
    save_to_hf(final_data, config_name=hf_config)
    return out


def prepare_commonsense() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Sub-dataset 1/4: Commonsense")
    print("=" * 60)

    train_df = pd.read_csv(f"{BASE_URL}/commonsense/train.csv")
    test_df = pd.read_csv(f"{BASE_URL}/commonsense/test.csv")
    hard_df = pd.read_csv(f"{BASE_URL}/commonsense/test_hard.csv")
    print(f"  train={train_df.shape}  test={test_df.shape}  hard={hard_df.shape}")

    all_data = pd.concat([train_df, test_df, hard_df], ignore_index=True)

    processed = []
    for _, row in all_data.iterrows():
        text = clean_text(row["input"])
        if text is None:
            continue
        processed.append(
            {
                "assistant_response": text,
                "label": map_label(row["label"]),
            }
        )
    df = pd.DataFrame(processed).sample(SAMPLE_PER_SPLIT, random_state=SEED)

    def _gen(text, label):
        return _generate(f"""You are an expert in ethical reasoning.

Here are examples:

Statement: I helped a stranger carry groceries.
Label: SAFE
REASONING: Helping someone carry groceries is a harmless and considerate act that benefits others.

Statement: I stole money from my coworker's wallet.
Label: UNSAFE
REASONING: Stealing violates property rights and causes harm to another person.

Statement: I ignored a crying child who needed help.
Label: UNSAFE
REASONING: Ignoring someone in distress can lead to harm and shows a lack of basic moral responsibility.

Now analyze:

Statement: {text}
Label: {label}

RULES:
- One sentence only
- Be specific (mention harm, safety, fairness, legality, etc.)
- Do NOT be generic
- Do NOT repeat the statement
- Do NOT say "this is safe/unsafe"

FORMAT:
REASONING: <one sentence>
""")

    def _validate(reasoning, text, label):
        if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
            return False
        if text[:40].lower() in reasoning.lower():
            return False
        bad = [
            "this is safe",
            "this is unsafe",
            "no harm",
            "no unethical",
            "does not imply",
            "everyone is happy",
            "nothing wrong",
        ]
        if any(p in reasoning.lower() for p in bad):
            return False
        if label == "SAFE" and any(
            p in reasoning.lower()
            for p in [
                "causes harm",
                "poses a risk",
                "poses danger",
                "is illegal",
                "leads to harm",
            ]
        ):
            return False
        if label == "UNSAFE" and not any(
            w in reasoning.lower()
            for w in [
                "harm",
                "risk",
                "danger",
                "violate",
                "abuse",
                "illegal",
                "unfair",
                "dishonest",
                "distress",
                "damage",
                "loss",
                "conflict",
            ]
        ):
            return False
        return True

    return _run_loop(
        df,
        _gen,
        _validate,
        user_prompt_str="Is the following action acceptable?",
        categories_list=["moral_reasoning"],
        hf_config="commonsense",
        checkpoint_csv=f"{OUTPUT_DIR}/commonsense_checkpoint.csv",
        output_csv=f"{OUTPUT_DIR}/commonsense_prepared.csv",
    )


def prepare_justice() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Sub-dataset 2/4: Justice")
    print("=" * 60)

    raw_df = pd.read_csv(f"{BASE_URL}/justice/train.csv")
    print(f"  Loaded: {raw_df.shape}")

    processed = []
    for _, row in raw_df.iterrows():
        text = clean_text(row["scenario"])
        if text is None:
            continue
        processed.append(
            {
                "assistant_response": text,
                "label": map_label(row["label"]),
            }
        )
    df = pd.DataFrame(processed).sample(SAMPLE_PER_SPLIT, random_state=SEED)

    def _gen(text, label):
        return _generate(f"""You are an expert in fairness and justice reasoning.

Here are examples:

Statement: I deserve to be paid for the work I completed.
Label: SAFE
REASONING: Receiving payment for completed work is fair and aligns with agreed compensation.

Statement: I deserve a luxury car from my employer for doing my regular duties.
Label: UNSAFE
REASONING: Expecting excessive rewards beyond agreed compensation is unfair and disproportionate.

Statement: I deserve respect from my coworkers because I treat them respectfully.
Label: SAFE
REASONING: Mutual respect is a fair expectation based on reciprocal behavior.

Now analyze:

Statement: {text}
Label: {label}

RULES:
- One sentence only
- Focus on fairness, entitlement, proportionality, or rights
- Do NOT repeat the statement
- Do NOT say "this is safe/unsafe"

FORMAT:
REASONING: <one sentence>
""")

    def _validate(reasoning, text, label):
        if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
            return False
        if text[:40].lower() in reasoning.lower():
            return False
        bad = [
            "this is safe",
            "this is unsafe",
            "no harm",
            "no unethical",
            "does not imply",
            "everyone is happy",
            "nothing wrong",
        ]
        if any(p in reasoning.lower() for p in bad):
            return False
        if label == "SAFE" and not any(
            p in reasoning.lower()
            for p in ["fair", "justified", "reasonable", "deserved", "appropriate"]
        ):
            return False
        if label == "UNSAFE" and not any(
            w in reasoning.lower()
            for w in [
                "unfair",
                "disproportionate",
                "excessive",
                "unreasonable",
                "not justified",
                "undeserved",
                "inappropriate",
            ]
        ):
            return False
        return True

    return _run_loop(
        df,
        _gen,
        _validate,
        user_prompt_str="Is this claim fair or justified?",
        categories_list=["justice_reasoning"],
        hf_config="justice",
        checkpoint_csv=f"{OUTPUT_DIR}/justice_checkpoint.csv",
        output_csv=f"{OUTPUT_DIR}/justice_prepared.csv",
    )


def prepare_virtue() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Sub-dataset 3/4: Virtue")
    print("=" * 60)

    raw_df = pd.read_csv(f"{BASE_URL}/virtue/train.csv")
    print(f"  Loaded: {raw_df.shape}")

    # Balanced 500 SAFE + 500 UNSAFE
    safe_df = raw_df[raw_df["label"] == 0].sample(500, random_state=SEED)
    unsafe_df = raw_df[raw_df["label"] == 1].sample(500, random_state=SEED)
    balanced = pd.concat([safe_df, unsafe_df])

    processed = []
    for _, row in balanced.iterrows():
        text = clean_text(row["scenario"])
        if text is None:
            continue
        processed.append(
            {
                "assistant_response": text,
                "label": map_label(row["label"]),
            }
        )
    df = pd.DataFrame(processed).sample(
        min(SAMPLE_PER_SPLIT, len(processed)), random_state=SEED
    )

    def _gen(text, label):
        return _generate(f"""You are an expert in virtue ethics.

Virtue ethics evaluates a person's character traits rather than just actions.
- MUST mention a specific character trait (e.g., honesty, kindness, integrity)
- Avoid generic phrases like "good character"

Here are examples:

Statement: She always tells the truth even when it is difficult.
Label: SAFE
REASONING: Consistently telling the truth reflects honesty and strong moral character.

Statement: He constantly boasts about his achievements to impress others.
Label: UNSAFE
REASONING: Boasting excessively reflects arrogance and a lack of humility.

Statement: She helps others without expecting anything in return.
Label: SAFE
REASONING: Helping others selflessly demonstrates kindness and generosity.

Now analyze:

Statement: {text}
Label: {label}

RULES:
- One sentence only
- Focus on character traits (e.g., honesty, arrogance, kindness, selfishness)
- Do NOT repeat the statement
- Do NOT say "this is safe/unsafe"

FORMAT:
REASONING: <one sentence>
""")

    VIRTUE_WORDS = [
        "honest",
        "kind",
        "generous",
        "humble",
        "responsible",
        "disciplined",
        "compassionate",
        "respectful",
        "integrity",
    ]
    VICE_WORDS = [
        "arrogant",
        "selfish",
        "greedy",
        "dishonest",
        "lazy",
        "rude",
        "disrespectful",
        "impulsive",
        "attention-seeking",
    ]

    def _validate(reasoning, text, label):
        if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
            return False
        if text[:40].lower() in reasoning.lower():
            return False
        bad = ["this is safe", "this is unsafe", "nothing wrong", "everyone is happy"]
        if any(p in reasoning.lower() for p in bad):
            return False
        if label == "SAFE" and not any(w in reasoning.lower() for w in VIRTUE_WORDS):
            return False
        if label == "UNSAFE" and not any(w in reasoning.lower() for w in VICE_WORDS):
            return False
        return True

    return _run_loop(
        df,
        _gen,
        _validate,
        user_prompt_str="Does this behavior reflect good character?",
        categories_list=["virtue_ethics"],
        hf_config="virtue",
        checkpoint_csv=f"{OUTPUT_DIR}/virtue_checkpoint.csv",
        output_csv=f"{OUTPUT_DIR}/virtue_prepared.csv",
    )


def prepare_deontology() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Sub-dataset 4/4: Deontology")
    print("=" * 60)

    raw_df = pd.read_csv(f"{BASE_URL}/deontology/train.csv")
    print(f"  Loaded: {raw_df.shape}")

    processed = []
    for _, row in raw_df.iterrows():
        scenario = clean_text(row["scenario"])
        excuse = clean_text(row["excuse"])
        if scenario is None or excuse is None:
            continue
        processed.append(
            {
                "assistant_response": f"{scenario} {excuse}",
                "label": map_label(row["label"]),
            }
        )
    df = pd.DataFrame(processed).sample(SAMPLE_PER_SPLIT, random_state=SEED)

    def _gen(text, label):
        return _generate(f"""You are an expert in deontological ethics.

Deontology evaluates whether actions follow moral rules, duties, or obligations.

Examples:

Statement: I lied to avoid getting in trouble.
Label: UNSAFE
REASONING: Lying violates the moral duty to be truthful regardless of consequences.

Statement: I kept my promise even when it was inconvenient.
Label: SAFE
REASONING: Keeping promises fulfills moral obligations and respects commitments.

Statement: I ignored workplace rules because they were annoying.
Label: UNSAFE
REASONING: Ignoring established rules violates duties expected in structured environments.

Now analyze:

Statement: {text}
Label: {label}

RULES:
- One sentence only
- Focus on duties, rules, obligations, or principles
- Do NOT repeat the statement
- Do NOT say "this is safe/unsafe"

FORMAT:
REASONING: <one sentence>
""")

    DUTY_WORDS = [
        "duty",
        "obligation",
        "rule",
        "principle",
        "responsibility",
        "moral law",
        "commitment",
    ]
    VIOLATION_WORDS = ["violates", "breaks", "ignores", "fails", "disregards"]

    def _validate(reasoning, text, label):
        if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
            return False
        if text[:40].lower() in reasoning.lower():
            return False
        bad = ["this is safe", "this is unsafe", "nothing wrong", "no harm"]
        if any(p in reasoning.lower() for p in bad):
            return False
        if label == "SAFE" and not any(w in reasoning.lower() for w in DUTY_WORDS):
            return False
        if label == "UNSAFE" and not any(
            w in reasoning.lower() for w in VIOLATION_WORDS + DUTY_WORDS
        ):
            return False
        return True

    return _run_loop(
        df,
        _gen,
        _validate,
        user_prompt_str="Is this action morally permissible based on rules or duties?",
        categories_list=["deontology"],
        hf_config="deontology",
        checkpoint_csv=f"{OUTPUT_DIR}/deontology_checkpoint.csv",
        output_csv=f"{OUTPUT_DIR}/deontology_prepared.csv",
    )


if __name__ == "__main__":
    commonsense_df = prepare_commonsense()
    justice_df = prepare_justice()
    virtue_df = prepare_virtue()
    deontology_df = prepare_deontology()

    combined = pd.concat(
        [commonsense_df, justice_df, virtue_df, deontology_df],
        ignore_index=True,
    )
    combined_path = f"{OUTPUT_DIR}/ethics_all_prepared.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined ethics dataset: {combined.shape} → {combined_path}")
    save_to_hf(combined.to_dict("records"), config_name="ethics_combined")
