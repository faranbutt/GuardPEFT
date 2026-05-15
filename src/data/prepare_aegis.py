import os
import re
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from huggingface_hub import login
from src.utils.common import (clean_reasoning, extract_reasoning, get_device, load_model_and_tokenizer, print_stats, run_judge, save_to_hf, set_seed, clean_text, map_label, free_memory)
from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config = load_config("configs/data.yaml")
SEED = config["seed"]
HF_REPO = "faranbutt789/ethics-checkpoints"


TARGET = config["aegis"]["target"]
CHECKPOINT_N = config["aegis"]["checkpoint_n"]
RECENT_WINDOW = config["aegis"]["recent_window"]
OUTPUT_DIR = os.path.dirname(config["paths"]["aegis_prepared"])
CHECKPOINT_CSV = config["paths"]["aegis_checkpoint"]
OUTPUT_CSV = config["paths"]["aegis_prepared"]
HF_CONFIG = "aegis"


set_seed()
login_to_hf()
device = get_device()
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {device}")

print("\n[1/5] Loading Aegis dataset …")
dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
df = dataset["train"].to_pandas()

df = df.dropna(subset=["violated_categories"])
df = df.replace("", np.nan).dropna(subset=["violated_categories"])
df = df[["prompt", "response", "violated_categories", "response_label"]].dropna()

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
def _balance(df_, label_col, cap):
    over = df_.groupby(label_col).filter(lambda x: len(x) > cap)
    under = df_.groupby(label_col).filter(lambda x: len(x) <= cap)
    return (
        pd.concat([over.groupby(label_col).sample(n=cap, random_state=SEED), under])
        .sample(frac=1)
        .reset_index(drop=True)
    )


df_balanced = _balance(df_final, "response_label", cap_size=7000)

# Even category distribution → ~5000 total
categories = df_balanced["violated_categories"].unique()
target_per_cat = 5000 // len(categories)


def _get_best(group):
    return (
        group
        if len(group) <= target_per_cat
        else group.sample(n=target_per_cat, random_state=SEED)
    )


df_top = df_balanced.groupby("violated_categories", group_keys=False).apply(_get_best)
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
print(
    f"Working set: {df_top.shape}  |  labels: {df_top['label'].value_counts().to_dict()}"
)

# ─────────────────────────────────────────────
# 2. Load model
# ─────────────────────────────────────────────
print("\n[2/5] Loading Qwen2.5-7B-Instruct …")
model, tokenizer = load_model_and_tokenizer()

# Token IDs needed for judge
YES_ID = tokenizer.encode("YES", add_special_tokens=False)[0]
NO_ID = tokenizer.encode("NO", add_special_tokens=False)[0]

# ─────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────
GENERATION_SYSTEM = (
    "You are a precise content-safety annotator. "
    "Your only job is to write ONE factual sentence explaining why an AI response "
    "received a given safety label. Be specific — name the exact harm or safe behavior. "
    "Never be generic. Never repeat the prompt or response text."
)

GENERATION_USER = """\
Label the following AI interaction with exactly ONE sentence of reasoning.

=== INTERACTION ===
User asked: {user_prompt}
AI responded: {assistant_response}
Safety category: {category}
Label assigned: {label_upper}

=== GOOD EXAMPLES ===
[SAFE] User asked about making weapons. AI said it won't help.
→ The assistant withholds dangerous manufacturing details, preventing potential physical harm.

[UNSAFE] User asked how to evade police. AI gave a detailed evasion plan.
→ The assistant supplies actionable guidance on evading law enforcement, actively supporting criminal activity.

[SAFE] User expressed suicidal thoughts. AI offered crisis resources.
→ The response prioritises the user's wellbeing by directing them to professional mental health support.

[UNSAFE] User asked for hate speech. AI wrote a slur-filled post.
→ The assistant produces explicit content targeting a protected group, violating basic dignity and safety policies.

=== YOUR TASK ===
Write ONE sentence for the interaction above.
Rules:
- Start with "The assistant" or "The response"
- Name the specific harm category or safe action
- Do NOT use "step-by-step"
- Do NOT copy phrases from the user prompt or AI response
- Do NOT include any label word (SAFE / UNSAFE)
- Do NOT add any prefix like "Reasoning:" or "Output:"

Sentence:"""

TEMP_SCHEDULE = [0.55, 0.70, 0.80, 0.65, 0.75]


def _build_prompt(user_prompt, assistant_response, category, label):
    user_text = GENERATION_USER.format(
        user_prompt=user_prompt[:300],
        assistant_response=assistant_response[:400],
        category=category,
        label_upper=label.upper(),
    )
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _generate_raw(user_prompt, assistant_response, category, label, attempt=0):
    prompt = _build_prompt(user_prompt, assistant_response, category, label)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1200
    ).to(device)
    temp = TEMP_SCHEDULE[attempt % len(TEMP_SCHEDULE)]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=55,
            do_sample=True,
            temperature=temp,
            top_p=0.90,
            repetition_penalty=1.20,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def _extract_and_clean(raw: str) -> str | None:
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
        if not line:
            continue
        if re.match(
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


SAFE_CUES = [
    "declines",
    "refuses",
    "withholds",
    "avoids",
    "does not provide",
    "does not assist",
    "refrains",
    "redirects",
    "refers",
    "offers support",
    "provides crisis",
    "recommends professional",
    "correctly rejects",
    "appropriately declines",
    "prioritises",
    "prioritizes",
    "encourages",
    "does not engage",
    "cannot assist",
    "will not",
    "won't",
    "can't",
]
UNSAFE_CUES = [
    "provides",
    "supplies",
    "facilitates",
    "enables",
    "instructs",
    "produces",
    "generates",
    "contains",
    "delivers",
    "offers guidance",
    "assists with",
    "details how",
    "explains how",
    "shares",
    "gives",
    "spreads",
    "promotes",
    "glorifies",
    "actively assists",
    "actively supports",
    "actively enables",
]
TEMPLATED = [
    "step-by-step instructions",
    "step-by-step",
    "this response",
    "this is safe",
    "this is unsafe",
    "user's request",
]
CAT_KW = {
    "controlled": [
        "drug",
        "substance",
        "narcotic",
        "illegal",
        "synthesiz",
        "manufactur",
        "chemical",
    ],
    "hate": ["hate", "discriminat", "slur", "derogat", "prejudic", "group", "target"],
    "violence": [
        "violen",
        "injur",
        "harm",
        "attack",
        "assault",
        "kill",
        "murder",
        "physical",
    ],
    "self harm": [
        "self-harm",
        "suicid",
        "self harm",
        "crisis",
        "mental health",
        "wellbeing",
    ],
    "sexual": ["sexual", "explicit", "consent", "minor", "graphic"],
    "weapons": ["weapon", "firearm", "explosive", "bomb", "ammunit", "manufactur"],
    "fraud": ["fraud", "scam", "deceiv", "fake", "impersonat", "phish"],
    "criminal": ["illegal", "criminal", "law", "arrest", "felony", "evad"],
    "stalking": ["stalk", "track", "surveil", "consent", "privacy", "harass"],
    "harassment": ["harass", "threaten", "intimidat", "bully", "target"],
    "malware": ["malware", "virus", "exploit", "hack", "phish", "inject"],
}


def _rule_validate(reasoning: str, label: str, category: str) -> tuple[bool, int]:
    if not reasoning:
        return False, 0
    r, score = reasoning.lower(), 0
    wc = len(reasoning.split())
    if 8 <= wc <= 35:
        score += 2
    if len(re.findall(r"[.!?]", reasoning)) <= 1:
        score += 1
    if re.match(r"^The (assistant|response)", reasoning, re.IGNORECASE):
        score += 1
    if not any(t in r for t in TEMPLATED):
        score += 1
    label_up = label.upper()
    if label_up == "SAFE":
        if any(c in r for c in SAFE_CUES):
            score += 3
        else:
            score -= 1
    else:
        if any(c in r for c in UNSAFE_CUES):
            score += 3
        else:
            score -= 1
    if label_up == "SAFE" and any(c in r for c in UNSAFE_CUES[:8]):
        score -= 2
    if label_up == "UNSAFE" and any(c in r for c in SAFE_CUES[:6]):
        score -= 1
    for ck, kws in CAT_KW.items():
        if ck in category.lower():
            if any(kw in r for kw in kws):
                score += 1
            break
    return (score >= 6), score


# Aegis-specific judge (takes full row context)
AEGIS_JUDGE_PROMPTS = [
    (
        "Does this reasoning correctly explain why the assistant's response is {label_upper}?\n"
        "User: {user_prompt}\nAssistant: {assistant_response}\n"
        "Category: {category}\nReasoning: {reasoning}\n"
        "Answer YES if accurate and specific. Answer NO if wrong, generic, or contradicts the label.\n"
        "Answer (YES or NO):"
    ),
    (
        "Is this reasoning specific to the actual assistant response, or too generic?\n"
        "Assistant response: {assistant_response}\nReasoning: {reasoning}\n"
        "Answer YES if specific. Answer NO if generic.\nAnswer (YES or NO):"
    ),
    (
        "Label: {label_upper}. Does the reasoning support this label without contradiction?\n"
        "Assistant response: {assistant_response}\nReasoning: {reasoning}\n"
        "Answer YES if consistent. Answer NO if contradicts.\nAnswer (YES or NO):"
    ),
]


def _run_aegis_judge(
    user_prompt, assistant_response, label, category, reasoning
) -> bool:
    votes, YES_MARGIN = 0, 0.5
    label_up = label.upper()
    for template in AEGIS_JUDGE_PROMPTS:
        prompt = template.format(
            user_prompt=user_prompt[:250],
            assistant_response=assistant_response[:350],
            category=category,
            label_upper=label_up,
            reasoning=reasoning,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=768
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        if (logits[YES_ID] - logits[NO_ID]).item() > YES_MARGIN:
            votes += 1
    return votes >= 2


def _is_duplicate(reasoning: str, recent) -> bool:
    def _norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()

    nw = set(_norm(reasoning).split())
    for prev in recent:
        pw = set(_norm(prev).split())
        if pw and len(nw & pw) / len(nw | pw) > 0.70:
            return True
    return False


def _generate_best(row, attempts=5, recent=None) -> str | None:
    up = str(row["user_prompt"])
    ar = str(row["assistant_response"])
    lab = str(row["label"])
    cat = str(row["categories"])
    best_cand, best_score = None, -999

    for attempt in range(attempts):
        raw = _generate_raw(up, ar, cat, lab, attempt)
        reasoning = _extract_and_clean(raw)
        if not reasoning:
            continue
        if recent and _is_duplicate(reasoning, recent):
            continue
        is_valid, score = _rule_validate(reasoning, lab, cat)
        if score > best_score:
            best_score, best_cand = score, reasoning
        if not is_valid:
            continue
        if _run_aegis_judge(up, ar, lab, cat, reasoning):
            return reasoning

    return best_cand if (best_cand and best_score >= 5) else None


print(f"\n[3/5] Generating reasoning (target={TARGET}) …")
df_shuffled = df_top.sample(frac=1, random_state=SEED).reset_index(drop=True)

final_data = []
stats = {"total": 0, "accepted": 0, "rejected": 0}
recent_q = deque(maxlen=RECENT_WINDOW)

for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled)):
    if stats["accepted"] >= TARGET:
        print(f"\nReached {TARGET} accepted rows — stopping.")
        break

    stats["total"] += 1
    reasoning = _generate_best(row, attempts=5, recent=recent_q)

    if reasoning:
        out = row.to_dict()
        out["reasoning"] = reasoning
        out["source"] = "aegis"
        final_data.append(out)
        recent_q.append(reasoning)
        stats["accepted"] += 1
    else:
        stats["rejected"] += 1

    if stats["total"] % 10 == 0:
        rate = stats["accepted"] / stats["total"] * 100
        print(
            f"  [{stats['total']:>5}] Accepted: {stats['accepted']}  "
            f"Rejected: {stats['rejected']}  Rate: {rate:.1f}%"
        )

    if stats["total"] % CHECKPOINT_N == 0 and final_data:
        pd.DataFrame(final_data).to_csv(CHECKPOINT_CSV, index=False)
        save_to_hf(final_data, config_name=HF_CONFIG)
        print(f"  Checkpoint: {len(final_data)} rows")


out_df = pd.DataFrame(final_data)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(out_df)} rows → {OUTPUT_CSV}")
save_to_hf(final_data, config_name=HF_CONFIG)
