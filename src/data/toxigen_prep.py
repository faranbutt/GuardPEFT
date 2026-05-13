import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils.constants import SEED
from ..utils.helpers import set_seed

set_seed()


def prepare_toxigen(
    output_csv="data/toxigen/toxigen_processed.csv", use_generation=True
):
    """
    Load ToxiGen, balance classes, generate reasoning using Qwen2.5-7B,
    and save processed DataFrame.
    If output_csv already exists and use_generation=False, load and return it.
    """
    import os

    if os.path.exists(output_csv) and not use_generation:
        print(f"Loading existing ToxiGen dataset from {output_csv}")
        return pd.read_csv(output_csv)

    print("Loading raw ToxiGen dataset...")
    toxigen_ds = load_dataset("toxigen/toxigen-data", "annotated")
    toxigen_df = toxigen_ds["train"].to_pandas()

    toxigen_df = toxigen_df.rename(
        columns={"text": "assistant_response", "predicted_group": "categories"}
    )

    def map_toxicity(score):
        return "UNSAFE" if score >= 3 else "SAFE"

    toxigen_df["label"] = toxigen_df["toxicity_human"].apply(map_toxicity)
    toxigen_df["categories"] = toxigen_df["categories"].apply(
        lambda x: [str(x).lower()]
    )
    toxigen_df = toxigen_df[["assistant_response", "categories", "label"]]
    toxigen_df = toxigen_df.dropna()
    toxigen_df["assistant_response"] = (
        toxigen_df["assistant_response"].astype(str).str.strip()
    )
    toxigen_df = toxigen_df[toxigen_df["assistant_response"] != ""]

    # Balance classes
    min_samples = min(
        toxigen_df["label"].value_counts()["SAFE"],
        toxigen_df["label"].value_counts()["UNSAFE"],
    )
    safe_df = toxigen_df[toxigen_df["label"] == "SAFE"].sample(
        n=min_samples, random_state=SEED
    )
    unsafe_df = toxigen_df[toxigen_df["label"] == "UNSAFE"].sample(
        n=min_samples, random_state=SEED
    )
    toxigen_balanced = (
        pd.concat([safe_df, unsafe_df])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    print(f"Balanced dataset: {toxigen_balanced['label'].value_counts().to_dict()}")

    # Sample a subset for reasoning generation (to keep manageable)
    SAMPLE_SIZE = 2000
    toxigen_sample = toxigen_balanced.sample(
        min(SAMPLE_SIZE, len(toxigen_balanced)), random_state=SEED
    )

    # Load model for reasoning generation
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    def generate_reasoning(text, label):
        prompt = f"""
        You are an expert in content safety and toxicity detection.

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
            output = model.generate(
                **inputs, max_new_tokens=80, temperature=0.5, top_p=0.9
            )
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    def extract_reasoning(raw_text):
        raw_text = re.sub(r"```[\s\S]*?```", "", raw_text)
        matches = re.findall(r"REASONING:\s*(.*?)(?=\n|$)", raw_text, re.DOTALL)
        for r in matches:
            r = r.strip()
            if len(r.split()) >= 5:
                return r
        return None

    def clean_reasoning(reasoning):
        if not reasoning:
            return None
        reasoning = re.sub(r"REASONING:\s*", "", reasoning)
        reasoning = re.sub(r"[`]+", "", reasoning)
        reasoning = re.sub(r"\.{2,}", ".", reasoning)
        words = reasoning.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.6:
                return None
        reasoning = reasoning.split(".")[0].strip()
        if len(reasoning.split()) < 6:
            return None
        return reasoning + "."

    def is_valid_reasoning(reasoning, text, label):
        if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
            return False
        if text[:40].lower() in reasoning.lower():
            return False
        unsafe_keywords = [
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
        safe_keywords = [
            "neutral",
            "harmless",
            "non-offensive",
            "respectful",
            "benign",
            "civil",
        ]
        if label == "UNSAFE":
            return any(kw in reasoning.lower() for kw in unsafe_keywords)
        else:
            return any(kw in reasoning.lower() for kw in safe_keywords)

    # Judge prompts
    JUDGE_PROMPTS = [
        "Does this reasoning mention a specific harm, risk, or benefit that justifies the {label} label?\nStatement: {text}\nReasoning: {reasoning}\nAnswer only YES or NO:",
        "Is this reasoning logically consistent with labeling the statement as {label}?\nStatement: {text}\nReasoning: {reasoning}\nAnswer only YES or NO:",
        "Is this reasoning specific (not generic) and directly relevant to the statement?\nStatement: {text}\nReasoning: {reasoning}\nAnswer only YES or NO:",
    ]

    def run_judge(text, label, reasoning):
        yes_id = tokenizer.encode("YES", add_special_tokens=False)[0]
        no_id = tokenizer.encode("NO", add_special_tokens=False)[0]
        MARGIN = 2.5
        votes = 0
        for template in JUDGE_PROMPTS:
            prompt = template.format(text=text, label=label, reasoning=reasoning)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            p_yes = next_token_logits[yes_id].item()
            p_no = next_token_logits[no_id].item()
            gap = p_no - p_yes
            if gap < MARGIN:
                votes += 1
        return votes >= 2

    # Generate reasoning for each sample
    final_data = []
    stats = {"total": 0, "rule_pass": 0, "judge_pass": 0, "rejected": 0}

    for idx, row in tqdm(
        toxigen_sample.iterrows(),
        total=len(toxigen_sample),
        desc="Generating ToxiGen reasoning",
    ):
        stats["total"] += 1
        text = row["assistant_response"]
        label = row["label"]
        categories = row["categories"]

        raw = generate_reasoning(text, label)
        reasoning = extract_reasoning(raw)
        reasoning = clean_reasoning(reasoning)

        if not reasoning:
            stats["rejected"] += 1
            continue

        if not is_valid_reasoning(reasoning, text, label):
            stats["rejected"] += 1
            continue
        stats["rule_pass"] += 1

        if not run_judge(text, label, reasoning):
            stats["rejected"] += 1
            continue

        stats["judge_pass"] += 1
        final_data.append(
            {
                "user_prompt": "Continue the following text in a natural way:",
                "assistant_response": text,
                "label": label,
                "categories": str(categories),
                "reasoning": reasoning,
            }
        )

        if stats["total"] % 50 == 0:
            print(f"Progress: {stats['judge_pass']}/{stats['total']} accepted")

    result_df = pd.DataFrame(final_data)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved ToxiGen dataset with {len(result_df)} rows to {output_csv}")
    return result_df
