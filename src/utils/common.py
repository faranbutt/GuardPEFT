import gc
import random
import re
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .hf_utils import get_hf_token
from huggingface_hub import login

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()

def load_model_and_tokenizer(model_name: str):
    """Load model in 4-bit quantisation."""
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
    return model, tokenizer

def clean_text(text) -> str | None:
    if pd.isna(text):
        return None
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < 3:
        return None
    return text

def map_label(label) -> str:
    """Map numeric label: 1 → UNSAFE, 0 → SAFE."""
    return "UNSAFE" if int(label) == 1 else "SAFE"

def extract_reasoning(raw_text: str) -> str | None:
    """Pull the first valid REASONING: line from model output."""
    if not raw_text:
        return None
    raw_text = re.sub(r"```[\s\S]*?```", "", raw_text)
    matches = re.findall(r"REASONING:\s*(.*?)(?=\n|$)", raw_text, re.DOTALL)
    for r in matches:
        r = r.strip()
        if len(r.split()) >= 5:
            return r
    return None

def clean_reasoning(reasoning: str | None) -> str | None:
    """Normalise and validate a raw reasoning string."""
    if not reasoning:
        return None
    reasoning = re.sub(r"REASONING:\s*", "", reasoning)
    reasoning = re.sub(r"[`]+", "", reasoning)
    reasoning = re.sub(r"\.{2,}", ".", reasoning)
    words = reasoning.split()
    if not words:
        return None
    if len(set(words)) / len(words) < 0.6:  # too repetitive
        return None
    reasoning = reasoning.split(".")[0].strip()
    if len(reasoning.split()) < 6:
        return None
    return reasoning + "."

def run_judge(
    text: str,
    label: str,
    reasoning: str,
    model,
    tokenizer,
    device: torch.device,
    margin: float = 2.5,
    votes_needed: int = 2,
) -> bool:
    JUDGE_PROMPTS = [
        (
            "Does this reasoning mention a specific harm, risk, or benefit that justifies "
            "the {label} label?\nStatement: {text}\nReasoning: {reasoning}\n"
            "Answer only YES or NO:"
        ),
        (
            "Is this reasoning logically consistent with labeling the statement as {label}?\n"
            "Statement: {text}\nReasoning: {reasoning}\nAnswer only YES or NO:"
        ),
        (
            "Is this reasoning specific (not generic) and directly relevant to the statement?\n"
            "Statement: {text}\nReasoning: {reasoning}\nAnswer only YES or NO:"
        ),
    ]

    yes_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_id = tokenizer.encode("NO", add_special_tokens=False)[0]
    votes = 0

    for template in JUDGE_PROMPTS:
        prompt = template.format(text=text, label=label, reasoning=reasoning)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        p_yes = logits[yes_id].item()
        p_no = logits[no_id].item()
        if (p_no - p_yes) < margin:
            votes += 1

    return votes >= votes_needed

def save_to_hf(
    data: list[dict],
    repo_name: str,
    config_name: str | None = None,
) -> None:
    if not data:
        print("  [HF] Nothing to push.")
        return
    
    # Try to login first
    token = get_hf_token()
    if token:
        login(token)
        
    ds = Dataset.from_pandas(pd.DataFrame(data))
    kwargs = {"private": True}
    if config_name:
        kwargs["config_name"] = config_name
    try:
        ds.push_to_hub(repo_name, **kwargs)
        print(
            f"  [HF] Pushed {len(data)} rows → {repo_name} ({config_name or 'default'})"
        )
    except Exception as exc:
        print(f"  [HF] Push failed: {exc}")

def print_stats(stats: dict) -> None:
    total = max(stats.get("total", 1), 1)
    print(
        f"  Accepted: {stats.get('judge_pass', stats.get('accepted', 0))}  |  "
        f"Rejected: {stats.get('rejected', 0)}  |  "
        f"Rate: {stats.get('judge_pass', stats.get('accepted', 0)) / total * 100:.1f}%"
    )
