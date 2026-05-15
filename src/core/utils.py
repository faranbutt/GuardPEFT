import gc
import random
import re

import numpy as np
import pandas as pd
import torch

from .config import SEED


def set_seed(seed: int = SEED) -> None:
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Returns the available device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_memory() -> None:
    """Cleans up memory by collecting garbage and clearing CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clean_text(text) -> str | None:
    """Normalizes text by removing extra whitespace."""
    if pd.isna(text):
        return None
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < 3:
        return None
    return text


def map_label(label) -> str:
    """Maps numeric label: 1 → UNSAFE, 0 → SAFE."""
    try:
        return "UNSAFE" if int(label) == 1 else "SAFE"
    except (ValueError, TypeError):
        return str(label).upper().strip()


def clean_reasoning(reasoning: str | None) -> str | None:
    """Normalizes and validates a raw reasoning string."""
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


def extract_reasoning(raw_text: str) -> str | None:
    """Pulls the first valid REASONING: line from model output."""
    if not raw_text:
        return None
    raw_text = re.sub(r"```[\s\S]*?```", "", raw_text)
    matches = re.findall(r"REASONING:\s*(.*?)(?=\n|$)", raw_text, re.DOTALL)
    for r in matches:
        r = r.strip()
        if len(r.split()) >= 5:
            return r
    return None


def print_stats(stats: dict) -> None:
    """Prints generation statistics."""
    total = max(stats.get("total", 0), 1)
    accepted = stats.get("judge_pass", 0)
    rejected = stats.get("rejected", 0)
    print(
        f"  Accepted: {accepted}  |  "
        f"Rejected: {rejected}  |  "
        f"Rate: {accepted / total * 100:.1f}%"
    )
