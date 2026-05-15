import torch

from .utils import get_device

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


def run_judge(
    text: str,
    label: str,
    reasoning: str,
    model,
    tokenizer,
    margin: float = 2.5,
    votes_needed: int = 2,
) -> bool:
    """Runs a multi-prompt voting judge to validate reasoning quality."""
    device = get_device()
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
