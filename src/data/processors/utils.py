import re

# Aegis shared cues and validation
AEGIS_SAFE_CUES = [
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
AEGIS_UNSAFE_CUES = [
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
AEGIS_TEMPLATED = [
    "step-by-step instructions",
    "step-by-step",
    "this response",
    "this is safe",
    "this is unsafe",
    "user's request",
]
AEGIS_CAT_KW = {
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


def aegis_rule_validate(reasoning: str, label: str, category: str) -> tuple[bool, int]:
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
    if not any(t in r for t in AEGIS_TEMPLATED):
        score += 1

    label_up = label.upper()
    if label_up == "SAFE":
        if any(c in r for c in AEGIS_SAFE_CUES):
            score += 3
        else:
            score -= 1
    else:
        if any(c in r for c in AEGIS_UNSAFE_CUES):
            score += 3
        else:
            score -= 1

    if label_up == "SAFE" and any(c in r for c in AEGIS_UNSAFE_CUES[:8]):
        score -= 2
    if label_up == "UNSAFE" and any(c in r for c in AEGIS_SAFE_CUES[:6]):
        score -= 1

    for ck, kws in AEGIS_CAT_KW.items():
        if ck in category.lower():
            if any(kw in r for kw in kws):
                score += 1
            break
    return (score >= 6), score


# Gretel validation
GRETEL_SAFE_CUES = [
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
GRETEL_UNSAFE_CUES = [
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


def gretel_is_valid(reasoning: str, label: str) -> bool:
    if not reasoning or len(reasoning.split()) < 4:
        return False
    r = reasoning.lower()
    if label == "SAFE" and not any(c in r for c in GRETEL_SAFE_CUES):
        return False
    if label == "UNSAFE" and not any(c in r for c in GRETEL_UNSAFE_CUES):
        return False
    return True


# PKU validation
PKU_SAFE_CUES = [
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
    "prioritises",
    "prioritizes",
]
PKU_UNSAFE_CUES = [
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
    "glorifies",
    "gives",
]


def pku_is_valid(reasoning: str, label: str) -> bool:
    if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
        return False
    r = reasoning.lower()
    if not re.match(r"^the (assistant|response)", reasoning, re.IGNORECASE):
        return False
    if label == "SAFE" and not any(c in r for c in PKU_SAFE_CUES):
        return False
    if label == "UNSAFE" and not any(c in r for c in PKU_UNSAFE_CUES):
        return False
    return True


# Toxigen validation
TOXIGEN_UNSAFE_KW = [
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
TOXIGEN_SAFE_KW = [
    "neutral",
    "harmless",
    "non-offensive",
    "respectful",
    "benign",
    "civil",
]


def toxigen_is_valid_reasoning(reasoning: str, text: str, label: str) -> bool:
    if not reasoning or len(reasoning.split()) < 6 or len(reasoning.split()) > 40:
        return False
    if text[:40].lower() in reasoning.lower():
        return False
    if label == "UNSAFE":
        return any(kw in reasoning.lower() for kw in TOXIGEN_UNSAFE_KW)
    return any(kw in reasoning.lower() for kw in TOXIGEN_SAFE_KW)
