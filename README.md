# GuardPEFT: Parameter‑Efficient Cross‑Domain Safety Alignment for LLMs

**Faran Taimoor Butt** – Thesis Project

## Abstract

Large language models often suffer from ethical misalignment, producing biased or unsafe outputs. GuardPEFT unifies reasoning‑based safety alignment with parameter‑efficient fine‑tuning (LoRA/DoRA) and activation steering. The framework generates structured outputs (safety label, violation categories, natural‑language critique) while using ≤1% of trainable parameters. We evaluate on in‑domain datasets (BiasMD, ToxiGen, Gretel, Aegis) and out‑of‑domain (ETHICS, PKU BeaverTails). Our best LoRA adapter (r=32) achieves 74.3% accuracy on a held‑out test set; activation steering further reduces over‑refusal by 23.8% on ETHICS.

## Installation

```bash
git clone https://github.com/yourusername/GuardPEFT.git
cd GuardPEFT
pip install -r requirements.txt
huggingface-cli login   

```
