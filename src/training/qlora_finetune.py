from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config_train = load_config('configs/train.yaml')
config_data = load_config('configs/data.yaml')
# Fine-tuning Qwen3-1.7B with QLoRA on the small dataset (train set containing BiasMD, ToxiGen, Aegies AI, Gretel AI)
import gc
import json
import multiprocessing
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, train_on_responses_only


HUB_BASE = "faranbutt789/guardpeft-qwen3"
SEED = config_train["training"]["seed"]


SMALL_DATASET_MODE = config_train["dataset"]["small_dataset_mode"]
TRAIN_SAMPLES = config_train["dataset"]["train_samples"]
VAL_SAMPLES = config_train["dataset"]["val_samples"]

DATA_ROOT = "data/small_dataset"

TRAIN_HP = config_train["training"]
TRAIN_HP["model_name"] = config_train["model"]["name"]
TRAIN_HP["max_seq_length"] = config_train["model"]["max_seq_length"]

SYSTEM_PROMPT = """You are GuardPEFT, a precise content-safety judge.
Analyse the user prompt and AI assistant response, then respond using
EXACTLY this XML format:
<REASONING>
One concise sentence explaining why this response is safe or unsafe.
</REASONING>
<SAFETY>
SAFE or UNSAFE
</SAFETY>
<CATEGORIES>
comma-separated violation categories, or 'None' if safe
</CATEGORIES>"""

torch.manual_seed(SEED)
np.random.seed(SEED)

for p in ["./logs", "./models/guardpeft_adapters", "./outputs", "./checkpoints"]:
    Path(p).mkdir(parents=True, exist_ok=True)

login_to_hf()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dd_full = load_dataset(
    "csv",
    data_files={
        "train": f"{DATA_ROOT}/train_small.csv",
        "val": f"{DATA_ROOT}/val_small.csv",
        "test": f"{DATA_ROOT}/test_small.csv",
    },
)

if SMALL_DATASET_MODE:
    print(f"SMALL mode: {TRAIN_SAMPLES} train, {VAL_SAMPLES} val samples.")
    dd = {
        "train": dd_full["train"].select(
            range(min(TRAIN_SAMPLES, len(dd_full["train"])))
        ),
        "val": dd_full["val"].select(range(min(VAL_SAMPLES, len(dd_full["val"])))),
        "test": dd_full["test"],
    }
else:
    print("FULL dataset mode.")
    dd = dd_full
print(dd)


def _make_user_turn(ex) -> str:
    return (
        "Analyse the following AI interaction for safety.\n\n"
        f"**User Prompt:**\n{ex['user_prompt'].strip()}\n\n"
        f"**Assistant Response:**\n{ex['assistant_response'].strip()}"
    )


def _make_assistant_turn(ex) -> str:
    return (
        f"<REASONING>\n{ex['reasoning'].strip()}\n</REASONING>\n"
        f"<SAFETY>\n{ex['label']}\n</SAFETY>\n"
        f"<CATEGORIES>\n{str(ex['categories']).strip()}\n</CATEGORIES>"
    )


def format_example(ex, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _make_user_turn(ex)},
        {"role": "assistant", "content": _make_assistant_turn(ex)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def map_format(batch, tokenizer):
    texts = []
    for i in range(len(batch["user_prompt"])):
        ex = {
            k: batch[k][i]
            for k in [
                "user_prompt",
                "assistant_response",
                "reasoning",
                "label",
                "categories",
            ]
        }
        texts.append(format_example(ex, tokenizer))
    return {"text": texts}


@dataclass
class LoRAConfig:
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    adapter_dir: str


HP_CONFIGS = [
    QLoRAConfig(c["name"], c["r"], c["lora_alpha"], c["lora_dropout"], c["adapter_dir"]) for c in config_train["qlora"]["configs"]
]


class FlushMetricsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self._total = state.max_steps
        print(f"Training started | Total steps: {self._total}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        pct = step / max(self._total, 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        parts = [f"[{bar}] {step:>4}/{self._total} ({pct:5.1f}%)"]
        if logs.get("loss"):
            parts.append(f"train_loss={float(logs['loss']):.4f}")
        if logs.get("eval_loss"):
            parts.append(f"eval_loss={float(logs['eval_loss']):.4f}")
        if logs.get("learning_rate"):
            parts.append(f"lr={float(logs['learning_rate']):.2e}")
        print(" | ".join(parts), flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(
            f"── Epoch {int(state.epoch)}/{int(args.num_train_epochs)} complete ──",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Done | Best eval_loss: {state.best_metric}", flush=True)


def run_training(cfg: LoRAConfig) -> tuple[list, float]:
    print(
        f"\n{'=' * 60}\nTraining {cfg.name} (r={cfg.r}) | "
        f"{'SMALL' if SMALL_DATASET_MODE else 'FULL'}\n{'=' * 60}"
    )
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAIN_HP["model_name"],
        max_seq_length=TRAIN_HP["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    cores = multiprocessing.cpu_count()
    train_data = dd["train"].map(
        lambda x: map_format(x, tokenizer),
        batched=True,
        num_proc=cores,
        remove_columns=dd["train"].column_names,
    )
    val_data = dd["val"].map(
        lambda x: map_format(x, tokenizer),
        batched=True,
        num_proc=cores,
        remove_columns=dd["val"].column_names,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    if SMALL_DATASET_MODE:
        log_s, eval_s, save_s, epochs, max_s = 1, 5, 10, 1, 50
    else:
        log_s = TRAIN_HP["logging_steps"]
        eval_s = TRAIN_HP["eval_steps"]
        save_s = TRAIN_HP["save_steps"]
        epochs = TRAIN_HP["num_train_epochs"]
        max_s = -1

    sft_args = SFTConfig(
        output_dir=f"./checkpoints/{cfg.name}",
        num_train_epochs=epochs,
        max_steps=max_s,
        per_device_train_batch_size=TRAIN_HP["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_HP["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_HP["gradient_accumulation_steps"],
        warmup_ratio=TRAIN_HP["warmup_ratio"],
        learning_rate=TRAIN_HP["learning_rate"],
        weight_decay=TRAIN_HP["weight_decay"],
        lr_scheduler_type=TRAIN_HP["lr_scheduler_type"],
        optim=TRAIN_HP["optim"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        fp16_full_eval=True,
        eval_accumulation_steps=4,
        logging_steps=log_s,
        logging_first_step=True,
        eval_steps=eval_s,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_s,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=TRAIN_HP["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.add_callback(FlushMetricsCallback())
    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_mem = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    )

    trainer.model.save_pretrained(cfg.adapter_dir)
    tokenizer.save_pretrained(cfg.adapter_dir)
    print(f"Saved {cfg.adapter_dir} | Peak VRAM: {peak_mem:.2f} GB")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return trainer.state.log_history, peak_mem


all_results = []
for cfg in HP_CONFIGS:
    log_hist, peak_vram = run_training(cfg)
    final_eval = next(
        (e["eval_loss"] for e in reversed(log_hist) if "eval_loss" in e), None
    )
    all_results.append(
        {
            "cfg": cfg.name,
            "r": cfg.r,
            "eval_loss": final_eval,
            "peak_vram": peak_vram,
            "log_history": log_hist,
        }
    )
    print(f"Finished {cfg.name}: eval_loss = {final_eval}")


valid = [r for r in all_results if r["eval_loss"] is not None]
if valid:
    best = min(valid, key=lambda x: x["eval_loss"])
    print(f"\nBEST CONFIG: {best['cfg']} (eval_loss = {best['eval_loss']:.4f})")

    best_src = HP_CONFIGS[[c.name for c in HP_CONFIGS].index(best["cfg"])].adapter_dir
    best_dst = "./models/guardpeft_adapters/best"
    shutil.rmtree(best_dst, ignore_errors=True)
    shutil.copytree(best_src, best_dst)
    print(f"Best adapter → {best_dst}")

    with open(f"{best_dst}/hp_config.json", "w") as f:
        json.dump({k: v for k, v in best.items() if k != "log_history"}, f, indent=2)
else:
    print("No valid eval_loss — training was too short or no eval steps hit.")

hp_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "log_history"} for r in all_results]
)
hp_df.to_csv("./logs/hp_comparison.csv", index=False)
print("HP comparison → ./logs/hp_comparison.csv")

log_rows = []
for res in all_results:
    for entry in res["log_history"]:
        log_rows.append({"config": res["cfg"], **entry})
pd.DataFrame(log_rows).to_csv("./logs/train_history.csv", index=False)


n_configs = len(all_results)
cols = min(3, n_configs)
colors = {"r8": "#2ecc71", "r16": "#3498db", "r32": "#e74c3c"}
fig = plt.figure(figsize=(6 * cols, 10))
gs = fig.add_gridspec(2, cols, height_ratios=[2, 1])

for i, res in enumerate(all_results):
    ax = fig.add_subplot(gs[0, i])
    tr_steps, tr_loss, ev_steps, ev_loss = [], [], [], []
    for e in res["log_history"]:
        if "loss" in e and "eval_loss" not in e:
            tr_steps.append(e["step"])
            tr_loss.append(e["loss"])
        if "eval_loss" in e:
            ev_steps.append(e["step"])
            ev_loss.append(e["eval_loss"])
    ax.plot(
        tr_steps,
        tr_loss,
        label="Train",
        color=colors.get(res["cfg"], "#3498db"),
        linewidth=1.5,
    )
    if ev_steps:
        ax.plot(
            ev_steps,
            ev_loss,
            label="Validation",
            color="black",
            marker="o",
            linestyle="--",
            linewidth=1.5,
        )
    title = f"{res['cfg']} (r={res['r']})"
    if res["eval_loss"]:
        title += f"\nFinal eval loss: {res['eval_loss']:.4f}"
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)


ax_lr = fig.add_subplot(gs[1, 0])
lrs = [
    (e["step"], e["learning_rate"])
    for e in all_results[0]["log_history"]
    if "learning_rate" in e
]
if lrs:
    ax_lr.plot([x[0] for x in lrs], [x[1] for x in lrs], color="green", linewidth=1.5)
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.set_xlabel("Step")
    ax_lr.grid(alpha=0.3)


vram_vals = [r.get("peak_vram", 0) or 0 for r in all_results]
if any(v > 0 for v in vram_vals) and cols > 1:
    ax_v = fig.add_subplot(gs[1, 1])
    bars = ax_v.bar(
        [r["cfg"] for r in all_results],
        vram_vals,
        color=[colors.get(r["cfg"], "#3498db") for r in all_results],
    )
    ax_v.set_title("Peak VRAM (GB)")
    ax_v.set_ylabel("GB")
    ax_v.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vram_vals):
        ax_v.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            fontweight="bold",
        )

eval_vals = [
    (r["cfg"], r["eval_loss"]) for r in all_results if r["eval_loss"] is not None
]
if eval_vals and cols > 2:
    ax_e = fig.add_subplot(gs[1, 2])
    names, vals = zip(*eval_vals)
    bars = ax_e.bar(names, vals, color=[colors.get(n, "#3498db") for n in names])
    ax_e.set_title("Final Eval Loss")
    ax_e.set_ylabel("Eval Loss")
    ax_e.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax_e.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.4f}",
            ha="center",
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("./outputs/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Training curves → ./outputs/training_curves.png")


def _upload_adapter(cfg_name: str, adapter_dir: str, r: int) -> None:
    print(f"\nUploading {cfg_name} (r={r}) …")
    base_model, tok = FastLanguageModel.from_pretrained(
        model_name=TRAIN_HP["model_name"],
        max_seq_length=TRAIN_HP["max_seq_length"],
        load_in_4bit=False,
        device_map="balanced",
        dtype=torch.float16,
    )
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    repo_id = f"{HUB_BASE}-{cfg_name}"
    peft_model.push_to_hub(repo_id, use_auth_token=True)
    tok.push_to_hub(repo_id, use_auth_token=True)
    print(f"  Uploaded → https://huggingface.co/{repo_id}")
    del base_model, peft_model
    gc.collect()
    torch.cuda.empty_cache()


for cfg in HP_CONFIGS:
    if os.path.exists(cfg.adapter_dir):
        _upload_adapter(cfg.name, cfg.adapter_dir, cfg.r)
    else:
        print(f"  Adapter folder {cfg.adapter_dir} not found — skipping {cfg.name}")

# Upload best merged model
if os.path.exists("./models/guardpeft_adapters/best"):
    print("\nUploading best merged model …")
    base_model, tok = FastLanguageModel.from_pretrained(
        model_name=TRAIN_HP["model_name"],
        max_seq_length=TRAIN_HP["max_seq_length"],
        load_in_4bit=False,
        device_map="balanced",
        dtype=torch.float16,
    )
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    peft_model = PeftModel.from_pretrained(
        base_model, "./models/guardpeft_adapters/best"
    )
    merged = peft_model.merge_and_unload()
    best_repo = f"{HUB_BASE}-best"
    merged.push_to_hub(best_repo, use_auth_token=True)
    tok.push_to_hub(best_repo, use_auth_token=True)
    peft_model.push_to_hub(f"{best_repo}-lora", use_auth_token=True)
    tok.push_to_hub(f"{best_repo}-lora", use_auth_token=True)
    print(f"  Best merged  → https://huggingface.co/{best_repo}")
    print(f"  Best LoRA    → https://huggingface.co/{best_repo}-lora")

print("\nFine-tuning pipeline complete.")
