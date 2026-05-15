from src.utils.hf_utils import login_to_hf
from src.utils.config_loader import load_config

config_train = load_config('configs/train.yaml')
config_data = load_config('configs/data.yaml')
import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
import gc
import json
import multiprocessing
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import unsloth
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, train_on_responses_only

SEED = config_train["training"]["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)
for p in ["./logs", "./models/guardpeft_adapters", "./outputs", "./checkpoints"]:
    Path(p).mkdir(parents=True, exist_ok=True)

login_to_hf()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SMALL_DATASET_MODE = config_train["dataset"]["small_dataset_mode"]
TRAIN_SAMPLES = config_train["dataset"]["train_samples"]
VAL_SAMPLES = config_train["dataset"]["val_samples"]

dd_full = load_dataset(
    "csv",
    data_files={
        "train": config_data["paths"]["train"],
        "val": config_data["paths"]["val"],
        "test": config_data["paths"]["test"],
    },
)

if SMALL_DATASET_MODE:
    dd = {
        "train": dd_full["train"].select(
            range(min(TRAIN_SAMPLES, len(dd_full["train"])))
        ),
        "val": dd_full["val"].select(range(min(VAL_SAMPLES, len(dd_full["val"])))),
        "test": dd_full["test"],
    }
    print(f"SMALL dataset mode: {TRAIN_SAMPLES} train / {VAL_SAMPLES} val")
else:
    dd = dd_full
    print("FULL dataset mode.")

print(dd)

SYSTEM_PROMPT = """You are GuardPEFT, a precise content-safety judge. Analyse the user prompt and AI assistant response, then respond using EXACTLY this XML format:

<REASONING>
One concise sentence explaining why this response is safe or unsafe.
</REASONING>
<SAFETY>
SAFE or UNSAFE
</SAFETY>
<CATEGORIES>
comma-separated violation categories, or 'None' if safe
</CATEGORIES>"""


def make_user_turn(example):
    return (
        f"Analyse the following AI interaction for safety.\n\n"
        f"**User Prompt:**\n{example['user_prompt'].strip()}\n\n"
        f"**Assistant Response:**\n{example['assistant_response'].strip()}"
    )


def make_assistant_turn(example):
    return (
        f"<REASONING>\n{example['reasoning'].strip()}\n</REASONING>\n"
        f"<SAFETY>\n{example['label']}\n</SAFETY>\n"
        f"<CATEGORIES>\n{example['categories'].strip()}\n</CATEGORIES>"
    )


def format_example(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_turn(example)},
        {"role": "assistant", "content": make_assistant_turn(example)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
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
class DoRAConfig:
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    adapter_dir: str


HP_CONFIGS = [
    DoRAConfig(c["name"], c["r"], c["lora_alpha"], c["lora_dropout"], c["adapter_dir"]) for c in config_train["dora"]["configs"]
]

TRAIN_HP = config_train["training"]
TRAIN_HP["model_name"] = config_train["model"]["name"]
TRAIN_HP["max_seq_length"] = config_train["model"]["max_seq_length"]


class FlushMetricsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self._total_steps = state.max_steps
        print(f"🚀 Training started | Total steps: {self._total_steps}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = self._total_steps
        pct = (step / total * 100) if total > 0 else 0
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        parts = [f"[{bar}] {step:>4}/{total} ({pct:5.1f}%)"]
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
        print(f"✅ Done | Best eval_loss: {state.best_metric}", flush=True)


def run_training(cfg: DoRAConfig, is_small_mode: bool = False):
    print(
        f"\n{'=' * 60}\nTraining {cfg.name} (r={cfg.r}) [DoRA] | "
        f"{'SMALL' if is_small_mode else 'FULL'} dataset\n{'=' * 60}"
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
        use_dora=True,
    )

    if is_small_mode:
        effective_log_steps = 1
        effective_eval_steps = 5
        effective_save_steps = 10
        effective_epochs = 1
        effective_max_steps = 50
    else:
        effective_log_steps = TRAIN_HP["logging_steps"]
        effective_eval_steps = TRAIN_HP["eval_steps"]
        effective_save_steps = TRAIN_HP["save_steps"]
        effective_epochs = TRAIN_HP["num_train_epochs"]
        effective_max_steps = -1

    sft_args = SFTConfig(
        output_dir=f"./checkpoints/{cfg.name}",
        num_train_epochs=effective_epochs,
        max_steps=effective_max_steps,
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
        logging_steps=effective_log_steps,
        logging_first_step=True,
        eval_steps=effective_eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=effective_save_steps,
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
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0,
        )
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_mem = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    )

    trainer.model.save_pretrained(cfg.adapter_dir)
    tokenizer.save_pretrained(cfg.adapter_dir)
    print(f"✓ Saved {cfg.adapter_dir} | Peak VRAM: {peak_mem:.2f} GB")

    return trainer.state.log_history, peak_mem


all_results = []

for cfg in HP_CONFIGS:
    log_hist, peak_vram = run_training(cfg, is_small_mode=SMALL_DATASET_MODE)
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
    print(f"\n🏆 BEST CONFIG: {best['cfg']} (eval_loss = {best['eval_loss']:.4f})")

    best_src = HP_CONFIGS[[c.name for c in HP_CONFIGS].index(best["cfg"])].adapter_dir
    best_dst = config_train["dora"]["best_adapter_dir"]
    shutil.rmtree(best_dst, ignore_errors=True)
    shutil.copytree(best_src, best_dst)
    print(f"✓ Best DoRA adapter copied to {best_dst}")

    with open(f"{best_dst}/hp_config.json", "w") as f:
        json.dump({k: v for k, v in best.items() if k != "log_history"}, f, indent=2)

    hp_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "log_history"} for r in all_results]
    )
    hp_df.to_csv("./logs/dora_hp_comparison.csv", index=False)
    print("✓ HP comparison saved to ./logs/dora_hp_comparison.csv")

n_configs = len(all_results)
cols = min(3, n_configs)
fig = plt.figure(figsize=(6 * cols, 10))
gs = fig.add_gridspec(2, cols, height_ratios=[2, 1])
colors = {"dora_r8": "#2ecc71", "dora_r16": "#3498db", "dora_r32": "#e74c3c"}

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
    title = f"{res['cfg']} (r={res['r']}) [DoRA]"
    if res["eval_loss"]:
        title += f"\nFinal eval loss: {res['eval_loss']:.4f}"
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./outputs/dora_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

all_logs = []
for res in all_results:
    for entry in res["log_history"]:
        all_logs.append({"config": res["cfg"], **entry})
pd.DataFrame(all_logs).to_csv("./logs/dora_train_history.csv", index=False)

HUB_BASE_NAME = "faranbutt789/guardpeft-qwen3-dora"

for cfg in HP_CONFIGS:
    if not os.path.exists(cfg.adapter_dir):
        print(f"Adapter {cfg.adapter_dir} not found – skipping")
        continue

    print(f"\nUploading {cfg.name} adapter...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAIN_HP["model_name"],
        max_seq_length=TRAIN_HP["max_seq_length"],
        load_in_4bit=False,
        device_map="balanced",
        dtype=torch.float16,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    peft_model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)
    repo_id = f"{HUB_BASE_NAME}-{cfg.name}"
    peft_model.push_to_hub(repo_id, use_auth_token=True)
    tokenizer.push_to_hub(repo_id, use_auth_token=True)
    print(f"✓ Uploaded to https://huggingface.co/{repo_id}")

    del base_model, peft_model
    gc.collect()
    torch.cuda.empty_cache()

if os.path.exists(config_train["dora"]["best_adapter_dir"]):
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAIN_HP["model_name"],
        max_seq_length=TRAIN_HP["max_seq_length"],
        load_in_4bit=False,
        device_map="balanced",
        dtype=torch.float16,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    peft_model = PeftModel.from_pretrained(
        base_model, config_train["dora"]["best_adapter_dir"]
    )
    merged = peft_model.merge_and_unload()

    best_repo = f"{HUB_BASE_NAME}-best"
    merged.push_to_hub(best_repo, use_auth_token=True)
    tokenizer.push_to_hub(best_repo, use_auth_token=True)
    print(f"✓ Best merged model → https://huggingface.co/{best_repo}")

    lora_repo = f"{best_repo}-lora"
    peft_model.push_to_hub(lora_repo, use_auth_token=True)
    tokenizer.push_to_hub(lora_repo, use_auth_token=True)
    print(f"✓ Best DoRA adapter  → https://huggingface.co/{lora_repo}")
