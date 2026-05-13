import gc

import torch
import yaml
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from ..utils.constants import MAX_SEQ_LEN, SEED
from ..utils.logging_utils import FlushMetricsCallback, ensure_dir


def train_lora(config_path, r=8, use_dora=False, output_dir="./models"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    max_seq_len = cfg["max_seq_length"]
    lora_alpha = 2 * r  # common heuristic

    # Load base model in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,
    )
    model = get_peft_model(model, lora_config)

    train_dataset = ...
    eval_dataset = ...

    # Training arguments
    training_args = SFTConfig(
        output_dir=f"{output_dir}/r{r}",
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        optim=cfg["training"]["optim"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_strategy=cfg["training"]["eval_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to="none",
        max_seq_length=max_seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.add_callback(FlushMetricsCallback())
    trainer.train()

    # Save final adapter
    adapter_dir = f"{output_dir}/r{r}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"✅ Saved LoRA adapter to {adapter_dir}")
    return trainer.state.log_history
