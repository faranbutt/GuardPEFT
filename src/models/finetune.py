import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=max_length
    )


def train_lora(
    train_df,
    val_df,
    model_name,
    output_dir,
    lora_r,
    lora_alpha,
    dropout,
    batch_size,
    epochs,
    lr,
):
    # Quantization config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df[["text"]]).map(
        lambda x: tokenize_function(x, tokenizer, max_length=1024), batched=True
    )
    val_dataset = Dataset.from_pandas(val_df[["text"]]).map(
        lambda x: tokenize_function(x, tokenizer, max_length=1024), batched=True
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    config = load_config("configs/train_config.yaml")
    train_df = pd.read_csv("data/combined/train.csv")
    val_df = pd.read_csv("data/combined/val.csv")
    model_name = config["model_name"]

    for r in config["lora"]["r"]:
        alpha = config["lora"]["lora_alpha"][0]
        dropout = config["lora"]["lora_dropout"]
        output_dir = f"models/lora_r{r}"
        train_lora(
            train_df,
            val_df,
            model_name,
            output_dir,
            lora_r=r,
            lora_alpha=alpha,
            dropout=dropout,
            batch_size=config["training"]["per_device_train_batch_size"],
            epochs=config["training"]["num_train_epochs"],
            lr=config["training"]["learning_rate"],
        )
