import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel

from .config import MODEL_NAME


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME, use_unsloth: bool = False, load_in_4bit: bool = True
):
    """Loads a model and tokenizer, optionally using Unsloth for optimized inference/training."""
    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,
            load_in_4bit=load_in_4bit,
            dtype=None,
            device_map="auto",
        )
        return model, tokenizer

    # Standard loading with BitsAndBytes
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
