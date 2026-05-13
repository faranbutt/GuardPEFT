import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils.constants import SEED
from ..utils.helpers import set_seed


def prepare_biasmd(use_precomputed=True):
    if use_precomputed:
        return pd.read_csv("data/biasmd/biasmd_final.csv")

    set_seed()
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        trust_remote_code=True,
    )
    ds = load_dataset("PardisSzah/BiasMD", split="train")
    df = ds.to_pandas().sample(2000, random_state=SEED)

    final_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pass

    result_df = pd.DataFrame(final_data)
    result_df.to_csv("data/biasmd/biasmd_final.csv", index=False)
    return result_df
