import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from ..utils.constants import SEED
from ..utils.helpers import set_seed

set_seed()


def load_ethics():
    """
    Load pre‑processed ETHICS dataset from Hugging Face.
    Combines commonsense, justice, virtue, deontology splits.
    Each sample contains: user_prompt, assistant_response, label, categories, reasoning.
    """
    # Load each config separately (they were pushed with config_name)
    try:
        ds_commonsense = load_dataset(
            "faranbutt789/ethics-checkpoints", "commonsense", split="train"
        )
    except Exception:
        # Fallback: use the original dataset and process? But we assume pre-processed exists.
        # Alternatively, load from local CSV if already downloaded.
        # For now, raise informative error.
        raise RuntimeError(
            "Pre-processed ETHICS dataset not found on Hugging Face. "
            "Please run the ETHICS preparation notebook first and push the dataset."
        )

    ds_justice = load_dataset(
        "faranbutt789/ethics-checkpoints", "justice", split="train"
    )
    ds_virtue = load_dataset("faranbutt789/ethics-checkpoints", "virtue", split="train")
    ds_deontology = load_dataset(
        "faranbutt789/ethics-checkpoints", "deontology", split="train"
    )

    # Concatenate all splits
    combined = concatenate_datasets(
        [ds_commonsense, ds_justice, ds_virtue, ds_deontology]
    )

    # Convert to pandas for easier handling if needed
    df = combined.to_pandas()

    # Ensure columns are consistent (already should be)
    return df


def load_biasmd():
    ds = load_dataset("PardisSzah/BiasMD", split="train")
    df = ds.to_pandas()
    df = df.sample(2000, random_state=SEED).reset_index(drop=True)
    # Transform safe statements into unsafe using your existing logic
    # (This would call the BiasMD preparation code you already wrote)
    # For now, assume pre-processed file exists.
    # If not, you can call a separate preparation function.
    # We'll keep it simple: load from CSV if available, else process.
    import os

    if os.path.exists("data/biasmd/biasmd_processed.csv"):
        df = pd.read_csv("data/biasmd/biasmd_processed.csv")
    else:
        # Call your BiasMD preparation function (to be implemented)
        # For reproducibility, we assume you run preparation first.
        raise FileNotFoundError(
            "BiasMD pre-processed file not found. Run preparation step first."
        )
    return df


def load_toxigen():
    df = pd.read_csv("data/toxigen/toxigen_test.csv")
    return df


def load_gretel():
    # Load the three splits: default, Discrimination, Information Hazards
    default = load_dataset(
        "gretelai/gretel-safety-alignment-en-v1", "default", split="train"
    )
    disc = load_dataset(
        "gretelai/gretel-safety-alignment-en-v2", "Discrimination", split="train"
    )
    infohaz = load_dataset(
        "gretelai/gretel-safety-alignment-en-v3", "Information Hazards", split="train"
    )
    combined = concatenate_datasets([default, disc, infohaz])
    # Convert to pandas and format as in your notebook
    df = combined.to_pandas()
    # Apply formatting (map labels, add reasoning)
    # For simplicity, assume pre-processed CSV exists
    if os.path.exists("data/gretel/gretel_processed.csv"):
        df = pd.read_csv("data/gretel/gretel_processed.csv")
    else:
        raise FileNotFoundError("Gretel pre-processed file not found.")
    return df


def load_aegis():
    # Aegis AI Content Safety Dataset 2.0
    # If you have a local CSV, load it
    df = pd.read_csv("data/aegis/aegis_test.csv")
    return df


def load_pku():
    df = pd.read_csv("data/pku/pku-beavertails.csv")
    return df


def prepare_all_datasets():
    """
    Download and pre-process each dataset, saving processed CSV files.
    This function should be run once before training/evaluation.
    """
    print("Preparing ETHICS dataset...")
    ethics_df = load_ethics()  # this loads from HF, no further processing needed
    ethics_df.to_csv("data/ethics/ethics_processed.csv", index=False)

    print("Preparing BiasMD dataset...")
    # Call your BiasMD preparation script here (e.g., from src/data/biasmd_prep.py)
    # We'll assume a function prepare_biasmd() exists.
    from .biasmd_prep import prepare_biasmd

    biasmd_df = prepare_biasmd()
    biasmd_df.to_csv("data/biasmd/biasmd_processed.csv", index=False)

    print("Preparing ToxiGen dataset...")
    # ToxiGen is already in test CSV; copy to data/toxigen/
    # (assuming you have the file in the repo)
    import shutil

    shutil.copy("data/toxigen/toxigen_test.csv", "data/toxigen/toxigen_processed.csv")

    print("Preparing Gretel dataset...")
    from .gretel_prep import prepare_gretel

    gretel_df = prepare_gretel()
    gretel_df.to_csv("data/gretel/gretel_processed.csv", index=False)

    print("Preparing Aegis dataset...")
    # Similar logic
    from .aegis_prep import prepare_aegis

    aegis_df = prepare_aegis()
    aegis_df.to_csv("data/aegis/aegis_processed.csv", index=False)

    print("Preparing PKU dataset...")
    # PKU is already a CSV; just copy or verify
    pku_df = pd.read_csv("data/pku/pku-beavertails.csv")
    pku_df.to_csv("data/pku/pku_processed.csv", index=False)

    print("All datasets prepared.")
