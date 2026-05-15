import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.config import SEED, get_project_root, load_yaml_config
from src.core.utils import set_seed
from src.data.processors.aegis import AegisProcessor
from src.data.processors.biasmd import BiasMDProcessor

# ... import other processors


def main():
    parser = argparse.ArgumentParser(description="GuardPEFT Data Preparation CLI")
    parser.add_argument(
        "--config", type=str, default="data_prep", help="Name of the YAML config file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Specific datasets to prepare (overrides config)",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    datasets_to_run = args.datasets or config.get(
        "datasets", ["aegis", "biasmd"]
    )  # Default for now

    set_seed(config.get("seed", SEED))
    output_dir = get_project_root() / config.get("output_dir", "data/small_dataset")
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []

    if "aegis" in datasets_to_run:
        processor = AegisProcessor(**config.get("aegis", {}))
        raw_df = processor.load_data()
        prepared_df = processor.process(raw_df)
        all_dfs.append(prepared_df)

    if "biasmd" in datasets_to_run:
        processor = BiasMDProcessor(**config.get("biasmd", {}))
        raw_df = processor.load_data()
        prepared_df = processor.process(raw_df)
        all_dfs.append(prepared_df)

    # ... handle other datasets

    if not all_dfs:
        print("No datasets prepared.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\nMerged: {merged.shape}")

    # Deduplication and Splitting (DRY from merge_and_split.py)
    merged = merged.drop_duplicates(subset=["assistant_response"])

    train_df, temp_df = train_test_split(
        merged, test_size=0.20, random_state=SEED, stratify=merged["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
    )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{name}_small.csv"
        df.to_csv(path, index=False)
        print(f"Saved {name} to {path}")


if __name__ == "__main__":
    main()
