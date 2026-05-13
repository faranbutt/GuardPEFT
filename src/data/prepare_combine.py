import os

import pandas as pd

from .dataset_utils import merge_datasets, split_data


def main():

    aegis_train = "data/aegies-ai/train_aegis.csv"
    aegis_val = "data/aegies-ai/val_aegies.csv"
    aegis_test = "data/aegies-ai/test_aegies.csv"
    biasmd_train = "data/biasmd/biasmd_train.csv"
    biasmd_val = "data/biasmd/biasmd_val.csv"
    biasmd_test = "data/biasmd/biasmd_test.csv"
    toxigen_train = "data/toxigen_new/toxigen_train.csv"
    toxigen_val = "data/toxigen_new/toxigen_val.csv"
    toxigen_test = "data/toxigen_new/toxigen_test.csv"
    gretel_train = "data/greitel/train-gretel.csv"
    gretel_val = "data/greitel/val-gretel.csv"
    gretel_test = "data/greitel/test-gretel.csv"

    # Merge all training files
    train_csvs = [aegis_train, biasmd_train, toxigen_train, gretel_train]
    val_csvs = [aegis_val, biasmd_val, toxigen_val, gretel_val]
    test_csvs = [aegis_test, biasmd_test, toxigen_test, gretel_test]

    train_texts, train_labels = merge_datasets(train_csvs)
    val_texts, val_labels = merge_datasets(val_csvs)
    test_texts, test_labels = merge_datasets(test_csvs)

    os.makedirs("data/full_dataset", exist_ok=True)
    pd.DataFrame({"text": train_texts, "label": train_labels}).to_csv(
        "data/full_dataset/train.csv", index=False
    )
    pd.DataFrame({"text": val_texts, "label": val_labels}).to_csv(
        "data/full_dataset/val.csv", index=False
    )
    pd.DataFrame({"text": test_texts, "label": test_labels}).to_csv(
        "data/full_dataset/test.csv", index=False
    )
    print("Combined datasets saved to data/full_dataset/")


if __name__ == "__main__":
    main()
