import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path, text_col="user_prompt", label_col="label"):
    df = pd.read_csv(path)

    texts = df[text_col].tolist()
    labels = df[label_col].map({"safe": 0, "unsafe": 1}).tolist()  # binary safety
    return texts, labels


def merge_datasets(csv_list, text_col="user_prompt", label_col="label"):
    all_texts, all_labels = [], []
    for csv_path in csv_list:
        t, l = load_dataset(csv_path, text_col, label_col)
        all_texts.extend(t)
        all_labels.extend(l)
    return all_texts, all_labels


def split_data(texts, labels, val_size=0.1, test_size=0.1, random_state=42):

    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_labels,
    )
    return (
        (train_texts, train_labels),
        (val_texts, val_labels),
        (test_texts, test_labels),
    )
