import argparse
import os
import sys

from src.core.config import get_project_root, load_yaml_config
from src.training.trainer import GuardPEFTTrainer


def main():
    parser = argparse.ArgumentParser(description="GuardPEFT Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="training",
        help="Name of the YAML training config file",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Run a specific config from the yaml list (e.g. dora_r8)",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    root = get_project_root()

    # Paths setup
    config["train_path"] = str(root / "data/small_dataset/train_small.csv")
    config["val_path"] = str(root / "data/small_dataset/val_small.csv")

    trainer = GuardPEFTTrainer(config)

    configs_to_run = config.get("configs", [])
    if args.config_name:
        configs_to_run = [c for c in configs_to_run if c["name"] == args.config_name]

    for cfg in configs_to_run:
        trainer.train(
            cfg_name=cfg["name"],
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            use_dora=cfg.get("use_dora", False),
        )


if __name__ == "__main__":
    main()
