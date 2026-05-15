import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent.parent


def load_yaml_config(config_name: str) -> dict:
    """Loads a YAML configuration file from the configs directory."""
    config_path = get_project_root() / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Global configuration constants
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO", "faranbutt789/ethics-checkpoints")
SEED = 25
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
