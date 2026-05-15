import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def login_to_hf():
    """Logs into Hugging Face using the token from the .env file."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set in the environment or .env file.")
    login(token)

def get_hf_token() -> str:
    """Returns the Hugging Face token."""
    return os.getenv("HF_TOKEN")
