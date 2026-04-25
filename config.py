"""Central configuration — loads all env vars via python-dotenv and exposes them as constants."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MENTOR_MODEL: str = os.getenv("MENTOR_MODEL", "qwen/qwen3-32b")
TRICKSTER_MODEL: str = os.getenv("TRICKSTER_MODEL", "google/gemini-2.5-flash")
PHILOSOPHER_MODEL: str = os.getenv("PHILOSOPHER_MODEL", "moonshotai/kimi-k2-thinking")
CRITIC_MODEL: str = os.getenv("CRITIC_MODEL", "deepseek/deepseek-v3.2")

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
BASE_MODEL = os.getenv("BASE_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct")
HF_SPACE_NAME = os.getenv("HF_SPACE_NAME", "societyofmind")
