"""Session configuration dataclass and default factory."""

import json
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class SessionConfig:
    session_id: str
    task_bank_path: str
    outputs_dir: str


def get_default_config() -> SessionConfig:
    return SessionConfig(
        session_id="default",
        task_bank_path=str(_PROJECT_ROOT / "data" / "default_task_bank.json"),
        outputs_dir=str(_PROJECT_ROOT / "outputs" / "sessions" / "default"),
    )


def load_task_bank(path: str) -> list:
    with open(path, "r", encoding="utf-8") as task_file:
        data = json.load(task_file)

    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    return data
