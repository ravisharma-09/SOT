"""Session configuration dataclass and default factory."""

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
