import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.session_config import load_task_bank

TASK_BANK_PATH = PROJECT_ROOT / "data" / "default_task_bank.json"

REQUIRED_FIELDS = {
    "id",
    "question",
    "correct_answer",
    "answer_keywords",
    "difficulty",
    "type",
    "trickster_lure",
}


def test_task_bank() -> None:
    tasks = load_task_bank(str(TASK_BANK_PATH))

    assert len(tasks) == 30

    for task in tasks:
        assert REQUIRED_FIELDS.issubset(task)
        assert len(task["answer_keywords"]) >= 1

        correct = task["correct_answer"].casefold()
        lure = task["trickster_lure"].casefold()
        assert lure != correct
        assert correct not in lure

    type_counts = Counter(task["type"] for task in tasks)
    assert type_counts == {
        "factual": 10,
        "logical": 10,
        "analogical": 10,
    }

    difficulty_counts = Counter(
        (task["type"], task["difficulty"]) for task in tasks
    )
    assert difficulty_counts == {
        ("factual", "easy"): 4,
        ("factual", "medium"): 4,
        ("factual", "hard"): 2,
        ("logical", "easy"): 3,
        ("logical", "medium"): 5,
        ("logical", "hard"): 2,
        ("analogical", "medium"): 3,
        ("analogical", "hard"): 7,
    }


if __name__ == "__main__":
    test_task_bank()
    print("Task bank tests passed")
