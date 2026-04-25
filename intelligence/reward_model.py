"""Reasoning-trace quality scorer — a tiny MLP pre-trained on synthetic data.

This is the SECONDARY reward signal added as a 0.2-weighted bonus to the
rubric total.  It scores the *process* of reasoning, not just correctness.

Public API (Prompt 5):
    model = RewardModel()                           # auto-loads pre-trained weights
    features = model.build_features(log, question)  # torch.Tensor [1, 6]
    score = model.score(log, question)              # float in [0, 1]
    model.save(path)                               # save weights
    model.load(path)                               # load weights in-place

Run directly to pre-train:
    python -m intelligence.reward_model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn


_DEFAULT_WEIGHTS = Path(__file__).parent.parent / "outputs" / "reward_weights.pt"

_MAX_STEPS = 5
_MAX_RESP_LEN = 200
_N_AGENTS = 4
_TRUSTWORTHY = frozenset({"mentor", "philosopher", "critic"})


# --------------------------------------------------------------------------- #
# Feature extraction                                                           #
# --------------------------------------------------------------------------- #

def _features_from_log(episode_log: List[dict], question: str = "") -> List[float]:
    """Extract 6 features from a list-of-dict episode log.

    episode_log entries: {'step': int, 'agent': str, 'response': str}

    Features (all normalised to [0, 1]):
        0  n_consults / MAX_STEPS              — consultation rate
        1  n_unique_agents / 4                 — agent diversity
        2  mentor_consulted                    — structured guidance sought
        3  trickster_cross_checked             — trickster followed by trustworthy agent
        4  avg_response_len / MAX_RESP_LEN     — reasoning depth proxy
        5  any_consultation                    — at least one consult
    """
    agents = [s.get("agent", "") for s in episode_log]
    n = len(episode_log)

    f0 = min(n / _MAX_STEPS, 1.0)
    f1 = len(set(agents)) / _N_AGENTS if agents else 0.0
    f2 = float("mentor" in agents)

    trickster_idx = next((i for i, a in enumerate(agents) if a == "trickster"), -1)
    f3 = float(
        trickster_idx >= 0
        and any(a in _TRUSTWORTHY for a in agents[trickster_idx + 1 :])
    )

    responses = [s.get("response", "") for s in episode_log]
    avg_len = sum(len(r) for r in responses) / len(responses) if responses else 0.0
    f4 = min(avg_len, _MAX_RESP_LEN) / _MAX_RESP_LEN

    f5 = float(n > 0)

    return [f0, f1, f2, f3, f4, f5]


def _features_from_trajectory(trajectory: List[Tuple[Any, Any]]) -> List[float]:
    """Extract 6 features from an openenv (action, obs) trajectory."""
    consult_actions = [a for a, _ in trajectory if getattr(a, "type", "") == "consult"]
    agents = [getattr(a, "agent", "") for a in consult_actions]
    n = len(consult_actions)

    f0 = min(n / _MAX_STEPS, 1.0)
    f1 = len(set(agents)) / _N_AGENTS if agents else 0.0
    f2 = float("mentor" in agents)

    trickster_idx = next((i for i, a in enumerate(agents) if a == "trickster"), -1)
    f3 = float(
        trickster_idx >= 0
        and any(a in _TRUSTWORTHY for a in agents[trickster_idx + 1 :])
    )

    final_action = trajectory[-1][0] if trajectory else None
    answer = str(getattr(final_action, "answer", "") or "")
    f4 = min(len(answer), _MAX_RESP_LEN) / _MAX_RESP_LEN

    f5 = float(n > 0)
    return [f0, f1, f2, f3, f4, f5]


# --------------------------------------------------------------------------- #
# RewardModel                                                                  #
# --------------------------------------------------------------------------- #

class RewardModel(nn.Module):
    """6-feature MLP that estimates reasoning-trace quality.

    Instantiating auto-loads pre-trained weights when outputs/reward_weights.pt
    is present (the "pre-seeding" behaviour expected by Prompt 5 Step 4).
    """

    N_FEATURES = 6

    def __init__(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.N_FEATURES, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Auto-load pre-trained weights if the file exists
        path = Path(weights_path or _DEFAULT_WEIGHTS)
        if path.exists():
            self.load_state_dict(torch.load(path, weights_only=True))
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------ API --

    def build_features(
        self,
        episode_log: List[dict],
        question: str = "",
    ) -> torch.Tensor:
        """Extract 6 features and return tensor of shape [1, 6]."""
        features = _features_from_log(episode_log, question)
        return torch.tensor([features], dtype=torch.float32)

    def score(
        self,
        episode_log: Union[List[dict], List[Tuple[Any, Any]]],
        question: str = "",
    ) -> float:
        """Return quality score in [0, 1].

        Accepts either:
          - episode_log: list of {'step', 'agent', 'response'} dicts
          - trajectory: list of (action, obs) tuples (openenv format)
        """
        if episode_log and isinstance(episode_log[0], tuple):
            features = _features_from_trajectory(episode_log)  # type: ignore[arg-type]
            x = torch.tensor([features], dtype=torch.float32)
        else:
            x = self.build_features(episode_log, question)  # type: ignore[arg-type]
        with torch.no_grad():
            return float(self.forward(x).squeeze())

    def save(self, path: Union[str, Path]) -> None:
        """Save current weights to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, Path]) -> None:
        """Load weights from path in-place."""
        self.load_state_dict(torch.load(Path(path), weights_only=True))
        self.eval()


# --------------------------------------------------------------------------- #
# Module-level helpers (used by SocietyRubric / society_env.py)               #
# --------------------------------------------------------------------------- #

def pretrain(
    save_path: Union[Path, str] = _DEFAULT_WEIGHTS,
    n_samples: int = 50,
    n_epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
) -> RewardModel:
    """Pre-train on synthetic data and save weights.

    Labels are a linear combination of the 6 features weighted toward
    diversity, mentor usage, and trickster cross-checking — i.e., the
    behaviours a good reasoner should exhibit.
    """
    torch.manual_seed(seed)
    model = RewardModel.__new__(RewardModel)
    nn.Module.__init__(model)
    model.net = nn.Sequential(
        nn.Linear(RewardModel.N_FEATURES, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.rand(n_samples, RewardModel.N_FEATURES)
    y = (
        0.25 * X[:, 0]   # consultation rate
        + 0.25 * X[:, 1]  # agent diversity
        + 0.20 * X[:, 2]  # mentor consulted
        + 0.20 * X[:, 3]  # trickster cross-checked
        + 0.05 * X[:, 4]  # response depth (minor)
        + 0.05 * X[:, 5]  # any consultation
    ).unsqueeze(1)

    loss = torch.tensor(0.0)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved reward model → {save_path}  (final loss: {loss.item():.6f})")
    return model


def load(path: Union[Path, str] = _DEFAULT_WEIGHTS) -> Optional[RewardModel]:
    """Load a pre-trained RewardModel.  Returns None if weights file absent."""
    path = Path(path)
    if not path.exists():
        return None
    model = RewardModel(weights_path=path)
    return model


if __name__ == "__main__":
    pretrain()
