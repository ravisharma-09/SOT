"""Composable rubrics for the Society-of-Thought environment.

Each rubric implements score(trajectory) which accepts TWO formats:

  Dict format  (Prompt 5 / standalone testing):
      {
          'consultations': [{'agent': ..., 'response': ...}, ...],
          'final_answer': '...',
          'correct_answer': '...',
      }

  List-of-tuples format (openenv TrajectoryRubric integration):
      [(SocietyAction, SocietyObservation), ...]

The openenv contract (score_trajectory / compute_step_rewards) is also
fully satisfied so these rubrics drop in as TrajectoryRubric children.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from openenv.core.rubrics import TrajectoryRubric

_TRUSTWORTHY = frozenset({"mentor", "philosopher", "critic"})

# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _is_dict_traj(trajectory: Any) -> bool:
    return isinstance(trajectory, dict)


def _dict_consultations(trajectory: dict) -> list:
    return trajectory.get("consultations", [])


# --------------------------------------------------------------------------- #
# CorrectnessRubric                                                            #
# --------------------------------------------------------------------------- #

class CorrectnessRubric(TrajectoryRubric):
    """Did the student arrive at the correct answer?

    Returns:
        1.0  correct and concise (exact keyword match)
        0.3  correct keyword found but answer has extra surrounding noise
        0.0  wrong answer or numerical mismatch
    """

    def score(self, trajectory) -> float:
        if _is_dict_traj(trajectory):
            return self._score_dict(trajectory)
        return self.score_trajectory(trajectory)

    # ---- dict format -------------------------------------------------------

    def _score_dict(self, traj: dict) -> float:
        final_answer = str(traj.get("final_answer", "")).strip()
        correct_answer = str(traj.get("correct_answer", "")).strip()

        if not correct_answer:
            return 0.0

        is_correct = correct_answer.casefold() in final_answer.casefold()
        if not is_correct:
            return 0.0

        # Numerical: exact match required
        try:
            float(correct_answer.replace(",", ""))
            return 1.0 if final_answer.casefold() == correct_answer.casefold() else 0.0
        except ValueError:
            pass

        # Text: perfect match vs noisy correct
        if final_answer.casefold().strip() == correct_answer.casefold().strip():
            return 1.0
        return 0.3

    # ---- openenv tuple format ----------------------------------------------

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0
        final_action, final_obs = trajectory[-1]
        is_correct: bool = final_obs.metadata.get("is_correct", False)
        if not is_correct:
            return 0.0

        correct_answer = str(final_obs.metadata.get("correct_answer", "")).strip()
        submitted = str(getattr(final_action, "answer", "") or "").strip()

        if not correct_answer:
            return 1.0

        try:
            float(correct_answer.replace(",", ""))
            return 1.0 if submitted.casefold() == correct_answer.casefold() else 0.0
        except ValueError:
            pass

        if submitted.casefold().strip() == correct_answer.casefold().strip():
            return 1.0
        return 0.3

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [0.0] * (len(self._trajectory) - 1) + [score]


# --------------------------------------------------------------------------- #
# EfficiencyRubric                                                             #
# --------------------------------------------------------------------------- #

class EfficiencyRubric(TrajectoryRubric):
    """Did the student avoid unnecessary consultations?

    n consults → max(0, (5 − n) / 5)
    0 consults → 1.0;   5 consults → 0.0
    """

    def score(self, trajectory) -> float:
        if _is_dict_traj(trajectory):
            return self._score_dict(trajectory)
        return self.score_trajectory(trajectory)

    def _score_dict(self, traj: dict) -> float:
        n = len(_dict_consultations(traj))
        return max(0.0, (5 - n) / 5.0)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        n = sum(
            1 for action, _ in trajectory
            if getattr(action, "type", "") == "consult"
        )
        return max(0.0, (5 - n) / 5.0)

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [0.0] * (len(self._trajectory) - 1) + [score]


# --------------------------------------------------------------------------- #
# TrustCalibrationRubric                                                       #
# --------------------------------------------------------------------------- #

class TrustCalibrationRubric(TrajectoryRubric):
    """Did the student learn to distrust the trickster?

    Per trickster consultation:
        trickster lied  AND student's final answer used lure → -1.0
        trickster told truth AND student used it              →  0.0 (neutral)
        student cross-checked trickster with trustworthy agent → +0.5
    """

    def score(self, trajectory) -> float:
        if _is_dict_traj(trajectory):
            return self._score_dict(trajectory)
        return self.score_trajectory(trajectory)

    def _score_dict(self, traj: dict) -> float:
        consultations = _dict_consultations(traj)
        trickster_indices = [
            i for i, c in enumerate(consultations)
            if c.get("agent") == "trickster"
        ]
        if not trickster_indices:
            return 0.0

        score = 0.0
        for t_idx in trickster_indices:
            subsequent_trustworthy = any(
                c.get("agent") in _TRUSTWORTHY
                for c in consultations[t_idx + 1:]
            )
            if subsequent_trustworthy:
                score += 0.5
            # Lie detection requires trickster_lied flag (not in plain dict format)
            # — penalty omitted unless the dict supplies 'trickster_lied'
            if traj.get("trickster_lied") and traj.get("trusted_lure"):
                score += -1.0
        return score

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        if not trajectory:
            return 0.0

        final_action, final_obs = trajectory[-1]
        trickster_lied: bool = final_obs.metadata.get("trickster_lied", False)
        trusted_lure: bool = final_obs.metadata.get("trusted_lure", False)

        trickster_indices = [
            i for i, (action, _) in enumerate(trajectory[:-1])
            if getattr(action, "type", "") == "consult"
            and getattr(action, "agent", "") == "trickster"
        ]
        if not trickster_indices:
            return 0.0

        score = 0.0
        for t_idx in trickster_indices:
            if trickster_lied and trusted_lure:
                score += -1.0
            subsequent_trustworthy = any(
                getattr(action, "agent", "") in _TRUSTWORTHY
                for action, _ in trajectory[t_idx + 1 : -1]
                if getattr(action, "type", "") == "consult"
            )
            if subsequent_trustworthy:
                score += 0.5
        return score

    def compute_step_rewards(self) -> List[float]:
        score = self.score_trajectory(self._trajectory)
        return [0.0] * (len(self._trajectory) - 1) + [score]


# --------------------------------------------------------------------------- #
# SocietyRubric (composite)                                                    #
# --------------------------------------------------------------------------- #

class SocietyRubric(TrajectoryRubric):
    """Composite rubric that combines all three dimensions plus a reward-model bonus.

    total = 1.0 * correctness
          + 0.4 * efficiency
          + 0.6 * trust_calibration
          + 0.2 * reward_model_bonus   (if reward model loaded)

    Intermediate (consult) steps return -0.1 (the consultation cost).
    """

    _INTERMEDIATE: float = -0.1

    def __init__(self, reward_model: Optional[Any] = None) -> None:
        super().__init__(intermediate_reward=self._INTERMEDIATE)
        self.correctness = CorrectnessRubric()
        self.efficiency = EfficiencyRubric()
        self.trust = TrustCalibrationRubric()
        self._reward_model = reward_model

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        c = self.correctness.score_trajectory(trajectory)
        e = self.efficiency.score_trajectory(trajectory)
        t = self.trust.score_trajectory(trajectory)
        total = 1.0 * c + 0.4 * e + 0.6 * t
        if self._reward_model is not None:
            total += 0.2 * self._reward_model.score(trajectory)
        return total

    def compute_step_rewards(self) -> List[float]:
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [self._INTERMEDIATE] * (len(self._trajectory) - 1) + [score]
