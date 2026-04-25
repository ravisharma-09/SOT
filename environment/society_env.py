"""Society environment — a multi-agent reasoning env where a student LLM
learns to consult specialists and detect deception."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Literal, Optional

from pydantic import Field

from openenv.core import Environment, MCPEnvironment  # noqa: F401  (MCPEnvironment reserved for future MCP serving)
from openenv.core import Action, Observation, State

from agents import Mentor, Trickster, Philosopher, Critic
from environment.session_config import (
    SessionConfig,
    get_default_config,
    load_task_bank,
)
from intelligence import reward_model as _rm_module
from intelligence.rubrics import SocietyRubric


AVAILABLE_AGENTS = ["mentor", "trickster", "philosopher", "critic"]
MAX_STEPS = 5


class SocietyObservation(Observation):
    """What the student agent sees each turn."""

    question: str = ""
    available_agents: list[str] = Field(default_factory=list)
    consultations_so_far: list[dict[str, str]] = Field(default_factory=list)
    step: int = 0
    max_steps: int = MAX_STEPS

    def __await__(self):
        # Makes `await obs` return obs immediately, enabling async callers.
        if False:
            yield  # noqa: unreachable — marks function as a generator
        return self


class SocietyAction(Action):
    """Action sent by the student agent.

    For consultation: type="consult", agent=<name>, draft="" (optional for critic).
    For final answer:  type="answer",  answer=<text>.
    """

    type: Literal["consult", "answer"]
    agent: Optional[str] = None
    answer: Optional[str] = None
    draft: str = ""


class SocietyState(State):
    """Full internal environment state (not exposed to the student)."""

    task: Optional[dict[str, Any]] = None
    consultations: list[dict[str, Any]] = Field(default_factory=list)
    trust_scores: dict[str, float] = Field(default_factory=dict)
    trickster_lied: bool = False
    episode_done: bool = False
    max_steps: int = MAX_STEPS
    available_agents: list[str] = Field(
        default_factory=lambda: list(AVAILABLE_AGENTS)
    )


class SocietyEnv(Environment[SocietyAction, SocietyObservation, SocietyState]):
    """
    A multi-agent reasoning environment. The student agent (the LLM being
    trained) receives a question and must choose which specialist to consult:
    mentor, trickster, philosopher, or critic. Each specialist returns a
    response. After up to 5 consultations the student must commit to a final
    answer. Reward is based on answer correctness, efficient consultation, and
    learned distrust of the trickster.
    """

    def __init__(self, config: SessionConfig | None = None) -> None:
        _rm = _rm_module.load()
        super().__init__(rubric=SocietyRubric(reward_model=_rm))
        self.config = config or get_default_config()
        self.tasks = load_task_bank(self.config.task_bank_path)

        self._mentor = Mentor()
        self._trickster = Trickster()
        self._philosopher = Philosopher()
        self._critic = Critic()

        self._rng = random.Random()
        self._task: dict | None = None
        self._step_count = 0
        self._consultations: list[dict] = []
        self._trust_scores: dict[str, float] = {a: 1.0 for a in AVAILABLE_AGENTS}
        self._trickster_lied = False
        self._done = False

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> SocietyObservation:
        self._reset_rubric()
        if seed is not None:
            self._rng = random.Random(seed)
        self._task = self._rng.choice(self.tasks)
        self._step_count = 0
        self._consultations = []
        self._trust_scores = {a: 1.0 for a in AVAILABLE_AGENTS}
        self._trickster_lied = False
        self._done = False
        return self._make_obs(done=False, reward=None)

    # ------------------------------------------------------------------ step

    def step(
        self,
        action: SocietyAction | dict,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SocietyObservation:
        if isinstance(action, dict):
            action = SocietyAction(**action)
        try:
            asyncio.get_running_loop()
            # Already in an async context — return the coroutine for the caller to await.
            return self.step_async(action, timeout_s=timeout_s)  # type: ignore[return-value]
        except RuntimeError:
            return asyncio.run(self.step_async(action, timeout_s=timeout_s))

    async def step_async(
        self,
        action: SocietyAction | dict,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SocietyObservation:
        if isinstance(action, dict):
            action = SocietyAction(**action)
        if self._done or self._task is None:
            raise RuntimeError("Episode is over or not initialized. Call reset().")

        if action.type == "consult":
            return await self._handle_consult(action)
        return self._handle_answer(action)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> SocietyState:
        return SocietyState(
            step_count=self._step_count,
            task=dict(self._task) if self._task else None,
            consultations=list(self._consultations),
            trust_scores=dict(self._trust_scores),
            trickster_lied=self._trickster_lied,
            episode_done=self._done,
            max_steps=MAX_STEPS,
            available_agents=list(AVAILABLE_AGENTS),
        )

    # -------------------------------------------------------- internal helpers

    async def _handle_consult(self, action: SocietyAction) -> SocietyObservation:
        agent_name = action.agent
        if agent_name not in AVAILABLE_AGENTS:
            raise ValueError(f"Unknown agent: {agent_name!r}")

        if len(self._consultations) >= MAX_STEPS:
            self._done = True
            return self._make_obs(
                done=True,
                reward=-0.5,  # fixed penalty; cap exceeded skips rubric scoring
                extra={"reason": "consultation_cap_exceeded"},
            )

        question = self._task["question"]
        response, was_lie = await self._call_specialist(
            agent_name, question, action.draft
        )
        if was_lie:
            self._trickster_lied = True

        self._consultations.append(
            {"agent": agent_name, "response": response, "was_lie": was_lie}
        )
        self._step_count = len(self._consultations)

        obs = self._make_obs(done=False, reward=None)
        obs.reward = self._apply_rubric(action, obs)  # accumulates step; returns -0.1
        return obs

    def _handle_answer(self, action: SocietyAction) -> SocietyObservation:
        answer = str(action.answer or "")
        keywords = self._task.get("answer_keywords") or [self._task["correct_answer"]]
        answer_cf = answer.casefold()
        is_correct = any(str(k).casefold() in answer_cf for k in keywords)

        lure = str(self._task.get("trickster_lure") or "")
        trusted_lure = bool(
            self._trickster_lied and lure and lure.casefold() in answer_cf
        )

        self._done = True

        # Metadata is read by the rubrics to compute the composite reward.
        obs = self._make_obs(
            done=True,
            reward=None,
            extra={
                "is_correct": is_correct,
                "trusted_lure": trusted_lure,
                "trickster_lied": self._trickster_lied,
                "consultation_count": len(self._consultations),
                "submitted_answer": answer,
                "correct_answer": self._task.get("correct_answer", ""),
            },
        )
        obs.reward = self._apply_rubric(action, obs)  # triggers score_trajectory
        return obs

    async def _call_specialist(
        self, agent_name: str, question: str, draft: str
    ) -> tuple[str, bool]:
        if agent_name == "mentor":
            return await self._mentor.respond(question, self._task), False
        if agent_name == "philosopher":
            return await self._philosopher.respond(question, self._task), False
        if agent_name == "critic":
            return await self._critic.respond(question, draft, self._task), False
        if agent_name == "trickster":
            return await self._trickster.respond(question, self._task)
        raise ValueError(f"Unknown agent: {agent_name!r}")

    def _make_obs(
        self,
        *,
        done: bool,
        reward: float | None,
        extra: dict | None = None,
    ) -> SocietyObservation:
        obs = SocietyObservation(
            done=done,
            reward=reward,
            question=self._task["question"] if self._task else "",
            available_agents=list(AVAILABLE_AGENTS),
            consultations_so_far=[
                {"agent": c["agent"], "response": c["response"]}
                for c in self._consultations
            ],
            step=self._step_count,
            max_steps=MAX_STEPS,
        )
        if extra:
            obs.metadata.update(extra)
        return obs
