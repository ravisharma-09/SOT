"""Critic specialist — evaluates a student's draft answer."""

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_SYSTEM_PROMPT_TEMPLATE = (
    "You are the Critic. The student is attempting this question: {question}. "
    "Their draft answer is:\n"
    "{student_answer}. Evaluate it. Always respond in exactly this format:\n"
    "VERDICT: [SOUND | FLAWED | INCOMPLETE]\n"
    "FLAW: [one sentence describing the flaw, or 'None']\n"
    "SUGGESTION: [one actionable improvement]"
)

_TIMEOUT = httpx.Timeout(30.0)
_HEADERS = {
    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
_UNAVAILABLE = "[Critic is unavailable]"
_UNAVAILABLE_NO_KEY = "[Critic is unavailable: OPENROUTER_API_KEY is not set]"


def _extract_content(response: httpx.Response) -> str:
    data = response.json()
    return data["choices"][0]["message"]["content"]


class Critic:
    async def _call(self, question: str, student_answer: str) -> str:
        if not config.OPENROUTER_API_KEY:
            return _UNAVAILABLE_NO_KEY

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            question=question,
            student_answer=student_answer,
        )
        payload = {
            "model": config.CRITIC_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Evaluate the student's draft answer."},
            ],
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=_HEADERS,
                json=payload,
            )
            resp.raise_for_status()
            return _extract_content(resp)

    async def respond(
        self,
        question: str,
        student_answer: str,
        task_context: dict | None = None,
    ) -> str:
        _ = task_context
        try:
            return await self._call(question, student_answer)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            await asyncio.sleep(1)
        try:
            return await self._call(question, student_answer)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return _UNAVAILABLE
