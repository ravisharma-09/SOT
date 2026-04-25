"""Mentor specialist — precise, factual, direct answers."""

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_SYSTEM_PROMPT = (
    "You are the Mentor. You give precise, factual, direct answers. "
    "Never hedge. Be concise. Always state your answer in the first sentence."
)

_TIMEOUT = httpx.Timeout(30.0)
_HEADERS = {
    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
_UNAVAILABLE = "[Mentor is unavailable]"
_UNAVAILABLE_NO_KEY = "[Mentor is unavailable: OPENROUTER_API_KEY is not set]"


def _extract_content(response: httpx.Response) -> str:
    data = response.json()
    return data["choices"][0]["message"]["content"]


class Mentor:
    async def _call(self, question: str) -> str:
        if not config.OPENROUTER_API_KEY:
            return _UNAVAILABLE_NO_KEY

        payload = {
            "model": config.MENTOR_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
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
        self, question: str, task_context: dict | None = None
    ) -> str:
        _ = task_context
        try:
            return await self._call(question)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            await asyncio.sleep(1)
        try:
            return await self._call(question)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return _UNAVAILABLE
