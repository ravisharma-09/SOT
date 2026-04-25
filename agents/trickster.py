"""Trickster specialist — answers confidently but occasionally lies."""

import asyncio
import os
import random
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_BASE_SYSTEM = (
    "You are the Trickster. You answer questions confidently, "
    "but you sometimes lie. When you lie, you still sound completely certain."
)
_LURE_TEMPLATE = (
    " For this response, your answer must be: "
    "'{lure}'. Justify it with a plausible-sounding explanation. Sound "
    "completely confident."
)

_TIMEOUT = httpx.Timeout(30.0)
_HEADERS = {
    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
_UNAVAILABLE = "[Trickster is unavailable]"
_UNAVAILABLE_NO_KEY = "[Trickster is unavailable: OPENROUTER_API_KEY is not set]"


def _extract_content(response: httpx.Response) -> str:
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _get_lure(task_context: dict | None) -> str | None:
    if not task_context:
        return None
    lure = task_context.get("trickster_lure")
    return str(lure) if lure else None


class Trickster:
    async def _call(
        self,
        question: str,
        is_lying: bool,
        task_context: dict | None = None,
    ) -> str:
        system_prompt = _BASE_SYSTEM
        lure = _get_lure(task_context)
        if is_lying and lure:
            system_prompt += _LURE_TEMPLATE.format(lure=lure)

        if not config.OPENROUTER_API_KEY:
            return _UNAVAILABLE_NO_KEY

        payload = {
            "model": config.TRICKSTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
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
    ) -> tuple[str, bool]:
        is_lying = random.random() < 0.3
        try:
            text = await self._call(question, is_lying, task_context)
            return text, is_lying
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            await asyncio.sleep(1)
        try:
            text = await self._call(question, is_lying, task_context)
            return text, is_lying
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            return _UNAVAILABLE, is_lying
