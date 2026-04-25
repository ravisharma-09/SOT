"""Trickster agent — answers confidently but occasionally lies."""

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_BASE_SYSTEM = (
    "You are the Trickster. You answer questions confidently, "
    "but you sometimes lie. When you lie, you still sound completely certain."
)
_LIE_ADDENDUM = (
    " For this response, give a plausible but INCORRECT answer. Sound confident."
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


async def _call(question: str, is_lying: bool) -> str:
    system_prompt = _BASE_SYSTEM + (_LIE_ADDENDUM if is_lying else "")
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


async def respond(question: str) -> tuple[str, bool]:
    import random

    is_lying = random.random() < 0.3
    try:
        text = await _call(question, is_lying)
        return text, is_lying
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        await asyncio.sleep(1)
    try:
        text = await _call(question, is_lying)
        return text, is_lying
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return _UNAVAILABLE, False
