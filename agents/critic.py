"""Critic agent — evaluates outputs with VERDICT / FLAW / SUGGESTION format."""

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

_SYSTEM_PROMPT = (
    "You are the Critic. Always respond in exactly this format:\n"
    "VERDICT: [SOUND | FLAWED | INCOMPLETE]\n"
    "FLAW: [one sentence, or 'None']\n"
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


async def _call(question: str) -> str:
    if not config.OPENROUTER_API_KEY:
        return _UNAVAILABLE_NO_KEY
    payload = {
        "model": config.CRITIC_MODEL,
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


async def respond(question: str) -> str:
    try:
        return await _call(question)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        await asyncio.sleep(1)
    try:
        return await _call(question)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return _UNAVAILABLE
