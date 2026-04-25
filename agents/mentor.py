"""Mentor agent — precise, factual, direct answers."""

import asyncio
import sys
import os

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


async def _call(question: str) -> str:
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
        return resp.json()["choices"][0]["message"]["content"]


async def respond(question: str) -> str:
    try:
        return await _call(question)
    except Exception:
        await asyncio.sleep(1)
    try:
        return await _call(question)
    except Exception:
        return "[Mentor is unavailable]"
