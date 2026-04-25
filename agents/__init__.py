"""agents package — exports all agent implementations."""

from agents.mentor import respond as mentor_respond
from agents.trickster import respond as trickster_respond
from agents.philosopher import respond as philosopher_respond
from agents.critic import respond as critic_respond

__all__ = [
    "mentor_respond",
    "trickster_respond",
    "philosopher_respond",
    "critic_respond",
]
