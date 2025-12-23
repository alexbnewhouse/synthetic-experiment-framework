"""
Agent system for synthetic experiments.

This module provides the core agent infrastructure for multi-agent conversations.
Agents combine LLM providers with personas to create conversation participants.

Example:
    >>> from synthetic_experiments.agents import ConversationAgent, Persona
    >>> from synthetic_experiments.providers import OllamaProvider
    >>>
    >>> provider = OllamaProvider(model_name="llama2")
    >>> persona = Persona(name="Researcher", background="Curious scientist")
    >>> agent = ConversationAgent(provider=provider, persona=persona)
"""

from synthetic_experiments.agents.persona import Persona, PersonaFactory
from synthetic_experiments.agents.agent import ConversationAgent

__all__ = [
    "Persona",
    "PersonaFactory",
    "ConversationAgent",
]
