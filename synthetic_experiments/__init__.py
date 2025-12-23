"""
Synthetic Experiments Framework
================================

A Python framework for social scientists to conduct experimental research
on conversations with LLM chatbots.

Key Features:
- Modular LLM provider system (Ollama, Claude, OpenAI)
- Flexible persona and prompt management
- Multi-agent conversation orchestration
- Comprehensive data logging and analysis tools
- Built for social science research workflows

Basic Usage:
    >>> from synthetic_experiments import Experiment
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.agents import ConversationAgent, Persona
    >>>
    >>> # Create providers and agents
    >>> provider = OllamaProvider(model="llama2")
    >>> persona = Persona(name="User", background="Political moderate")
    >>> agent = ConversationAgent(provider=provider, persona=persona)
    >>>
    >>> # Run an experiment
    >>> experiment = Experiment.from_config("config.yaml")
    >>> results = experiment.run()
"""

__version__ = "0.1.0"
__author__ = "Alex Newhouse"

# Import key classes for easy access
from synthetic_experiments.providers.base import LLMProvider
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.experiments.experiment import Experiment

__all__ = [
    "LLMProvider",
    "ConversationAgent",
    "Persona",
    "Experiment",
]
