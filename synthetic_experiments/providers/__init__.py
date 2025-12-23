"""
LLM provider implementations for synthetic experiments.

This module provides a unified interface for interacting with different
LLM providers (local and cloud-based).

Available Providers:
- OllamaProvider: Local models via Ollama (primary choice)
- ClaudeProvider: Anthropic's Claude models
- OpenAIProvider: OpenAI's GPT models

Example:
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.providers.base import Message
    >>>
    >>> provider = OllamaProvider(model_name="llama2")
    >>> messages = [Message(role="user", content="Hello!")]
    >>> result = provider.generate(messages)
    >>> print(result.message.content)
"""

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult
)
from synthetic_experiments.providers.ollama import OllamaProvider
from synthetic_experiments.providers.claude import ClaudeProvider
from synthetic_experiments.providers.openai import OpenAIProvider

__all__ = [
    "LLMProvider",
    "Message",
    "ModelInfo",
    "GenerationConfig",
    "GenerationResult",
    "OllamaProvider",
    "ClaudeProvider",
    "OpenAIProvider",
]
