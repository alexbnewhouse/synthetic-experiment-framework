"""
Base classes for LLM providers.

This module defines the abstract interfaces that all LLM providers must implement,
ensuring consistent behavior across different models and APIs (Ollama, Claude, OpenAI, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Message:
    """
    Represents a single message in a conversation.

    This standardized format works across all LLM providers, abstracting away
    provider-specific message formats.

    Attributes:
        role: The role of the message sender ('user', 'assistant', or 'system')
        content: The text content of the message
        timestamp: When the message was created (auto-generated)
        metadata: Additional provider-specific or experiment-specific data
    """
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for storage/serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            from dateutil import parser
            data["timestamp"] = parser.parse(data["timestamp"])
        return cls(**data)


@dataclass
class ModelInfo:
    """
    Information about an LLM model.

    Attributes:
        provider: Name of the provider (e.g., 'ollama', 'claude', 'openai')
        model_name: Specific model identifier (e.g., 'llama2', 'claude-3-opus')
        context_window: Maximum number of tokens the model can process
        supports_streaming: Whether the model supports streaming responses
        cost_per_token: Cost per token (if applicable, None for local models)
        metadata: Additional model-specific information
    """
    provider: str
    model_name: str
    context_window: Optional[int] = None
    supports_streaming: bool = True
    cost_per_token: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        temperature: Controls randomness (0.0 = deterministic, higher = more random)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter (alternative to temperature)
        stop_sequences: List of sequences that stop generation
        stream: Whether to stream the response
        seed: Random seed for reproducibility (if supported by provider)
    """
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """
    Result from a text generation call.

    Attributes:
        message: The generated message
        tokens_used: Number of tokens used in generation
        finish_reason: Why generation stopped ('stop', 'length', 'error', etc.)
        cost: Cost of the generation (if applicable)
        metadata: Additional provider-specific data (model version, latency, etc.)
    """
    message: Message
    tokens_used: Optional[int] = None
    finish_reason: str = "stop"
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    This class defines the interface that all provider implementations
    (Ollama, Claude, OpenAI, etc.) must follow. It ensures consistent
    behavior and makes it easy to swap providers in experiments.

    Attributes:
        model_name: The specific model to use (e.g., 'llama2', 'claude-3-opus')
        config: Provider-specific configuration
    """

    def __init__(self, model_name: str, **config):
        """
        Initialize the provider.

        Args:
            model_name: Name of the model to use
            **config: Additional provider-specific configuration
        """
        self.model_name = model_name
        self.config = config
        self.validate_config()

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a response given a conversation history.

        This is the core method that all providers must implement. It takes
        a list of messages (the conversation history) and returns a generated
        response.

        Args:
            messages: List of messages representing the conversation history
            generation_config: Configuration for text generation (temperature, etc.)

        Returns:
            GenerationResult containing the generated message and metadata

        Raises:
            ValueError: If messages are invalid or empty
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the provider configuration.

        Called during initialization to ensure all required configuration
        is present and valid. Should raise ValueError if config is invalid.

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model.

        Returns:
            ModelInfo object with model capabilities and metadata
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Default implementation uses a simple approximation (words * 1.3).
        Providers should override with more accurate implementations if available.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        # Simple approximation: ~1.3 tokens per word on average
        words = len(text.split())
        return int(words * 1.3)

    def count_message_tokens(self, messages: List[Message]) -> int:
        """
        Count total tokens in a list of messages.

        Args:
            messages: List of messages

        Returns:
            Total estimated token count
        """
        total = 0
        for message in messages:
            # Add tokens for role and content
            total += self.estimate_tokens(message.role)
            total += self.estimate_tokens(message.content)
        return total

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
