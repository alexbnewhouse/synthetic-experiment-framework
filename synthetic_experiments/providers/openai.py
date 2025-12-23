"""
OpenAI provider for cloud-based LLM access.

This provider enables running experiments with OpenAI models (GPT-4, GPT-3.5, etc.)
via the OpenAI API. Useful for comparing different model families in research.

Example:
    >>> import os
    >>> os.environ['OPENAI_API_KEY'] = 'your-api-key'
    >>> from synthetic_experiments.providers import OpenAIProvider
    >>> from synthetic_experiments.providers.base import Message
    >>>
    >>> provider = OpenAIProvider(model_name="gpt-4")
    >>> messages = [Message(role="user", content="Hello!")]
    >>> result = provider.generate(messages)
"""

from typing import List, Optional
import os
import logging

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI package not found. Install with: pip install openai"
    )

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    LLM provider for OpenAI models.

    Requires an OpenAI API key, which should be set in the environment
    variable OPENAI_API_KEY or passed during initialization.

    Attributes:
        model_name: OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        api_key: OpenAI API key (if not set via environment variable)
    """

    # Model pricing per million tokens (input, output)
    MODEL_PRICING = {
        "gpt-4": (30.00, 60.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4o": (5.00, 15.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "gpt-4o-mini": (0.15, 0.60),
    }

    # Model context windows
    CONTEXT_WINDOWS = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gpt-3.5-turbo": 16385,
        "gpt-4o-mini": 128000,
    }

    def __init__(self, model_name: str, api_key: Optional[str] = None, **config):
        """
        Initialize OpenAI provider.

        Args:
            model_name: Name of OpenAI model to use
            api_key: OpenAI API key (optional, can use OPENAI_API_KEY env var)
            **config: Additional configuration
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        super().__init__(model_name, **config)

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.model_name:
            raise ValueError("model_name is required for OpenAI provider")

    def generate(
        self,
        messages: List[Message],
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a response using OpenAI.

        Args:
            messages: Conversation history
            generation_config: Generation parameters

        Returns:
            GenerationResult with OpenAI's response
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if generation_config is None:
            generation_config = GenerationConfig()

        # Convert our Message format to OpenAI's format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Prepare API call parameters
        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": generation_config.max_tokens,
            "temperature": generation_config.temperature,
        }

        if generation_config.top_p is not None:
            params["top_p"] = generation_config.top_p

        if generation_config.stop_sequences:
            params["stop"] = generation_config.stop_sequences

        if generation_config.seed is not None:
            params["seed"] = generation_config.seed

        try:
            response = self.client.chat.completions.create(**params)

            # Extract response
            choice = response.choices[0]
            result_message = Message(
                role=choice.message.role,
                content=choice.message.content or ""
            )

            # Calculate cost
            input_cost, output_cost = self.MODEL_PRICING.get(
                self.model_name,
                (0, 0)
            )
            total_cost = (
                (response.usage.prompt_tokens / 1_000_000) * input_cost +
                (response.usage.completion_tokens / 1_000_000) * output_cost
            )

            return GenerationResult(
                message=result_message,
                tokens_used=response.usage.total_tokens,
                finish_reason=choice.finish_reason or "stop",
                cost=total_cost,
                metadata={
                    "model": response.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": choice.finish_reason,
                }
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"Failed to generate response with OpenAI: {e}")

    def get_model_info(self) -> ModelInfo:
        """Get information about the OpenAI model."""
        input_cost, output_cost = self.MODEL_PRICING.get(
            self.model_name,
            (0, 0)
        )
        avg_cost = (input_cost + output_cost) / 2 / 1_000_000

        return ModelInfo(
            provider="openai",
            model_name=self.model_name,
            context_window=self.CONTEXT_WINDOWS.get(self.model_name, 8192),
            supports_streaming=True,
            cost_per_token=avg_cost,
            metadata={
                "input_cost_per_mtok": input_cost,
                "output_cost_per_mtok": output_cost,
            }
        )
