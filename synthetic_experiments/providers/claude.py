"""
Anthropic Claude provider for cloud-based LLM access.

This provider enables running experiments with Claude models via the Anthropic API.
Useful when you need Claude's capabilities or want to compare local vs cloud models.

Example:
    >>> import os
    >>> os.environ['ANTHROPIC_API_KEY'] = 'your-api-key'
    >>> from synthetic_experiments.providers import ClaudeProvider
    >>> from synthetic_experiments.providers.base import Message
    >>>
    >>> provider = ClaudeProvider(model_name="claude-3-opus-20240229")
    >>> messages = [Message(role="user", content="Hello!")]
    >>> result = provider.generate(messages)
"""

from typing import List, Optional
import os
import logging

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError(
        "Anthropic package not found. Install with: pip install anthropic"
    )

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult
)

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """
    LLM provider for Anthropic's Claude models.

    Requires an Anthropic API key, which should be set in the environment
    variable ANTHROPIC_API_KEY or passed during initialization.

    Attributes:
        model_name: Claude model to use (e.g., 'claude-3-opus-20240229')
        api_key: Anthropic API key (if not set via environment variable)
    """

    # Model pricing per million tokens (input, output)
    MODEL_PRICING = {
        "claude-3-opus-20240229": (15.00, 75.00),
        "claude-3-sonnet-20240229": (3.00, 15.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-5-sonnet-20240620": (3.00, 15.00),
    }

    # Model context windows
    CONTEXT_WINDOWS = {
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20240620": 200000,
    }

    def __init__(self, model_name: str, api_key: Optional[str] = None, **config):
        """
        Initialize Claude provider.

        Args:
            model_name: Name of Claude model to use
            api_key: Anthropic API key (optional, can use ANTHROPIC_API_KEY env var)
            **config: Additional configuration
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        super().__init__(model_name, **config)

    def validate_config(self) -> None:
        """Validate Claude configuration."""
        if not self.model_name:
            raise ValueError("model_name is required for Claude provider")

        # Verify API key works by making a small test call
        # We skip this for now to avoid unnecessary API calls during init

    def generate(
        self,
        messages: List[Message],
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a response using Claude.

        Args:
            messages: Conversation history
            generation_config: Generation parameters

        Returns:
            GenerationResult with Claude's response
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if generation_config is None:
            generation_config = GenerationConfig()

        # Separate system message from conversation messages
        system_message = None
        conversation_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Prepare API call parameters
        params = {
            "model": self.model_name,
            "messages": conversation_messages,
            "max_tokens": generation_config.max_tokens,
            "temperature": generation_config.temperature,
        }

        if system_message:
            params["system"] = system_message

        if generation_config.top_p is not None:
            params["top_p"] = generation_config.top_p

        if generation_config.stop_sequences:
            params["stop_sequences"] = generation_config.stop_sequences

        try:
            response = self.client.messages.create(**params)

            # Extract response content
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text

            result_message = Message(
                role="assistant",
                content=content
            )

            # Calculate cost
            input_cost, output_cost = self.MODEL_PRICING.get(
                self.model_name,
                (0, 0)
            )
            total_cost = (
                (response.usage.input_tokens / 1_000_000) * input_cost +
                (response.usage.output_tokens / 1_000_000) * output_cost
            )

            return GenerationResult(
                message=result_message,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason or "stop",
                cost=total_cost,
                metadata={
                    "model": response.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason,
                }
            )

        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise RuntimeError(f"Failed to generate response with Claude: {e}")

    def get_model_info(self) -> ModelInfo:
        """Get information about the Claude model."""
        input_cost, output_cost = self.MODEL_PRICING.get(
            self.model_name,
            (0, 0)
        )
        avg_cost = (input_cost + output_cost) / 2 / 1_000_000

        return ModelInfo(
            provider="claude",
            model_name=self.model_name,
            context_window=self.CONTEXT_WINDOWS.get(self.model_name, 200000),
            supports_streaming=True,
            cost_per_token=avg_cost,
            metadata={
                "input_cost_per_mtok": input_cost,
                "output_cost_per_mtok": output_cost,
            }
        )
