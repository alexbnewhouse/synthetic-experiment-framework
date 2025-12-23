"""
Ollama provider for local LLM models.

This provider enables running experiments with local models through Ollama,
which is ideal for privacy, cost control, and rapid iteration.

Example:
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.providers.base import Message, GenerationConfig
    >>>
    >>> provider = OllamaProvider(model_name="llama2")
    >>> messages = [Message(role="user", content="Hello!")]
    >>> result = provider.generate(messages)
    >>> print(result.message.content)
"""

from typing import List, Optional, Dict, Any
import logging

try:
    import ollama
except ImportError:
    raise ImportError(
        "Ollama package not found. Install with: pip install ollama\n"
        "Also ensure Ollama is installed on your system: https://ollama.ai"
    )

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult
)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    LLM provider for Ollama local models.

    Ollama provides easy access to local open-source models like Llama 2, Mistral,
    and others. This provider is the primary choice for experiments that prioritize
    privacy, cost control, and rapid iteration.

    Attributes:
        model_name: Name of the Ollama model (e.g., 'llama2', 'mistral', 'llama3')
        base_url: Ollama server URL (default: http://localhost:11434)
        auto_pull: Whether to automatically pull the model if not found
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        auto_pull: bool = True,
        timeout: int = 120,
        **config
    ):
        """
        Initialize Ollama provider.

        Args:
            model_name: Name of the Ollama model to use
            base_url: URL of Ollama server
            auto_pull: Automatically pull model if not available locally
            timeout: Request timeout in seconds
            **config: Additional configuration options
        """
        self.base_url = base_url
        self.auto_pull = auto_pull
        self.timeout = timeout
        super().__init__(model_name, **config)

        # Verify model is available
        self._ensure_model_available()

    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        if not self.model_name:
            raise ValueError("model_name is required for Ollama provider")

        # Check if Ollama is accessible
        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Ensure Ollama is running. Error: {e}"
            )

    def _ensure_model_available(self) -> None:
        """Ensure the model is available locally, pulling if necessary."""
        try:
            # Check if model exists
            models = ollama.list()
            model_names = [model['name'].split(':')[0] for model in models.get('models', [])]

            if self.model_name not in model_names:
                if self.auto_pull:
                    logger.info(f"Model '{self.model_name}' not found. Pulling from Ollama...")
                    ollama.pull(self.model_name)
                    logger.info(f"Successfully pulled model '{self.model_name}'")
                else:
                    raise ValueError(
                        f"Model '{self.model_name}' not found and auto_pull=False. "
                        f"Available models: {model_names}"
                    )
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")

    def generate(
        self,
        messages: List[Message],
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a response using Ollama.

        Args:
            messages: Conversation history as list of Messages
            generation_config: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            GenerationResult with the model's response

        Raises:
            ValueError: If messages is empty
            RuntimeError: If generation fails
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()

        # Convert our Message format to Ollama's format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Prepare generation options
        options = {
            "temperature": generation_config.temperature,
            "num_predict": generation_config.max_tokens,
        }

        if generation_config.top_p is not None:
            options["top_p"] = generation_config.top_p

        if generation_config.seed is not None:
            options["seed"] = generation_config.seed

        try:
            # Call Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                options=options,
                stream=generation_config.stream
            )

            # Handle streaming vs non-streaming
            if generation_config.stream:
                # For now, we'll collect the full stream
                # In future, this could yield chunks for real-time updates
                full_content = ""
                for chunk in response:
                    if 'message' in chunk and 'content' in chunk['message']:
                        full_content += chunk['message']['content']

                result_message = Message(
                    role="assistant",
                    content=full_content
                )
                tokens_used = None
                finish_reason = "stop"
            else:
                # Non-streaming response
                result_message = Message(
                    role=response['message']['role'],
                    content=response['message']['content']
                )

                # Extract token usage if available
                tokens_used = None
                if 'eval_count' in response:
                    tokens_used = response.get('prompt_eval_count', 0) + response.get('eval_count', 0)

                finish_reason = response.get('done_reason', 'stop')

            return GenerationResult(
                message=result_message,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                cost=0.0,  # Ollama is free (local)
                metadata={
                    "model": response.get('model', self.model_name),
                    "created_at": response.get('created_at'),
                    "total_duration": response.get('total_duration'),
                    "load_duration": response.get('load_duration'),
                    "prompt_eval_count": response.get('prompt_eval_count'),
                    "eval_count": response.get('eval_count'),
                }
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Failed to generate response with Ollama: {e}")

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Ollama model.

        Returns:
            ModelInfo with model capabilities
        """
        try:
            # Try to get model info from Ollama
            models = ollama.list()
            model_data = None

            for model in models.get('models', []):
                if model['name'].startswith(self.model_name):
                    model_data = model
                    break

            # Context window varies by model
            # These are approximate values for common models
            context_windows = {
                'llama2': 4096,
                'llama3': 8192,
                'mistral': 8192,
                'mixtral': 32768,
                'codellama': 16384,
                'phi': 2048,
            }

            context_window = context_windows.get(
                self.model_name.split(':')[0],
                4096  # default
            )

            return ModelInfo(
                provider="ollama",
                model_name=self.model_name,
                context_window=context_window,
                supports_streaming=True,
                cost_per_token=0.0,  # Local models are free
                metadata={
                    "size": model_data.get('size') if model_data else None,
                    "modified_at": model_data.get('modified_at') if model_data else None,
                }
            )
        except Exception as e:
            logger.warning(f"Could not get detailed model info: {e}")
            return ModelInfo(
                provider="ollama",
                model_name=self.model_name,
                context_window=4096,
                supports_streaming=True,
                cost_per_token=0.0
            )

    def list_available_models(self) -> List[str]:
        """
        List all models available in local Ollama installation.

        Returns:
            List of model names
        """
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            logger.error(f"Could not list Ollama models: {e}")
            return []
