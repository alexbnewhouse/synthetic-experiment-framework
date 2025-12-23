"""
Tests for the OpenAI provider.

Note: These tests use mocking to avoid requiring API keys and making real API calls.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from synthetic_experiments.providers.openai import OpenAIProvider
from synthetic_experiments.providers.base import (
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo
)


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider initialization."""

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_init_with_api_key(self, mock_openai):
        """Test initialization with explicit API key."""
        provider = OpenAIProvider(
            model_name="gpt-4",
            api_key="test-api-key"
        )
        
        assert provider.model_name == "gpt-4"
        assert provider.api_key == "test-api-key"
        mock_openai.assert_called_once_with(api_key="test-api-key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"})
    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_init_with_env_api_key(self, mock_openai):
        """Test initialization with API key from environment."""
        provider = OpenAIProvider(model_name="gpt-4")
        
        assert provider.api_key == "env-api-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        # Ensure no env var
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(model_name="gpt-4")

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_init_validates_model_name(self, mock_openai):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name is required"):
            OpenAIProvider(model_name="", api_key="test-key")


class TestOpenAIProviderGenerate:
    """Tests for OpenAIProvider.generate method."""

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_generate_basic(self, mock_openai):
        """Test basic generation."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Hello there!"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        messages = [Message(role="user", content="Hi")]
        
        result = provider.generate(messages)
        
        assert isinstance(result, GenerationResult)
        assert result.message.role == "assistant"
        assert result.message.content == "Hello there!"

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_generate_empty_messages_raises(self, mock_openai):
        """Test that empty messages raises ValueError."""
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            provider.generate([])

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_generate_with_system_message(self, mock_openai):
        """Test generation with system message."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "I am helpful."
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 5
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Who are you?")
        ]
        
        result = provider.generate(messages)
        
        # Verify messages were converted correctly
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 2

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_generate_with_config(self, mock_openai):
        """Test generation with custom config."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        mock_client.chat.completions.create.return_value = mock_response
        
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        
        messages = [Message(role="user", content="Test")]
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop_sequences=["END"],
            seed=42
        )
        
        result = provider.generate(messages, config)
        
        # Verify config was passed
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["seed"] == 42


class TestOpenAIProviderModelInfo:
    """Tests for OpenAIProvider.get_model_info method."""

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_get_model_info(self, mock_openai):
        """Test getting model info."""
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        
        info = provider.get_model_info()
        
        assert isinstance(info, ModelInfo)
        assert info.provider == "openai"
        assert info.model_name == "gpt-4"

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_context_window_known_model(self, mock_openai):
        """Test context window for known models."""
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        
        info = provider.get_model_info()
        
        assert info.context_window == 8192

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_context_window_turbo_model(self, mock_openai):
        """Test context window for turbo models."""
        provider = OpenAIProvider(model_name="gpt-4-turbo", api_key="test-key")
        
        info = provider.get_model_info()
        
        assert info.context_window == 128000


class TestOpenAIProviderPricing:
    """Tests for OpenAI pricing calculations."""

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_pricing_known_model(self, mock_openai):
        """Test that pricing exists for known models."""
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-key")
        
        # Check that pricing is defined
        assert "gpt-4" in OpenAIProvider.MODEL_PRICING
        
        input_cost, output_cost = OpenAIProvider.MODEL_PRICING["gpt-4"]
        assert input_cost > 0
        assert output_cost > 0

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_all_models_have_pricing(self, mock_openai):
        """Test that all listed models have pricing info."""
        for model in OpenAIProvider.CONTEXT_WINDOWS.keys():
            assert model in OpenAIProvider.MODEL_PRICING

    @patch('synthetic_experiments.providers.openai.OpenAI')
    def test_gpt4_more_expensive_than_gpt35(self, mock_openai):
        """Test that GPT-4 is more expensive than GPT-3.5."""
        gpt4_input, gpt4_output = OpenAIProvider.MODEL_PRICING["gpt-4"]
        gpt35_input, gpt35_output = OpenAIProvider.MODEL_PRICING["gpt-3.5-turbo"]
        
        assert gpt4_input > gpt35_input
        assert gpt4_output > gpt35_output
