"""
Tests for the Claude (Anthropic) provider.

Note: These tests use mocking to avoid requiring API keys and making real API calls.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from synthetic_experiments.providers.claude import ClaudeProvider
from synthetic_experiments.providers.base import (
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo
)


class TestClaudeProviderInit:
    """Tests for ClaudeProvider initialization."""

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_init_with_api_key(self, mock_anthropic):
        """Test initialization with explicit API key."""
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-api-key"
        )
        
        assert provider.model_name == "claude-3-opus-20240229"
        assert provider.api_key == "test-api-key"
        mock_anthropic.assert_called_once_with(api_key="test-api-key")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-api-key"})
    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_init_with_env_api_key(self, mock_anthropic):
        """Test initialization with API key from environment."""
        provider = ClaudeProvider(model_name="claude-3-opus-20240229")
        
        assert provider.api_key == "env-api-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        # Ensure no env var
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
        
        with pytest.raises(ValueError, match="Anthropic API key required"):
            ClaudeProvider(model_name="claude-3-opus-20240229")

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_init_validates_model_name(self, mock_anthropic):
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name is required"):
            ClaudeProvider(model_name="", api_key="test-key")


class TestClaudeProviderGenerate:
    """Tests for ClaudeProvider.generate method."""

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_generate_basic(self, mock_anthropic):
        """Test basic generation."""
        # Setup mock response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello there!")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        messages = [Message(role="user", content="Hi")]
        
        result = provider.generate(messages)
        
        assert isinstance(result, GenerationResult)
        assert result.message.role == "assistant"

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_generate_empty_messages_raises(self, mock_anthropic):
        """Test that empty messages raises ValueError."""
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            provider.generate([])

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_generate_with_system_message(self, mock_anthropic):
        """Test generation with system message."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I am helpful.")]
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Who are you?")
        ]
        
        result = provider.generate(messages)
        
        # Verify system message was passed separately
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "You are a helpful assistant."

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_generate_with_config(self, mock_anthropic):
        """Test generation with custom config."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        
        messages = [Message(role="user", content="Test")]
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop_sequences=["END"]
        )
        
        result = provider.generate(messages, config)
        
        # Verify config was passed
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


class TestClaudeProviderModelInfo:
    """Tests for ClaudeProvider.get_model_info method."""

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_get_model_info(self, mock_anthropic):
        """Test getting model info."""
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        
        info = provider.get_model_info()
        
        assert isinstance(info, ModelInfo)
        assert info.provider == "claude"
        assert info.model_name == "claude-3-opus-20240229"

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_context_window_known_model(self, mock_anthropic):
        """Test context window for known models."""
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        
        info = provider.get_model_info()
        
        # Claude 3 models have 200k context window
        assert info.context_window == 200000


class TestClaudeProviderPricing:
    """Tests for Claude pricing calculations."""

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_pricing_known_model(self, mock_anthropic):
        """Test that pricing exists for known models."""
        provider = ClaudeProvider(
            model_name="claude-3-opus-20240229",
            api_key="test-key"
        )
        
        # Check that pricing is defined
        assert "claude-3-opus-20240229" in ClaudeProvider.MODEL_PRICING
        
        input_cost, output_cost = ClaudeProvider.MODEL_PRICING["claude-3-opus-20240229"]
        assert input_cost > 0
        assert output_cost > 0

    @patch('synthetic_experiments.providers.claude.Anthropic')
    def test_all_models_have_pricing(self, mock_anthropic):
        """Test that all listed models have pricing info."""
        for model in ClaudeProvider.CONTEXT_WINDOWS.keys():
            assert model in ClaudeProvider.MODEL_PRICING
