"""
Tests for base provider classes and data structures.
"""

import pytest
from datetime import datetime
from dateutil import parser

from synthetic_experiments.providers.base import (
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult,
    LLMProvider
)


class TestMessage:
    """Tests for the Message dataclass."""

    def test_message_creation_minimal(self):
        """Test creating a message with minimal parameters."""
        msg = Message(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}

    def test_message_creation_full(self):
        """Test creating a message with all parameters."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        metadata = {"key": "value"}
        
        msg = Message(
            role="assistant",
            content="How can I help?",
            timestamp=timestamp,
            metadata=metadata
        )
        
        assert msg.role == "assistant"
        assert msg.content == "How can I help?"
        assert msg.timestamp == timestamp
        assert msg.metadata == {"key": "value"}

    def test_message_to_dict(self):
        """Test serializing message to dictionary."""
        msg = Message(role="user", content="Test")
        result = msg.to_dict()
        
        assert result["role"] == "user"
        assert result["content"] == "Test"
        assert "timestamp" in result
        assert result["metadata"] == {}

    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "role": "assistant",
            "content": "Response",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {"test": True}
        }
        
        msg = Message.from_dict(data)
        
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.timestamp.year == 2024
        assert msg.metadata == {"test": True}

    def test_message_from_dict_without_timestamp(self):
        """Test creating message from dict without timestamp."""
        data = {
            "role": "user",
            "content": "Hello"
        }
        
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_roles(self):
        """Test different message roles."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="Test")
            assert msg.role == role


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_minimal(self):
        """Test creating ModelInfo with minimal parameters."""
        info = ModelInfo(provider="test", model_name="test-model")
        
        assert info.provider == "test"
        assert info.model_name == "test-model"
        assert info.context_window is None
        assert info.supports_streaming is True
        assert info.cost_per_token is None
        assert info.metadata == {}

    def test_model_info_full(self):
        """Test creating ModelInfo with all parameters."""
        info = ModelInfo(
            provider="openai",
            model_name="gpt-4",
            context_window=8192,
            supports_streaming=True,
            cost_per_token=0.00003,
            metadata={"version": "latest"}
        )
        
        assert info.provider == "openai"
        assert info.model_name == "gpt-4"
        assert info.context_window == 8192
        assert info.supports_streaming is True
        assert info.cost_per_token == 0.00003
        assert info.metadata == {"version": "latest"}


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        config = GenerationConfig()
        
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p is None
        assert config.stop_sequences == []
        assert config.stream is False
        assert config.seed is None

    def test_generation_config_custom(self):
        """Test GenerationConfig with custom values."""
        config = GenerationConfig(
            temperature=0.5,
            max_tokens=500,
            top_p=0.9,
            stop_sequences=["END", "STOP"],
            stream=True,
            seed=42
        )
        
        assert config.temperature == 0.5
        assert config.max_tokens == 500
        assert config.top_p == 0.9
        assert config.stop_sequences == ["END", "STOP"]
        assert config.stream is True
        assert config.seed == 42

    def test_generation_config_temperature_bounds(self):
        """Test that temperature can be set to boundary values."""
        # Minimum temperature (deterministic)
        config_min = GenerationConfig(temperature=0.0)
        assert config_min.temperature == 0.0
        
        # Higher temperature (more random)
        config_max = GenerationConfig(temperature=2.0)
        assert config_max.temperature == 2.0


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_generation_result_minimal(self):
        """Test GenerationResult with minimal parameters."""
        msg = Message(role="assistant", content="Response")
        result = GenerationResult(message=msg)
        
        assert result.message == msg
        assert result.tokens_used is None
        assert result.finish_reason == "stop"
        assert result.cost is None
        assert result.metadata == {}

    def test_generation_result_full(self):
        """Test GenerationResult with all parameters."""
        msg = Message(role="assistant", content="Response")
        result = GenerationResult(
            message=msg,
            tokens_used=100,
            finish_reason="length",
            cost=0.01,
            metadata={"latency_ms": 500}
        )
        
        assert result.message == msg
        assert result.tokens_used == 100
        assert result.finish_reason == "length"
        assert result.cost == 0.01
        assert result.metadata == {"latency_ms": 500}


class TestLLMProviderAbstract:
    """Tests for abstract LLMProvider base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider(model_name="test")

    def test_mock_provider_implements_interface(self, mock_provider):
        """Test that MockLLMProvider properly implements the interface."""
        # Should have all required methods
        assert hasattr(mock_provider, 'generate')
        assert hasattr(mock_provider, 'validate_config')
        assert hasattr(mock_provider, 'get_model_info')
        assert hasattr(mock_provider, 'estimate_tokens')

    def test_mock_provider_generate(self, mock_provider):
        """Test mock provider generate method."""
        messages = [Message(role="user", content="Hello")]
        result = mock_provider.generate(messages)
        
        assert isinstance(result, GenerationResult)
        assert result.message.role == "assistant"
        assert result.message.content == "This is a mock response."

    def test_mock_provider_empty_messages_raises(self, mock_provider):
        """Test that generate raises ValueError for empty messages."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            mock_provider.generate([])

    def test_mock_provider_get_model_info(self, mock_provider):
        """Test mock provider get_model_info method."""
        info = mock_provider.get_model_info()
        
        assert isinstance(info, ModelInfo)
        assert info.provider == "mock"
        assert info.model_name == "mock-model"

    def test_mock_provider_with_custom_responses(self, mock_provider_with_responses):
        """Test mock provider with custom responses."""
        responses = ["Response 1", "Response 2", "Response 3"]
        provider = mock_provider_with_responses(responses)
        
        messages = [Message(role="user", content="Test")]
        
        for i, expected in enumerate(responses):
            result = provider.generate(messages)
            assert result.message.content == expected

    def test_mock_provider_cycles_responses(self, mock_provider_with_responses):
        """Test that mock provider cycles through responses."""
        responses = ["A", "B"]
        provider = mock_provider_with_responses(responses)
        messages = [Message(role="user", content="Test")]
        
        # First cycle
        assert provider.generate(messages).message.content == "A"
        assert provider.generate(messages).message.content == "B"
        
        # Second cycle
        assert provider.generate(messages).message.content == "A"
        assert provider.generate(messages).message.content == "B"

    def test_mock_provider_call_count(self, mock_provider):
        """Test that call count is tracked."""
        messages = [Message(role="user", content="Test")]
        
        assert mock_provider.call_count == 0
        
        mock_provider.generate(messages)
        assert mock_provider.call_count == 1
        
        mock_provider.generate(messages)
        assert mock_provider.call_count == 2

    def test_mock_provider_reset(self, mock_provider_with_responses):
        """Test mock provider reset functionality."""
        provider = mock_provider_with_responses(["A", "B"])
        messages = [Message(role="user", content="Test")]
        
        provider.generate(messages)
        provider.generate(messages)
        
        assert provider.call_count == 2
        
        provider.reset()
        
        assert provider.call_count == 0
        # Should start from first response again
        assert provider.generate(messages).message.content == "A"
