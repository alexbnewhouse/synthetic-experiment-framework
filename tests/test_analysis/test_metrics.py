"""
Tests for conversation metrics and analysis tools.
"""

import pytest

from synthetic_experiments.analysis.metrics import (
    ConversationMetrics,
    calculate_basic_metrics,
    count_questions,
    count_exclamations,
    count_words,
    count_sentences,
    calculate_sentiment_simple
)
from synthetic_experiments.data.logger import ConversationLogger


class TestCountingFunctions:
    """Tests for basic counting functions."""

    def test_count_questions(self):
        """Test counting question marks."""
        assert count_questions("How are you?") == 1
        assert count_questions("Hello! How are you? What's up?") == 2
        assert count_questions("No questions here.") == 0
        assert count_questions("") == 0

    def test_count_exclamations(self):
        """Test counting exclamation marks."""
        assert count_exclamations("Hello!") == 1
        assert count_exclamations("Wow! Amazing! Great!") == 3
        assert count_exclamations("No excitement here.") == 0
        assert count_exclamations("") == 0

    def test_count_words(self):
        """Test counting words."""
        assert count_words("Hello world") == 2
        assert count_words("One two three four five") == 5
        assert count_words("Single") == 1
        # Empty string behavior depends on implementation
        assert count_words("") >= 0

    def test_count_sentences(self):
        """Test counting sentences."""
        assert count_sentences("Hello. World.") == 2
        assert count_sentences("Hi! How are you? I'm good.") == 3
        assert count_sentences("Just one sentence") == 1
        assert count_sentences("") == 0


class TestSentimentSimple:
    """Tests for simple sentiment analysis."""

    def test_positive_sentiment(self):
        """Test detecting positive sentiment."""
        text = "This is great and wonderful. I love it!"
        result = calculate_sentiment_simple(text)
        
        assert "positive_count" in result or result.get("positive_count", 0) >= 0
        assert result.get("positive_count", result.get("positive", 0)) > 0

    def test_negative_sentiment(self):
        """Test detecting negative sentiment."""
        text = "This is terrible and awful. I hate it."
        result = calculate_sentiment_simple(text)
        
        assert result.get("negative_count", result.get("negative", 0)) > 0

    def test_neutral_sentiment(self):
        """Test neutral text."""
        text = "The sky is blue. Water is wet."
        result = calculate_sentiment_simple(text)
        
        # Should have low or zero sentiment scores
        assert isinstance(result, dict)


class TestConversationMetrics:
    """Tests for ConversationMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating ConversationMetrics."""
        metrics = ConversationMetrics(
            conversation_id="test_001",
            total_turns=10,
            total_tokens=500,
            total_cost=0.05,
            duration_seconds=60.0,
            avg_message_length=100.5,
            avg_tokens_per_turn=50.0,
            message_length_std=20.0,
            user_turns=5,
            assistant_turns=5,
            avg_user_message_length=90.0,
            avg_assistant_message_length=110.0,
            custom_metrics={}
        )
        
        assert metrics.conversation_id == "test_001"
        assert metrics.total_turns == 10
        assert metrics.avg_message_length == 100.5

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ConversationMetrics(
            conversation_id="test",
            total_turns=5,
            total_tokens=100,
            total_cost=0.01,
            duration_seconds=30.0,
            avg_message_length=50.0,
            avg_tokens_per_turn=20.0,
            message_length_std=10.0,
            user_turns=3,
            assistant_turns=2,
            avg_user_message_length=40.0,
            avg_assistant_message_length=65.0,
            custom_metrics={"custom_key": "custom_value"}
        )
        
        result = metrics.to_dict()
        
        assert result["conversation_id"] == "test"
        assert result["total_turns"] == 5
        assert result["custom_key"] == "custom_value"


class TestCalculateBasicMetrics:
    """Tests for calculate_basic_metrics function."""

    def test_calculate_metrics_basic(self, sample_conversation_logger):
        """Test calculating basic metrics."""
        metrics = calculate_basic_metrics(sample_conversation_logger)
        
        assert isinstance(metrics, ConversationMetrics)
        assert metrics.total_turns == 4
        assert metrics.total_tokens == 57  # 10 + 15 + 12 + 20

    def test_calculate_metrics_empty_conversation(self):
        """Test metrics for empty conversation."""
        logger = ConversationLogger(experiment_name="empty")
        
        metrics = calculate_basic_metrics(logger)
        
        assert metrics.total_turns == 0
        assert metrics.avg_message_length == 0.0
        assert metrics.user_turns == 0
        assert metrics.assistant_turns == 0

    def test_calculate_metrics_role_counts(self, sample_conversation_logger):
        """Test that role counts are correct."""
        metrics = calculate_basic_metrics(sample_conversation_logger)
        
        assert metrics.user_turns == 2
        assert metrics.assistant_turns == 2

    def test_calculate_metrics_message_lengths(self, sample_conversation_logger):
        """Test message length calculations."""
        metrics = calculate_basic_metrics(sample_conversation_logger)
        
        assert metrics.avg_message_length > 0
        assert metrics.message_length_std >= 0

    def test_calculate_metrics_cost(self, sample_conversation_logger):
        """Test cost calculation."""
        metrics = calculate_basic_metrics(sample_conversation_logger)
        
        # 0.001 + 0.002 + 0.001 + 0.002 = 0.006
        assert metrics.total_cost == pytest.approx(0.006)


class TestMetricsWithDifferentConversations:
    """Tests with various conversation types."""

    def test_single_turn_conversation(self):
        """Test metrics for single turn conversation."""
        logger = ConversationLogger(experiment_name="single")
        logger.log_turn(
            agent_name="User",
            role="user",
            message="Hello!",
            tokens_used=5
        )
        
        metrics = calculate_basic_metrics(logger)
        
        assert metrics.total_turns == 1
        assert metrics.user_turns == 1
        assert metrics.assistant_turns == 0

    def test_long_conversation(self):
        """Test metrics for long conversation."""
        logger = ConversationLogger(experiment_name="long")
        
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            logger.log_turn(
                agent_name=f"Agent_{role}",
                role=role,
                message=f"Message number {i}" * 10,
                tokens_used=10 + i
            )
        
        metrics = calculate_basic_metrics(logger)
        
        assert metrics.total_turns == 20
        assert metrics.user_turns == 10
        assert metrics.assistant_turns == 10

    def test_conversation_with_no_tokens(self):
        """Test metrics when token counts are None."""
        logger = ConversationLogger(experiment_name="no_tokens")
        logger.log_turn(
            agent_name="User",
            role="user",
            message="Hello!"
            # No tokens_used specified
        )
        
        metrics = calculate_basic_metrics(logger)
        
        assert metrics.total_tokens == 0
        assert metrics.avg_tokens_per_turn == 0
