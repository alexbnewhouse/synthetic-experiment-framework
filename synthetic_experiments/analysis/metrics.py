"""
Metrics and analysis tools for conversation experiments.

This module provides various metrics for analyzing conversations, including
basic statistics, linguistic features, and custom metric capabilities.

Example:
    >>> from synthetic_experiments.analysis import calculate_basic_metrics
    >>> from synthetic_experiments.data import ConversationLogger
    >>>
    >>> logger = ConversationLogger.from_json("conversation.json")
    >>> metrics = calculate_basic_metrics(logger)
    >>> print(f"Average message length: {metrics['avg_message_length']}")
"""

from typing import Dict, Any, List, Optional, Callable
import re
from dataclasses import dataclass
import logging

from synthetic_experiments.data.logger import ConversationLogger, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Container for conversation metrics."""
    conversation_id: str
    total_turns: int
    total_tokens: int
    total_cost: float
    duration_seconds: Optional[float]

    # Message statistics
    avg_message_length: float
    avg_tokens_per_turn: float
    message_length_std: float

    # Turn-taking statistics
    user_turns: int
    assistant_turns: int
    avg_user_message_length: float
    avg_assistant_message_length: float

    # Custom metrics
    custom_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "duration_seconds": self.duration_seconds,
            "avg_message_length": self.avg_message_length,
            "avg_tokens_per_turn": self.avg_tokens_per_turn,
            "message_length_std": self.message_length_std,
            "user_turns": self.user_turns,
            "assistant_turns": self.assistant_turns,
            "avg_user_message_length": self.avg_user_message_length,
            "avg_assistant_message_length": self.avg_assistant_message_length,
        }
        result.update(self.custom_metrics)
        return result


def calculate_basic_metrics(conversation: ConversationLogger) -> ConversationMetrics:
    """
    Calculate basic conversation metrics.

    Args:
        conversation: ConversationLogger to analyze

    Returns:
        ConversationMetrics with basic statistics
    """
    turns = conversation.turns

    if not turns:
        # Return empty metrics for empty conversation
        return ConversationMetrics(
            conversation_id=conversation.conversation_id,
            total_turns=0,
            total_tokens=0,
            total_cost=0.0,
            duration_seconds=conversation.get_duration(),
            avg_message_length=0.0,
            avg_tokens_per_turn=0.0,
            message_length_std=0.0,
            user_turns=0,
            assistant_turns=0,
            avg_user_message_length=0.0,
            avg_assistant_message_length=0.0,
            custom_metrics={}
        )

    # Basic counts
    total_turns = len(turns)
    total_tokens = conversation.get_total_tokens()
    total_cost = conversation.get_total_cost()

    # Message lengths
    message_lengths = [len(turn.message) for turn in turns]
    avg_message_length = sum(message_lengths) / len(message_lengths)

    # Standard deviation of message lengths
    mean_length = avg_message_length
    variance = sum((x - mean_length) ** 2 for x in message_lengths) / len(message_lengths)
    message_length_std = variance ** 0.5

    # Tokens per turn
    avg_tokens_per_turn = total_tokens / total_turns if total_turns > 0 else 0

    # Role-specific statistics
    user_messages = [turn.message for turn in turns if turn.role == "user"]
    assistant_messages = [turn.message for turn in turns if turn.role == "assistant"]

    user_turns = len(user_messages)
    assistant_turns = len(assistant_messages)

    avg_user_message_length = (
        sum(len(msg) for msg in user_messages) / len(user_messages)
        if user_messages else 0.0
    )

    avg_assistant_message_length = (
        sum(len(msg) for msg in assistant_messages) / len(assistant_messages)
        if assistant_messages else 0.0
    )

    return ConversationMetrics(
        conversation_id=conversation.conversation_id,
        total_turns=total_turns,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_seconds=conversation.get_duration(),
        avg_message_length=avg_message_length,
        avg_tokens_per_turn=avg_tokens_per_turn,
        message_length_std=message_length_std,
        user_turns=user_turns,
        assistant_turns=assistant_turns,
        avg_user_message_length=avg_user_message_length,
        avg_assistant_message_length=avg_assistant_message_length,
        custom_metrics={}
    )


def count_questions(text: str) -> int:
    """Count number of questions in text (simple heuristic)."""
    return text.count("?")


def count_exclamations(text: str) -> int:
    """Count number of exclamations in text."""
    return text.count("!")


def count_words(text: str) -> int:
    """Count number of words in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count number of sentences in text (simple heuristic)."""
    # Split on period, question mark, or exclamation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    return len([s for s in sentences if s.strip()])


def calculate_sentiment_simple(text: str) -> Dict[str, float]:
    """
    Simple sentiment analysis based on keyword matching.

    This is a basic implementation. For production use, consider
    using textblob or transformers-based sentiment analysis.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with sentiment scores
    """
    text_lower = text.lower()

    # Simple positive/negative word lists
    positive_words = [
        "good", "great", "excellent", "wonderful", "fantastic", "amazing",
        "helpful", "beneficial", "positive", "agree", "support", "love",
        "beautiful", "effective", "strong", "progress"
    ]

    negative_words = [
        "bad", "terrible", "awful", "horrible", "poor", "wrong",
        "harmful", "negative", "disagree", "oppose", "hate",
        "ugly", "ineffective", "weak", "problem", "fail"
    ]

    # Count occurrences
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)

    total_sentiment_words = positive_count + negative_count

    # Calculate polarity (-1 to 1)
    if total_sentiment_words > 0:
        polarity = (positive_count - negative_count) / total_sentiment_words
    else:
        polarity = 0.0

    return {
        "polarity": polarity,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sentiment_words": total_sentiment_words
    }


class MetricCalculator:
    """
    Extensible metric calculator for conversations.

    Allows registration of custom metric functions and batch processing
    of conversations.
    """

    def __init__(self):
        """Initialize metric calculator with default metrics."""
        self.metrics: Dict[str, Callable[[ConversationTurn], Any]] = {}

        # Register default metrics
        self.register_metric("question_count", lambda turn: count_questions(turn.message))
        self.register_metric("exclamation_count", lambda turn: count_exclamations(turn.message))
        self.register_metric("word_count", lambda turn: count_words(turn.message))
        self.register_metric("sentence_count", lambda turn: count_sentences(turn.message))

    def register_metric(self, name: str, func: Callable[[ConversationTurn], Any]) -> None:
        """
        Register a custom metric function.

        Args:
            name: Name of the metric
            func: Function that takes a ConversationTurn and returns a value
        """
        self.metrics[name] = func
        logger.info(f"Registered metric: {name}")

    def calculate_turn_metrics(self, turn: ConversationTurn) -> Dict[str, Any]:
        """
        Calculate all registered metrics for a single turn.

        Args:
            turn: ConversationTurn to analyze

        Returns:
            Dictionary of metric name -> value
        """
        results = {}
        for name, func in self.metrics.items():
            try:
                results[name] = func(turn)
            except Exception as e:
                logger.warning(f"Error calculating metric '{name}': {e}")
                results[name] = None

        return results

    def calculate_conversation_metrics(
        self,
        conversation: ConversationLogger
    ) -> List[Dict[str, Any]]:
        """
        Calculate metrics for all turns in a conversation.

        Args:
            conversation: ConversationLogger to analyze

        Returns:
            List of dictionaries, one per turn, with all metrics
        """
        results = []

        for turn in conversation.turns:
            turn_metrics = {
                "turn_number": turn.turn_number,
                "agent_name": turn.agent_name,
                "role": turn.role,
            }
            turn_metrics.update(self.calculate_turn_metrics(turn))
            results.append(turn_metrics)

        return results


def analyze_conversation_batch(
    conversation_paths: List[str],
    calculator: Optional[MetricCalculator] = None
) -> List[Dict[str, Any]]:
    """
    Analyze multiple conversations in batch.

    Args:
        conversation_paths: List of paths to conversation JSON files
        calculator: Optional custom MetricCalculator

    Returns:
        List of conversation metrics dictionaries
    """
    if calculator is None:
        calculator = MetricCalculator()

    results = []

    for path in conversation_paths:
        try:
            conversation = ConversationLogger.from_json(path)

            # Basic metrics
            basic_metrics = calculate_basic_metrics(conversation)

            # Custom metrics
            turn_metrics = calculator.calculate_conversation_metrics(conversation)

            results.append({
                "conversation_id": conversation.conversation_id,
                "basic_metrics": basic_metrics.to_dict(),
                "turn_metrics": turn_metrics
            })

        except Exception as e:
            logger.error(f"Error analyzing {path}: {e}")
            continue

    return results
