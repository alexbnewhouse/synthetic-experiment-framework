"""
Analysis tools for synthetic experiments.

This module provides various tools for analyzing conversation data,
including basic metrics, political language analysis, and custom metric
calculation.

Example:
    >>> from synthetic_experiments.analysis import calculate_basic_metrics
    >>> from synthetic_experiments.analysis.political import detect_political_language
    >>> from synthetic_experiments.data import ConversationLogger
    >>>
    >>> conversation = ConversationLogger.from_json("conversation.json")
    >>> metrics = calculate_basic_metrics(conversation)
    >>> political_analysis = detect_political_language(conversation.turns[0].message)
"""

from synthetic_experiments.analysis.metrics import (
    ConversationMetrics,
    calculate_basic_metrics,
    MetricCalculator,
    count_questions,
    count_words,
    count_sentences,
    calculate_sentiment_simple
)

from synthetic_experiments.analysis.political import (
    PoliticalLanguageAnalysis,
    detect_political_language,
    analyze_conversation_polarization,
    calculate_opinion_shift,
    count_agreement_disagreement
)

__all__ = [
    # Metrics
    "ConversationMetrics",
    "calculate_basic_metrics",
    "MetricCalculator",
    "count_questions",
    "count_words",
    "count_sentences",
    "calculate_sentiment_simple",
    # Political analysis
    "PoliticalLanguageAnalysis",
    "detect_political_language",
    "analyze_conversation_polarization",
    "calculate_opinion_shift",
    "count_agreement_disagreement",
]
