"""
Political language analysis tools.

This module provides specialized analysis for political discourse,
including political language detection, polarization metrics, and
opinion shift tracking.

Example:
    >>> from synthetic_experiments.analysis.political import detect_political_language
    >>>
    >>> text = "We need strong borders and immigration reform"
    >>> political_markers = detect_political_language(text)
    >>> print(political_markers["conservative_markers"])
"""

from typing import Dict, List, Any, Set
import re
from dataclasses import dataclass

from synthetic_experiments.data.logger import ConversationLogger


# Political language dictionaries
# These are simplified examples - expand based on your research needs

CONSERVATIVE_LANGUAGE = {
    "economy": ["free market", "capitalism", "small business", "entrepreneurship", "deregulation"],
    "government": ["limited government", "states' rights", "personal responsibility", "freedom", "liberty"],
    "security": ["law and order", "strong borders", "national security", "military strength"],
    "values": ["traditional values", "family values", "religious freedom", "second amendment"],
    "general": ["conservative", "right-wing", "republican", "fiscal responsibility"]
}

LIBERAL_LANGUAGE = {
    "social": ["equality", "social justice", "diversity", "inclusion", "equity"],
    "environment": ["climate change", "renewable energy", "sustainability", "green", "environmental"],
    "government": ["universal healthcare", "public education", "social safety net", "collective action"],
    "rights": ["civil rights", "human rights", "workers' rights", "voting rights"],
    "general": ["liberal", "progressive", "left-wing", "democratic", "reform"]
}

POLARIZING_TOPICS = [
    "abortion", "gun control", "immigration", "climate change",
    "healthcare", "taxes", "welfare", "affirmative action",
    "police reform", "voting laws"
]


@dataclass
class PoliticalLanguageAnalysis:
    """Results of political language analysis."""
    conservative_markers: List[str]
    liberal_markers: List[str]
    polarizing_topics: List[str]
    conservative_score: float
    liberal_score: float
    polarization_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conservative_markers": self.conservative_markers,
            "liberal_markers": self.liberal_markers,
            "polarizing_topics": self.polarizing_topics,
            "conservative_score": self.conservative_score,
            "liberal_score": self.liberal_score,
            "polarization_score": self.polarization_score
        }


def detect_political_language(text: str) -> PoliticalLanguageAnalysis:
    """
    Detect political language markers in text.

    Args:
        text: Text to analyze

    Returns:
        PoliticalLanguageAnalysis with detected markers and scores
    """
    text_lower = text.lower()

    # Find conservative markers
    conservative_markers = []
    for category, words in CONSERVATIVE_LANGUAGE.items():
        for word in words:
            if word in text_lower:
                conservative_markers.append(word)

    # Find liberal markers
    liberal_markers = []
    for category, words in LIBERAL_LANGUAGE.items():
        for word in words:
            if word in text_lower:
                liberal_markers.append(word)

    # Find polarizing topics
    found_topics = []
    for topic in POLARIZING_TOPICS:
        if topic in text_lower:
            found_topics.append(topic)

    # Calculate scores
    total_markers = len(conservative_markers) + len(liberal_markers)

    if total_markers > 0:
        conservative_score = len(conservative_markers) / total_markers
        liberal_score = len(liberal_markers) / total_markers
        # Polarization: how one-sided is the language (0 = balanced, 1 = very one-sided)
        polarization_score = abs(conservative_score - liberal_score)
    else:
        conservative_score = 0.0
        liberal_score = 0.0
        polarization_score = 0.0

    return PoliticalLanguageAnalysis(
        conservative_markers=conservative_markers,
        liberal_markers=liberal_markers,
        polarizing_topics=found_topics,
        conservative_score=conservative_score,
        liberal_score=liberal_score,
        polarization_score=polarization_score
    )


def analyze_conversation_polarization(conversation: ConversationLogger) -> Dict[str, Any]:
    """
    Analyze polarization dynamics in a conversation.

    Tracks how political language evolves over the course of the conversation.

    Args:
        conversation: ConversationLogger to analyze

    Returns:
        Dictionary with polarization metrics over time
    """
    turn_analyses = []

    for turn in conversation.turns:
        analysis = detect_political_language(turn.message)

        turn_analyses.append({
            "turn_number": turn.turn_number,
            "agent_name": turn.agent_name,
            "role": turn.role,
            "conservative_score": analysis.conservative_score,
            "liberal_score": analysis.liberal_score,
            "polarization_score": analysis.polarization_score,
            "polarizing_topics": analysis.polarizing_topics
        })

    # Calculate overall metrics
    user_turns = [t for t in turn_analyses if t["role"] == "user"]
    assistant_turns = [t for t in turn_analyses if t["role"] == "assistant"]

    def avg_score(turns, key):
        scores = [t[key] for t in turns if t[key] is not None]
        return sum(scores) / len(scores) if scores else 0.0

    return {
        "conversation_id": conversation.conversation_id,
        "turn_analyses": turn_analyses,
        "overall_metrics": {
            "avg_polarization": avg_score(turn_analyses, "polarization_score"),
            "user_avg_conservative": avg_score(user_turns, "conservative_score"),
            "user_avg_liberal": avg_score(user_turns, "liberal_score"),
            "assistant_avg_conservative": avg_score(assistant_turns, "conservative_score"),
            "assistant_avg_liberal": avg_score(assistant_turns, "liberal_score"),
        }
    }


def calculate_opinion_shift(
    conversation: ConversationLogger,
    initial_turns: int = 3,
    final_turns: int = 3
) -> Dict[str, float]:
    """
    Calculate opinion shift by comparing early vs late conversation.

    Args:
        conversation: ConversationLogger to analyze
        initial_turns: Number of initial turns to analyze
        final_turns: Number of final turns to analyze

    Returns:
        Dictionary with shift metrics
    """
    user_turns = [turn for turn in conversation.turns if turn.role == "user"]

    if len(user_turns) < initial_turns + final_turns:
        return {
            "conservative_shift": 0.0,
            "liberal_shift": 0.0,
            "polarization_shift": 0.0,
            "insufficient_data": True
        }

    # Analyze initial turns
    initial_analyses = [
        detect_political_language(turn.message)
        for turn in user_turns[:initial_turns]
    ]

    initial_conservative = sum(a.conservative_score for a in initial_analyses) / len(initial_analyses)
    initial_liberal = sum(a.liberal_score for a in initial_analyses) / len(initial_analyses)
    initial_polarization = sum(a.polarization_score for a in initial_analyses) / len(initial_analyses)

    # Analyze final turns
    final_analyses = [
        detect_political_language(turn.message)
        for turn in user_turns[-final_turns:]
    ]

    final_conservative = sum(a.conservative_score for a in final_analyses) / len(final_analyses)
    final_liberal = sum(a.liberal_score for a in final_analyses) / len(final_analyses)
    final_polarization = sum(a.polarization_score for a in final_analyses) / len(final_analyses)

    return {
        "conservative_shift": final_conservative - initial_conservative,
        "liberal_shift": final_liberal - initial_liberal,
        "polarization_shift": final_polarization - initial_polarization,
        "initial_conservative": initial_conservative,
        "initial_liberal": initial_liberal,
        "final_conservative": final_conservative,
        "final_liberal": final_liberal,
        "insufficient_data": False
    }


def count_agreement_disagreement(text: str) -> Dict[str, int]:
    """
    Count markers of agreement and disagreement in text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with agreement/disagreement counts
    """
    text_lower = text.lower()

    agreement_markers = [
        "i agree", "you're right", "that's true", "exactly", "absolutely",
        "makes sense", "good point", "i see what you mean"
    ]

    disagreement_markers = [
        "i disagree", "i don't think", "that's not", "actually", "but",
        "however", "on the other hand", "i'm not sure", "i doubt"
    ]

    agreement_count = sum(text_lower.count(marker) for marker in agreement_markers)
    disagreement_count = sum(text_lower.count(marker) for marker in disagreement_markers)

    return {
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "agreement_ratio": (
            agreement_count / (agreement_count + disagreement_count)
            if (agreement_count + disagreement_count) > 0 else 0.5
        )
    }
