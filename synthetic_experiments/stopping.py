"""
Smart stopping conditions for conversations.

This module provides configurable stopping conditions that go beyond
simple turn limits, enabling conversations to end based on semantic
and behavioral criteria.

Example:
    >>> from synthetic_experiments.stopping import (
    ...     StoppingConditionManager,
    ...     TopicDriftCondition,
    ...     SentimentExtremeCondition,
    ...     RepetitionCondition
    ... )
    >>> 
    >>> # Create manager with multiple conditions
    >>> stopper = StoppingConditionManager([
    ...     TopicDriftCondition(threshold=0.7),
    ...     SentimentExtremeCondition(threshold=0.9),
    ...     RepetitionCondition(window=5, threshold=0.8)
    ... ])
    >>> 
    >>> # Check during conversation
    >>> if stopper.should_stop(messages):
    ...     print(f"Stopping: {stopper.get_reason()}")
"""

from typing import List, Dict, Any, Optional, Callable, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class StoppingCondition(ABC):
    """Base class for stopping conditions."""
    
    @abstractmethod
    def check(self, messages: List[Dict[str, str]]) -> bool:
        """
        Check if stopping condition is met.
        
        Args:
            messages: List of messages with 'role' and 'content' keys
            
        Returns:
            True if conversation should stop
        """
        pass
    
    @abstractmethod
    def get_reason(self) -> str:
        """Get human-readable reason for stopping."""
        pass


class TopicDriftCondition(StoppingCondition):
    """
    Stop when conversation drifts too far from original topic.
    
    Uses keyword overlap to detect topic drift.
    """
    
    def __init__(self, threshold: float = 0.3, window: int = 5):
        """
        Args:
            threshold: Keyword overlap threshold (0-1, lower = more strict)
            window: Number of recent messages to compare
        """
        self.threshold = threshold
        self.window = window
        self._triggered = False
        self._drift_score = 0.0
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        # Simple keyword extraction - remove common words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
            'now', 'here', 'there', 'then', 'if', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under'
        }
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return {w for w in words if len(w) > 3 and w not in stopwords}
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        if len(messages) < self.window + 2:
            return False
        
        # Get keywords from early messages
        early_messages = messages[:self.window]
        early_keywords = set()
        for msg in early_messages:
            early_keywords.update(self._extract_keywords(msg.get('content', '')))
        
        # Get keywords from recent messages
        recent_messages = messages[-self.window:]
        recent_keywords = set()
        for msg in recent_messages:
            recent_keywords.update(self._extract_keywords(msg.get('content', '')))
        
        if not early_keywords or not recent_keywords:
            return False
        
        # Calculate overlap (Jaccard similarity)
        overlap = len(early_keywords & recent_keywords) / len(early_keywords | recent_keywords)
        self._drift_score = 1 - overlap
        
        if overlap < self.threshold:
            self._triggered = True
            logger.info(f"Topic drift detected: overlap={overlap:.2f}, threshold={self.threshold}")
            return True
        
        return False
    
    def get_reason(self) -> str:
        return f"Topic drift detected (drift score: {self._drift_score:.2f})"


class SentimentExtremeCondition(StoppingCondition):
    """
    Stop when sentiment becomes too extreme (very positive or negative).
    
    Uses simple keyword-based sentiment detection by default.
    """
    
    def __init__(
        self,
        threshold: float = 0.8,
        window: int = 3,
        sentiment_analyzer: Optional[Callable[[str], float]] = None
    ):
        """
        Args:
            threshold: Absolute sentiment threshold (0-1)
            window: Number of recent messages to analyze
            sentiment_analyzer: Custom function that returns sentiment (-1 to 1)
        """
        self.threshold = threshold
        self.window = window
        self._analyze = sentiment_analyzer or self._default_analyze
        self._triggered = False
        self._extreme_sentiment = 0.0
    
    def _default_analyze(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        positive_words = {
            'great', 'good', 'excellent', 'wonderful', 'amazing', 'fantastic',
            'love', 'happy', 'glad', 'pleased', 'delighted', 'agree', 'yes',
            'right', 'correct', 'absolutely', 'perfect', 'best', 'brilliant'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad',
            'upset', 'disagree', 'wrong', 'no', 'never', 'worst', 'stupid',
            'idiot', 'fool', 'evil', 'dangerous', 'ridiculous', 'absurd'
        }
        
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        if len(messages) < self.window:
            return False
        
        recent = messages[-self.window:]
        sentiments = [
            self._analyze(msg.get('content', ''))
            for msg in recent
        ]
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        self._extreme_sentiment = avg_sentiment
        
        if abs(avg_sentiment) >= self.threshold:
            self._triggered = True
            direction = "positive" if avg_sentiment > 0 else "negative"
            logger.info(f"Extreme {direction} sentiment detected: {avg_sentiment:.2f}")
            return True
        
        return False
    
    def get_reason(self) -> str:
        direction = "positive" if self._extreme_sentiment > 0 else "negative"
        return f"Extreme {direction} sentiment ({self._extreme_sentiment:.2f})"


class RepetitionCondition(StoppingCondition):
    """
    Stop when participants start repeating themselves.
    
    Detects repeated phrases or similar messages.
    """
    
    def __init__(self, window: int = 6, threshold: float = 0.7):
        """
        Args:
            window: Number of recent messages to check
            threshold: Similarity threshold for repetition detection
        """
        self.window = window
        self.threshold = threshold
        self._triggered = False
        self._repetition_score = 0.0
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-level similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        if len(messages) < self.window:
            return False
        
        recent = messages[-self.window:]
        normalized = [self._normalize(msg.get('content', '')) for msg in recent]
        
        # Check for similar messages
        max_similarity = 0.0
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                sim = self._similarity(normalized[i], normalized[j])
                max_similarity = max(max_similarity, sim)
        
        self._repetition_score = max_similarity
        
        if max_similarity >= self.threshold:
            self._triggered = True
            logger.info(f"Repetition detected: similarity={max_similarity:.2f}")
            return True
        
        return False
    
    def get_reason(self) -> str:
        return f"Repetition detected (similarity: {self._repetition_score:.2f})"


class ConsensusCondition(StoppingCondition):
    """
    Stop when participants reach consensus/agreement.
    """
    
    def __init__(self, window: int = 4, agreement_threshold: int = 3):
        """
        Args:
            window: Number of recent messages to check
            agreement_threshold: Number of agreement signals needed
        """
        self.window = window
        self.agreement_threshold = agreement_threshold
        self._triggered = False
        self._agreement_count = 0
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        if len(messages) < self.window:
            return False
        
        agreement_phrases = [
            r'\bi agree\b', r'\bthat\'s right\b', r'\byou\'re right\b',
            r'\bexactly\b', r'\babsolutely\b', r'\bprecisely\b',
            r'\byou make a good point\b', r'\bi see your point\b',
            r'\bfair point\b', r'\bgood point\b', r'\bwe\'re on the same page\b',
            r'\bi concur\b', r'\bwell said\b', r'\bi share\b'
        ]
        
        recent = messages[-self.window:]
        self._agreement_count = 0
        
        for msg in recent:
            content = msg.get('content', '').lower()
            for phrase in agreement_phrases:
                if re.search(phrase, content):
                    self._agreement_count += 1
                    break  # Count once per message
        
        if self._agreement_count >= self.agreement_threshold:
            self._triggered = True
            logger.info(f"Consensus reached: {self._agreement_count} agreement signals")
            return True
        
        return False
    
    def get_reason(self) -> str:
        return f"Consensus reached ({self._agreement_count} agreement signals)"


class DeadlockCondition(StoppingCondition):
    """
    Stop when conversation reaches a deadlock (repeated disagreement).
    """
    
    def __init__(self, window: int = 6, disagreement_threshold: int = 4):
        """
        Args:
            window: Number of recent messages to check
            disagreement_threshold: Number of disagreement signals needed
        """
        self.window = window
        self.disagreement_threshold = disagreement_threshold
        self._triggered = False
        self._disagreement_count = 0
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        if len(messages) < self.window:
            return False
        
        disagreement_phrases = [
            r'\bi disagree\b', r'\bthat\'s wrong\b', r'\byou\'re wrong\b',
            r'\bno way\b', r'\babsolutely not\b', r'\bi don\'t think so\b',
            r'\bthat\'s not true\b', r'\bi can\'t agree\b', r'\bwe disagree\b',
            r'\bincorrect\b', r'\bmisguided\b', r'\bfallacy\b'
        ]
        
        recent = messages[-self.window:]
        self._disagreement_count = 0
        
        for msg in recent:
            content = msg.get('content', '').lower()
            for phrase in disagreement_phrases:
                if re.search(phrase, content):
                    self._disagreement_count += 1
                    break
        
        if self._disagreement_count >= self.disagreement_threshold:
            self._triggered = True
            logger.info(f"Deadlock detected: {self._disagreement_count} disagreement signals")
            return True
        
        return False
    
    def get_reason(self) -> str:
        return f"Deadlock reached ({self._disagreement_count} disagreement signals)"


class MessageLengthCondition(StoppingCondition):
    """
    Stop when messages become too short (disengagement) or too long (lectures).
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 2000,
        window: int = 3
    ):
        """
        Args:
            min_length: Minimum average word count
            max_length: Maximum average word count
            window: Number of recent messages to check
        """
        self.min_length = min_length
        self.max_length = max_length
        self.window = window
        self._triggered = False
        self._avg_length = 0
        self._reason = ""
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        if len(messages) < self.window:
            return False
        
        recent = messages[-self.window:]
        lengths = [len(msg.get('content', '').split()) for msg in recent]
        self._avg_length = sum(lengths) / len(lengths)
        
        if self._avg_length < self.min_length:
            self._triggered = True
            self._reason = "disengagement (messages too short)"
            logger.info(f"Disengagement detected: avg length={self._avg_length:.1f}")
            return True
        
        if self._avg_length > self.max_length:
            self._triggered = True
            self._reason = "monologuing (messages too long)"
            logger.info(f"Monologuing detected: avg length={self._avg_length:.1f}")
            return True
        
        return False
    
    def get_reason(self) -> str:
        return f"Message length issue: {self._reason} (avg: {self._avg_length:.1f} words)"


class CustomCondition(StoppingCondition):
    """
    User-defined custom stopping condition.
    """
    
    def __init__(self, check_func: Callable[[List[Dict[str, str]]], bool], reason: str = "Custom condition"):
        """
        Args:
            check_func: Function that returns True when conversation should stop
            reason: Reason string for this condition
        """
        self._check_func = check_func
        self._reason = reason
        self._triggered = False
    
    def check(self, messages: List[Dict[str, str]]) -> bool:
        self._triggered = self._check_func(messages)
        return self._triggered
    
    def get_reason(self) -> str:
        return self._reason


class StoppingConditionManager:
    """
    Manages multiple stopping conditions.
    
    Example:
        >>> manager = StoppingConditionManager([
        ...     TopicDriftCondition(threshold=0.3),
        ...     RepetitionCondition(threshold=0.8),
        ...     ConsensusCondition()
        ... ], mode='any')
        >>> 
        >>> # During conversation loop
        >>> if manager.should_stop(messages):
        ...     print(manager.get_reason())
    """
    
    def __init__(
        self,
        conditions: List[StoppingCondition],
        mode: str = "any"
    ):
        """
        Args:
            conditions: List of StoppingCondition instances
            mode: 'any' (stop if any triggered) or 'all' (stop if all triggered)
        """
        self.conditions = conditions
        self.mode = mode
        self._triggered_conditions: List[StoppingCondition] = []
    
    def should_stop(self, messages: List[Dict[str, str]]) -> bool:
        """
        Check if any/all stopping conditions are met.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            True if conversation should stop
        """
        self._triggered_conditions = []
        
        for condition in self.conditions:
            if condition.check(messages):
                self._triggered_conditions.append(condition)
        
        if self.mode == "any":
            return len(self._triggered_conditions) > 0
        else:  # all
            return len(self._triggered_conditions) == len(self.conditions)
    
    def get_reason(self) -> str:
        """Get combined reason string."""
        if not self._triggered_conditions:
            return "No stopping conditions triggered"
        
        reasons = [c.get_reason() for c in self._triggered_conditions]
        return "; ".join(reasons)
    
    def get_triggered(self) -> List[StoppingCondition]:
        """Get list of triggered conditions."""
        return self._triggered_conditions.copy()
    
    def add_condition(self, condition: StoppingCondition):
        """Add a new stopping condition."""
        self.conditions.append(condition)
    
    def reset(self):
        """Reset all conditions."""
        self._triggered_conditions = []


# Convenience factory functions
def create_default_stopping_conditions() -> StoppingConditionManager:
    """
    Create a manager with sensible default conditions.
    
    Returns:
        StoppingConditionManager with topic drift, repetition, and extremes
    """
    return StoppingConditionManager([
        TopicDriftCondition(threshold=0.3, window=5),
        RepetitionCondition(window=6, threshold=0.75),
        SentimentExtremeCondition(threshold=0.85, window=3),
        DeadlockCondition(window=6, disagreement_threshold=4)
    ])


def create_quality_conditions() -> StoppingConditionManager:
    """
    Create conditions focused on conversation quality.
    
    Returns:
        StoppingConditionManager for quality-based stopping
    """
    return StoppingConditionManager([
        RepetitionCondition(window=4, threshold=0.6),
        MessageLengthCondition(min_length=15, max_length=1500, window=3),
        TopicDriftCondition(threshold=0.4, window=4)
    ])


def create_research_conditions() -> StoppingConditionManager:
    """
    Create conditions for research experiments.
    
    Returns:
        StoppingConditionManager for research use
    """
    return StoppingConditionManager([
        ConsensusCondition(window=4, agreement_threshold=3),
        DeadlockCondition(window=6, disagreement_threshold=5),
        TopicDriftCondition(threshold=0.25, window=6)
    ])
