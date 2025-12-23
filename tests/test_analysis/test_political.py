"""
Tests for political language analysis tools.
"""

import pytest

from synthetic_experiments.analysis.political import (
    detect_political_language,
    analyze_conversation_polarization,
    calculate_opinion_shift,
    PoliticalLanguageAnalysis,
    CONSERVATIVE_LANGUAGE,
    LIBERAL_LANGUAGE,
    POLARIZING_TOPICS
)
from synthetic_experiments.data.logger import ConversationLogger


class TestPoliticalLanguageAnalysis:
    """Tests for PoliticalLanguageAnalysis dataclass."""

    def test_analysis_creation(self):
        """Test creating PoliticalLanguageAnalysis."""
        analysis = PoliticalLanguageAnalysis(
            conservative_markers=["free market"],
            liberal_markers=["climate change"],
            polarizing_topics=["immigration"],
            conservative_score=0.5,
            liberal_score=0.5,
            polarization_score=0.0
        )
        
        assert analysis.conservative_score == 0.5
        assert analysis.liberal_score == 0.5
        assert analysis.polarization_score == 0.0

    def test_analysis_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = PoliticalLanguageAnalysis(
            conservative_markers=["limited government"],
            liberal_markers=[],
            polarizing_topics=[],
            conservative_score=1.0,
            liberal_score=0.0,
            polarization_score=1.0
        )
        
        result = analysis.to_dict()
        
        assert result["conservative_score"] == 1.0
        assert result["liberal_score"] == 0.0
        assert "conservative_markers" in result


class TestDetectPoliticalLanguage:
    """Tests for detect_political_language function."""

    def test_detect_liberal_language(self, political_text_liberal):
        """Test detecting liberal language markers."""
        result = detect_political_language(political_text_liberal)
        
        assert isinstance(result, PoliticalLanguageAnalysis)
        assert len(result.liberal_markers) > 0
        assert result.liberal_score > result.conservative_score

    def test_detect_conservative_language(self, political_text_conservative):
        """Test detecting conservative language markers."""
        result = detect_political_language(political_text_conservative)
        
        assert len(result.conservative_markers) > 0
        assert result.conservative_score > result.liberal_score

    def test_detect_neutral_language(self, political_text_neutral):
        """Test detecting neutral language."""
        result = detect_political_language(political_text_neutral)
        
        # Neutral text should have low or balanced scores
        assert result.polarization_score <= 1.0

    def test_detect_empty_text(self):
        """Test with empty text."""
        result = detect_political_language("")
        
        assert result.conservative_score == 0.0
        assert result.liberal_score == 0.0
        assert result.polarization_score == 0.0

    def test_detect_polarizing_topics(self):
        """Test detecting polarizing topics."""
        text = "We need to discuss immigration and gun control policies."
        result = detect_political_language(text)
        
        assert len(result.polarizing_topics) >= 1
        assert "immigration" in result.polarizing_topics or "gun control" in result.polarizing_topics

    def test_case_insensitivity(self):
        """Test that detection is case insensitive."""
        text_lower = "we need climate change action"
        text_upper = "WE NEED CLIMATE CHANGE ACTION"
        
        result_lower = detect_political_language(text_lower)
        result_upper = detect_political_language(text_upper)
        
        assert result_lower.liberal_score == result_upper.liberal_score


class TestAnalyzeConversationPolarization:
    """Tests for analyze_conversation_polarization function."""

    def test_analyze_polarization_basic(self, sample_conversation_logger):
        """Test basic polarization analysis."""
        result = analyze_conversation_polarization(sample_conversation_logger)
        
        assert "conversation_id" in result
        assert "turn_analyses" in result
        assert "overall_metrics" in result

    def test_analyze_polarization_turn_by_turn(self):
        """Test turn-by-turn analysis."""
        logger = ConversationLogger(experiment_name="polarization_test")
        
        logger.log_turn(
            agent_name="User",
            role="user",
            message="Climate change requires government action for renewable energy."
        )
        logger.log_turn(
            agent_name="Assistant",
            role="assistant",
            message="There are various perspectives on environmental policy."
        )
        
        result = analyze_conversation_polarization(logger)
        
        assert len(result["turn_analyses"]) == 2
        assert result["turn_analyses"][0]["role"] == "user"
        assert result["turn_analyses"][1]["role"] == "assistant"

    def test_analyze_polarization_metrics(self):
        """Test overall polarization metrics."""
        logger = ConversationLogger(experiment_name="metrics_test")
        
        # Liberal user messages
        logger.log_turn(
            agent_name="User",
            role="user",
            message="Universal healthcare and social justice are essential."
        )
        # Neutral assistant
        logger.log_turn(
            agent_name="Assistant",
            role="assistant",
            message="That's an interesting perspective."
        )
        
        result = analyze_conversation_polarization(logger)
        metrics = result["overall_metrics"]
        
        assert "avg_polarization" in metrics
        assert "user_avg_liberal" in metrics
        assert "user_avg_conservative" in metrics

    def test_analyze_empty_conversation(self):
        """Test analysis of empty conversation."""
        logger = ConversationLogger(experiment_name="empty")
        
        result = analyze_conversation_polarization(logger)
        
        assert len(result["turn_analyses"]) == 0


class TestCalculateOpinionShift:
    """Tests for calculate_opinion_shift function."""

    def test_opinion_shift_insufficient_data(self):
        """Test with insufficient data."""
        logger = ConversationLogger(experiment_name="short")
        logger.log_turn(agent_name="User", role="user", message="Hello")
        logger.log_turn(agent_name="Assistant", role="assistant", message="Hi")
        
        result = calculate_opinion_shift(logger)
        
        assert result.get("insufficient_data", False) is True

    def test_opinion_shift_basic(self):
        """Test basic opinion shift calculation."""
        logger = ConversationLogger(experiment_name="shift_test")
        
        # Early turns - conservative
        for i in range(3):
            logger.log_turn(
                agent_name="User",
                role="user",
                message="Free market capitalism and limited government are best."
            )
            logger.log_turn(
                agent_name="Assistant",
                role="assistant",
                message="I understand your perspective."
            )
        
        # Later turns - more liberal
        for i in range(3):
            logger.log_turn(
                agent_name="User",
                role="user",
                message="Maybe universal healthcare has some merit."
            )
            logger.log_turn(
                agent_name="Assistant",
                role="assistant",
                message="That's a thoughtful consideration."
            )
        
        result = calculate_opinion_shift(logger, initial_turns=3, final_turns=3)
        
        assert "conservative_shift" in result
        assert "liberal_shift" in result
        assert "polarization_shift" in result


class TestPoliticalDictionaries:
    """Tests for political language dictionaries."""

    def test_conservative_language_structure(self):
        """Test conservative language dictionary structure."""
        assert isinstance(CONSERVATIVE_LANGUAGE, dict)
        assert len(CONSERVATIVE_LANGUAGE) > 0
        
        for category, words in CONSERVATIVE_LANGUAGE.items():
            assert isinstance(category, str)
            assert isinstance(words, list)
            assert len(words) > 0

    def test_liberal_language_structure(self):
        """Test liberal language dictionary structure."""
        assert isinstance(LIBERAL_LANGUAGE, dict)
        assert len(LIBERAL_LANGUAGE) > 0
        
        for category, words in LIBERAL_LANGUAGE.items():
            assert isinstance(category, str)
            assert isinstance(words, list)
            assert len(words) > 0

    def test_polarizing_topics_structure(self):
        """Test polarizing topics list structure."""
        assert isinstance(POLARIZING_TOPICS, list)
        assert len(POLARIZING_TOPICS) > 0
        
        for topic in POLARIZING_TOPICS:
            assert isinstance(topic, str)


class TestPolarizationScoring:
    """Tests for polarization score calculations."""

    def test_fully_conservative_score(self):
        """Test score for fully conservative text."""
        text = "free market limited government personal responsibility"
        result = detect_political_language(text)
        
        # Should be mostly conservative
        if result.conservative_score + result.liberal_score > 0:
            assert result.conservative_score >= result.liberal_score

    def test_fully_liberal_score(self):
        """Test score for fully liberal text."""
        text = "climate change universal healthcare social justice equality"
        result = detect_political_language(text)
        
        # Should be mostly liberal
        if result.conservative_score + result.liberal_score > 0:
            assert result.liberal_score >= result.conservative_score

    def test_balanced_score(self):
        """Test score for balanced text."""
        text = "free market climate change limited government universal healthcare"
        result = detect_political_language(text)
        
        # Should have markers from both sides
        assert len(result.conservative_markers) > 0
        assert len(result.liberal_markers) > 0

    def test_polarization_score_range(self):
        """Test that polarization score is in valid range."""
        texts = [
            "free market capitalism",
            "climate change action",
            "balanced perspective on issues",
            ""
        ]
        
        for text in texts:
            result = detect_political_language(text)
            assert 0.0 <= result.polarization_score <= 1.0
