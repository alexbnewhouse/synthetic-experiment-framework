"""Tests for the polarization survey module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from synthetic_experiments.analysis.survey import (
    SurveyQuestion,
    SurveyResponse,
    SurveyResults,
    PolarizationSurvey,
    SurveyAdministrator,
    PolarizationDelta,
    PolarizationType,
    calculate_polarization_delta,
)
from synthetic_experiments.providers.base import LLMProvider, Message, GenerationConfig, GenerationResult
from synthetic_experiments.agents.persona import Persona


class TestSurveyQuestion:
    """Tests for SurveyQuestion dataclass."""
    
    def test_create_basic_question(self):
        """Test creating a basic survey question."""
        question = SurveyQuestion(
            id="q1",
            text="How do you feel about X?",
            scale_min=1,
            scale_max=7,
            polarization_type=PolarizationType.AFFECTIVE
        )
        assert question.id == "q1"
        assert question.scale_min == 1
        assert question.scale_max == 7
        assert question.reverse_coded is False
    
    def test_create_reverse_coded_question(self):
        """Test creating a reverse-coded question."""
        question = SurveyQuestion(
            id="q2",
            text="How much do you trust them?",
            scale_min=1,
            scale_max=10,
            polarization_type=PolarizationType.AFFECTIVE,
            reverse_coded=True
        )
        assert question.reverse_coded is True
    
    def test_question_with_labels(self):
        """Test question with scale labels."""
        question = SurveyQuestion(
            id="q3",
            text="Rate your agreement",
            scale_min=1,
            scale_max=5,
            polarization_type=PolarizationType.IDEOLOGICAL,
            scale_labels={"1": "Strongly Disagree", "5": "Strongly Agree"}
        )
        assert "Strongly Disagree" in question.scale_labels.values()
        assert "Strongly Agree" in question.scale_labels.values()
    
    def test_get_prompt(self):
        """Test generating question prompt."""
        question = SurveyQuestion(
            id="q1",
            text="How do you feel about X?",
            scale_min=1,
            scale_max=7,
            polarization_type=PolarizationType.AFFECTIVE
        )
        prompt = question.get_prompt()
        assert "How do you feel about X?" in prompt
        assert "1 to 7" in prompt


class TestSurveyResponse:
    """Tests for SurveyResponse dataclass."""
    
    def test_create_response(self):
        """Test creating a survey response."""
        response = SurveyResponse(
            question_id="q1",
            raw_response="5",
            numeric_value=5
        )
        assert response.question_id == "q1"
        assert response.numeric_value == 5
        assert response.valid is True
    
    def test_invalid_response(self):
        """Test invalid response."""
        response = SurveyResponse(
            question_id="q1",
            raw_response="invalid",
            numeric_value=None,
            valid=False
        )
        assert response.valid is False
        assert response.numeric_value is None


class TestSurveyResults:
    """Tests for SurveyResults dataclass."""
    
    def test_create_results(self):
        """Test creating survey results."""
        responses = [
            SurveyResponse("q1", "5", 5),
            SurveyResponse("q2", "3", 3),
        ]
        results = SurveyResults(
            survey_type="pre",
            responses=responses,
            affective_score=0.5,
            ideological_score=0.3,
            overall_score=0.4,
            valid_response_rate=1.0,
            seed=42,
            metadata={}
        )
        assert results.affective_score == 0.5
        assert results.ideological_score == 0.3
        assert results.overall_score == 0.4
        assert results.seed == 42
    
    def test_results_to_dict(self):
        """Test converting results to dictionary."""
        results = SurveyResults(
            survey_type="test",
            responses=[SurveyResponse("q1", "5", 5)],
            affective_score=0.5,
            ideological_score=0.3,
            overall_score=0.4,
            valid_response_rate=1.0,
            metadata={}
        )
        result_dict = results.to_dict()
        assert "survey_type" in result_dict
        assert "affective_score" in result_dict
        assert result_dict["affective_score"] == 0.5


class TestPolarizationSurvey:
    """Tests for PolarizationSurvey class."""
    
    def test_default_questions(self):
        """Test that default questions are loaded."""
        survey = PolarizationSurvey()
        assert len(survey.get_affective_questions()) == 6
        assert len(survey.get_ideological_questions()) == 6
    
    def test_all_questions(self):
        """Test getting all questions."""
        survey = PolarizationSurvey()
        all_questions = survey.get_questions()
        assert len(all_questions) == 12
    
    def test_get_question_prompt(self):
        """Test generating question prompt."""
        survey = PolarizationSurvey()
        question = survey.get_affective_questions()[0]
        prompt = question.get_prompt()
        assert "respond" in prompt.lower() or "scale" in prompt.lower()
    
    def test_parse_valid_response(self):
        """Test parsing valid numeric response."""
        question = SurveyQuestion(
            id="test",
            text="Test question",
            scale_min=1,
            scale_max=7,
            polarization_type=PolarizationType.AFFECTIVE
        )
        # Create administrator to test parsing
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        value = admin._parse_numeric_response("5", 1, 7)
        assert value == 5
    
    def test_parse_response_with_text(self):
        """Test parsing response with surrounding text."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        value = admin._parse_numeric_response("My answer is 6 because...", 1, 7)
        assert value == 6
    
    def test_parse_response_out_of_range(self):
        """Test parsing response outside valid range."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        value = admin._parse_numeric_response("10", 1, 7)
        assert value is None  # Out of range


class TestPolarizationDelta:
    """Tests for PolarizationDelta dataclass."""
    
    def test_create_delta(self):
        """Test creating polarization delta."""
        pre = SurveyResults(
            survey_type="pre",
            responses=[],
            affective_score=0.4,
            ideological_score=0.5,
            overall_score=0.45,
            valid_response_rate=1.0
        )
        post = SurveyResults(
            survey_type="post",
            responses=[],
            affective_score=0.6,
            ideological_score=0.45,
            overall_score=0.525,
            valid_response_rate=1.0
        )
        delta = PolarizationDelta(
            affective_delta=0.2,
            ideological_delta=-0.05,
            overall_delta=0.075,
            pre_results=pre,
            post_results=post
        )
        assert delta.affective_delta == 0.2
        assert delta.ideological_delta == -0.05
    
    def test_delta_to_dict(self):
        """Test converting delta to dictionary."""
        pre_results = SurveyResults(
            survey_type="pre",
            responses=[],
            affective_score=0.4,
            ideological_score=0.5,
            overall_score=0.45,
            valid_response_rate=1.0,
            metadata={}
        )
        post_results = SurveyResults(
            survey_type="post",
            responses=[],
            affective_score=0.55,
            ideological_score=0.45,
            overall_score=0.5,
            valid_response_rate=1.0,
            metadata={}
        )
        delta = PolarizationDelta(
            affective_delta=0.15,
            ideological_delta=-0.05,
            overall_delta=0.05,
            pre_results=pre_results,
            post_results=post_results
        )
        result = delta.to_dict()
        assert "affective_delta" in result
        assert "pre_survey" in result
        assert "post_survey" in result


class TestCalculatePolarizationDelta:
    """Tests for calculate_polarization_delta function."""
    
    def test_calculate_delta(self):
        """Test calculating delta between pre and post surveys."""
        pre = SurveyResults(
            survey_type="pre",
            responses=[],
            affective_score=0.4,
            ideological_score=0.5,
            overall_score=0.45,
            valid_response_rate=1.0,
            metadata={}
        )
        post = SurveyResults(
            survey_type="post",
            responses=[],
            affective_score=0.6,
            ideological_score=0.55,
            overall_score=0.575,
            valid_response_rate=1.0,
            metadata={}
        )
        delta = calculate_polarization_delta(pre, post)
        assert abs(delta.affective_delta - 0.2) < 0.001
        assert abs(delta.ideological_delta - 0.05) < 0.001
    
    def test_negative_delta(self):
        """Test negative change (depolarization)."""
        pre = SurveyResults(
            survey_type="pre",
            responses=[],
            affective_score=0.7,
            ideological_score=0.6,
            overall_score=0.65,
            valid_response_rate=1.0,
            metadata={}
        )
        post = SurveyResults(
            survey_type="post",
            responses=[],
            affective_score=0.5,
            ideological_score=0.5,
            overall_score=0.5,
            valid_response_rate=1.0,
            metadata={}
        )
        delta = calculate_polarization_delta(pre, post)
        assert abs(delta.affective_delta - (-0.2)) < 0.001
        assert abs(delta.ideological_delta - (-0.1)) < 0.001


class TestSurveyAdministrator:
    """Tests for SurveyAdministrator class."""
    
    def test_init_with_provider_config(self):
        """Test initializing with provider configuration."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test Advisor"),
            seed=42
        )
        assert admin.provider_class == OllamaProvider
        assert admin.seed == 42
    
    def test_init_default_seed(self):
        """Test initializing with default seed."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test Advisor")
        )
        assert admin.seed == 42  # Default seed
    
    @pytest.mark.skipif(True, reason="Requires running Ollama server")
    def test_fresh_provider_creation(self):
        """Test that fresh provider instances are created."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test Advisor"),
            seed=42
        )
        # Create two fresh providers - they should be different instances
        provider1 = admin._create_fresh_provider()
        provider2 = admin._create_fresh_provider()
        assert provider1 is not provider2
    
    def test_seed_consistency(self):
        """Test that same seed produces same configuration."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin1 = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test Advisor"),
            seed=42
        )
        admin2 = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test Advisor"),
            seed=42
        )
        assert admin1.seed == admin2.seed


class TestSurveyQuestionValidation:
    """Tests for survey question validation."""
    
    def test_affective_questions_have_correct_type(self):
        """Test that affective questions are typed correctly."""
        survey = PolarizationSurvey()
        for q in survey.get_affective_questions():
            assert q.polarization_type == PolarizationType.AFFECTIVE
    
    def test_ideological_questions_have_correct_type(self):
        """Test that ideological questions are typed correctly."""
        survey = PolarizationSurvey()
        for q in survey.get_ideological_questions():
            assert q.polarization_type == PolarizationType.IDEOLOGICAL
    
    def test_all_questions_have_valid_ranges(self):
        """Test that all questions have valid min/max ranges."""
        survey = PolarizationSurvey()
        for q in survey.get_questions():
            assert q.scale_min < q.scale_max
            assert q.scale_min >= 0
    
    def test_question_ids_are_unique(self):
        """Test that all question IDs are unique."""
        survey = PolarizationSurvey()
        ids = [q.id for q in survey.get_questions()]
        assert len(ids) == len(set(ids))


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_parse_empty_response(self):
        """Test parsing empty response."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        value = admin._parse_numeric_response("", 1, 7)
        assert value is None
    
    def test_parse_non_numeric_response(self):
        """Test parsing completely non-numeric response."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        value = admin._parse_numeric_response("I don't know", 1, 7)
        assert value is None
    
    def test_survey_results_with_no_valid_responses(self):
        """Test survey results when no responses are valid."""
        results = SurveyResults(
            survey_type="test",
            responses=[
                SurveyResponse("q1", "invalid", None, valid=False),
            ],
            affective_score=0.0,
            ideological_score=0.0,
            overall_score=0.0,
            valid_response_rate=0.0,
            metadata={}
        )
        assert results.overall_score == 0.0
        assert results.valid_response_rate == 0.0
    
    def test_survey_initialization_options(self):
        """Test survey can be initialized with different options."""
        # Affective only
        aff_survey = PolarizationSurvey(include_affective=True, include_ideological=False)
        assert len(aff_survey.get_questions()) == 6
        assert len(aff_survey.get_ideological_questions()) == 0
        
        # Ideological only
        ideo_survey = PolarizationSurvey(include_affective=False, include_ideological=True)
        assert len(ideo_survey.get_questions()) == 6
        assert len(ideo_survey.get_affective_questions()) == 0
    
    def test_custom_questions(self):
        """Test adding custom questions to survey."""
        custom = SurveyQuestion(
            id="custom_q1",
            text="Custom question",
            scale_min=1,
            scale_max=10,
            polarization_type=PolarizationType.IDEOLOGICAL
        )
        survey = PolarizationSurvey(custom_questions=[custom])
        assert len(survey.get_questions()) == 13  # 12 default + 1 custom


class TestContextIsolation:
    """Tests for context isolation behavior."""
    
    @pytest.mark.skipif(True, reason="Requires running Ollama server")
    def test_fresh_provider_per_survey(self):
        """Test that each survey gets a fresh provider."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        
        # Verify we can create multiple fresh providers
        providers = [admin._create_fresh_provider() for _ in range(3)]
        # Each should be a distinct instance
        assert len(set(id(p) for p in providers)) == 3


class TestPostSurveyWithContext:
    """Tests for post-survey administration with conversation context."""
    
    def test_post_survey_requires_agent(self):
        """Test that post-survey raises error without agent."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        
        with pytest.raises(ValueError, match="requires an agent"):
            admin.administer_survey(survey_type="post", agent=None)
    
    def test_invalid_survey_type_raises(self):
        """Test that invalid survey type raises error."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        
        with pytest.raises(ValueError, match="must be 'pre' or 'post'"):
            admin.administer_survey(survey_type="invalid")
    
    def test_administer_single_question_with_history(self):
        """Test that single question can include conversation history."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        from synthetic_experiments.agents.agent import ConversationAgent
        
        # Create mock provider
        mock_provider = Mock(spec=LLMProvider)
        mock_result = Mock()
        mock_result.message.content = "5"
        mock_result.tokens_used = 10
        mock_result.finish_reason = "stop"
        mock_provider.generate.return_value = mock_result
        
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        
        question = SurveyQuestion(
            id="test_q",
            text="Test question",
            scale_min=1,
            scale_max=7,
            polarization_type=PolarizationType.AFFECTIVE
        )
        
        # Create conversation history
        conversation_history = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        
        response = admin._administer_single_question(
            mock_provider,
            question,
            "System prompt",
            conversation_history=conversation_history
        )
        
        # Verify the provider was called with history included
        call_args = mock_provider.generate.call_args
        messages = call_args[0][0]
        
        # Should have: system + 2 history messages + question = 4 messages
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Hi there!"
    
    def test_results_metadata_includes_context_info(self):
        """Test that results metadata indicates whether conversation context was used."""
        pre_results = SurveyResults(
            survey_type="pre",
            responses=[],
            affective_score=0.5,
            ideological_score=0.5,
            overall_score=0.5,
            valid_response_rate=1.0,
            metadata={
                "has_conversation_context": False,
                "conversation_turns_in_context": 0
            }
        )
        
        post_results = SurveyResults(
            survey_type="post",
            responses=[],
            affective_score=0.6,
            ideological_score=0.6,
            overall_score=0.6,
            valid_response_rate=1.0,
            metadata={
                "has_conversation_context": True,
                "conversation_turns_in_context": 20
            }
        )
        
        assert pre_results.metadata["has_conversation_context"] is False
        assert post_results.metadata["has_conversation_context"] is True
        assert post_results.metadata["conversation_turns_in_context"] == 20
    
    def test_build_survey_system_prompt(self):
        """Test building survey system prompt."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test Advisor", background="A helpful advisor"),
            seed=42
        )
        
        prompt = admin._build_survey_system_prompt()
        assert "Test Advisor" in prompt
        assert "survey" in prompt.lower()
        assert "number" in prompt.lower()
    
    def test_build_survey_system_prompt_with_context(self):
        """Test building survey system prompt with additional context."""
        from synthetic_experiments.providers.ollama import OllamaProvider
        admin = SurveyAdministrator(
            provider_class=OllamaProvider,
            provider_kwargs={"model_name": "llama3.2"},
            persona=Persona(name="Test"),
            seed=42
        )
        
        prompt = admin._build_survey_system_prompt(additional_context="Extra info")
        assert "Extra info" in prompt