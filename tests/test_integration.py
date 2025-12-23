"""
Integration tests for the synthetic_experiments framework.

These tests verify that the different components work together correctly.
"""

import pytest
import tempfile
from pathlib import Path

from synthetic_experiments.agents.persona import Persona, PersonaFactory
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.experiments.experiment import Experiment, ExperimentConfig
from synthetic_experiments.data.logger import ConversationLogger
from synthetic_experiments.data.storage import ExperimentStorage
from synthetic_experiments.analysis.metrics import calculate_basic_metrics
from synthetic_experiments.analysis.political import (
    detect_political_language,
    analyze_conversation_polarization
)


class TestAgentPersonaIntegration:
    """Tests for Agent and Persona integration."""

    def test_agent_with_factory_persona(self, mock_provider):
        """Test agent with factory-created persona."""
        persona = PersonaFactory.create_political_user("liberal", "moderate")
        agent = ConversationAgent(
            provider=mock_provider,
            persona=persona,
            role="user"
        )
        
        response = agent.respond("What are your views on climate policy?")
        
        assert response is not None
        assert agent.persona.political_orientation == "liberal"

    def test_agent_persona_in_system_prompt(self, mock_provider):
        """Test that persona details appear in system prompt."""
        persona = Persona(
            name="Detailed User",
            background="Environmental scientist",
            political_orientation="liberal",
            goals="Discuss climate policies"
        )
        
        agent = ConversationAgent(
            provider=mock_provider,
            persona=persona,
            role="user"
        )
        
        messages = agent._prepare_messages_for_generation()
        system_prompt = messages[0].content
        
        assert "Detailed User" in system_prompt
        assert "Environmental scientist" in system_prompt


class TestExperimentEndToEnd:
    """End-to-end experiment tests."""

    def test_complete_experiment_flow(self, mock_provider_with_responses, temp_dir):
        """Test complete experiment from setup to analysis."""
        # Create agents with different personas
        user_responses = [
            "What do you think about renewable energy?",
            "I believe climate change is a serious issue.",
            "We need stronger environmental regulations."
        ]
        assistant_responses = [
            "That's a complex topic with multiple perspectives.",
            "There are indeed many factors to consider.",
            "Balancing economic and environmental concerns is important."
        ]
        
        user_provider = mock_provider_with_responses(user_responses)
        assistant_provider = mock_provider_with_responses(assistant_responses)
        
        user_persona = PersonaFactory.create_political_user("liberal", "moderate")
        assistant_persona = PersonaFactory.create_neutral_advisor()
        
        user_agent = ConversationAgent(
            provider=user_provider,
            persona=user_persona,
            role="user"
        )
        assistant_agent = ConversationAgent(
            provider=assistant_provider,
            persona=assistant_persona,
            role="assistant"
        )
        
        # Run experiment
        experiment = Experiment(
            name="integration_test",
            agents=[user_agent, assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(
            max_turns=3,
            initial_topic="climate policy",
            metadata={"condition": "liberal_user_neutral_advisor"}
        )
        
        # Verify conversation was logged
        assert len(result.turns) > 0
        
        # Verify conversation was saved
        saved_convs = experiment.storage.list_conversations()
        assert len(saved_convs) == 1
        
        # Calculate metrics
        metrics = calculate_basic_metrics(result)
        assert metrics.total_turns > 0
        
        # Analyze polarization
        polarization = analyze_conversation_polarization(result)
        assert "turn_analyses" in polarization

    def test_multiple_experiments_with_storage(self, mock_provider, temp_dir):
        """Test running multiple experiments and storing results."""
        persona = Persona(name="Test User")
        assistant_persona = Persona(name="Test Assistant")
        
        user_agent = ConversationAgent(
            provider=mock_provider,
            persona=persona,
            role="user"
        )
        assistant_agent = ConversationAgent(
            provider=mock_provider,
            persona=assistant_persona,
            role="assistant"
        )
        
        experiment = Experiment(
            name="multi_run_integration",
            agents=[user_agent, assistant_agent],
            output_dir=str(temp_dir)
        )
        
        # Run multiple times
        for i in range(3):
            result = experiment.run(
                max_turns=2,
                initial_topic=f"topic_{i}",
                conversation_id=f"conv_{i}"
            )
            assert result is not None
        
        # Export summary
        csv_path = experiment.storage.export_summary_csv()
        assert csv_path.exists()


class TestStorageAnalysisIntegration:
    """Tests for Storage and Analysis integration."""

    def test_save_load_analyze(self, sample_conversation_logger, temp_dir):
        """Test saving, loading, and analyzing conversations."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "integration"))
        sample_conversation_logger.finalize()
        
        # Save
        storage.save_conversation(sample_conversation_logger)
        
        # Load
        loaded = storage.load_conversation(sample_conversation_logger.conversation_id)
        
        # Analyze
        metrics = calculate_basic_metrics(loaded)
        
        assert metrics.total_turns == 4
        assert metrics.conversation_id == sample_conversation_logger.conversation_id

    def test_batch_analysis(self, temp_dir):
        """Test analyzing multiple conversations."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "batch"))
        
        # Create and save multiple conversations
        for i in range(5):
            logger = ConversationLogger(
                experiment_name="batch_test",
                conversation_id=f"batch_{i}"
            )
            
            for j in range(3):
                logger.log_turn(
                    agent_name="User" if j % 2 == 0 else "Assistant",
                    role="user" if j % 2 == 0 else "assistant",
                    message=f"Message {j} in conversation {i}",
                    tokens_used=10
                )
            
            logger.finalize()
            storage.save_conversation(logger)
        
        # Analyze all
        all_metrics = []
        for conv_id in storage.list_conversations():
            conv = storage.load_conversation(conv_id)
            metrics = calculate_basic_metrics(conv)
            all_metrics.append(metrics)
        
        assert len(all_metrics) == 5
        for m in all_metrics:
            assert m.total_turns == 3


class TestPoliticalAnalysisIntegration:
    """Tests for political analysis with real conversations."""

    def test_political_conversation_analysis(self, mock_provider_with_responses, temp_dir):
        """Test analyzing a politically-charged conversation."""
        liberal_responses = [
            "Climate change requires immediate action. We need renewable energy.",
            "Universal healthcare is a human right. Social justice matters.",
            "We must address inequality through progressive policies."
        ]
        
        neutral_responses = [
            "There are various perspectives on this issue.",
            "Different people have different views based on their values.",
            "It's important to consider multiple viewpoints."
        ]
        
        liberal_provider = mock_provider_with_responses(liberal_responses)
        neutral_provider = mock_provider_with_responses(neutral_responses)
        
        liberal_persona = PersonaFactory.create_political_user("liberal", "high")
        neutral_persona = PersonaFactory.create_neutral_advisor()
        
        liberal_agent = ConversationAgent(
            provider=liberal_provider,
            persona=liberal_persona,
            role="user"
        )
        neutral_agent = ConversationAgent(
            provider=neutral_provider,
            persona=neutral_persona,
            role="assistant"
        )
        
        experiment = Experiment(
            name="political_test",
            agents=[liberal_agent, neutral_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(
            max_turns=3,
            initial_topic="political issues"
        )
        
        # Analyze polarization
        polarization = analyze_conversation_polarization(result)
        
        # User should show more liberal markers
        user_turns = [t for t in polarization["turn_analyses"] if t["role"] == "user"]
        
        assert len(user_turns) > 0


class TestDataPipelineIntegration:
    """Tests for the complete data pipeline."""

    def test_full_data_pipeline(self, mock_provider, temp_dir):
        """Test from conversation to exported data."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        # Setup
        persona = Persona(name="Pipeline User")
        assistant_persona = Persona(name="Pipeline Assistant")
        
        user_agent = ConversationAgent(
            provider=mock_provider,
            persona=persona,
            role="user"
        )
        assistant_agent = ConversationAgent(
            provider=mock_provider,
            persona=assistant_persona,
            role="assistant"
        )
        
        # Run experiments with different conditions
        conditions = ["control", "treatment_a", "treatment_b"]
        
        for condition in conditions:
            experiment = Experiment(
                name=f"pipeline_{condition}",
                agents=[user_agent, assistant_agent],
                output_dir=str(temp_dir / condition)
            )
            
            for i in range(2):
                # Use unique conversation IDs to avoid file conflicts
                result = experiment.run(
                    max_turns=2,
                    initial_topic="test",
                    conversation_id=f"{condition}_trial_{i}",
                    metadata={"condition": condition, "trial": i}
                )
            
            # Export CSV for this condition
            csv_path = experiment.storage.export_summary_csv()
            
            # Verify CSV
            df = pd.read_csv(csv_path)
            assert len(df) == 2
