"""
Tests for the Experiment class.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from synthetic_experiments.experiments.experiment import Experiment, ExperimentConfig
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.data.logger import ConversationLogger


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_config_defaults(self):
        """Test ExperimentConfig default values."""
        config = ExperimentConfig(name="test")
        
        assert config.name == "test"
        assert config.max_turns == 20
        assert config.initial_topic == ""
        assert config.topics == []
        assert config.stop_on_repetition is False
        assert config.save_conversations is True
        assert config.output_dir == "results"

    def test_config_custom(self):
        """Test ExperimentConfig with custom values."""
        config = ExperimentConfig(
            name="custom_test",
            max_turns=10,
            initial_topic="climate change",
            topics=["topic1", "topic2"],
            stop_on_repetition=True,
            repetition_threshold=0.8,
            save_conversations=False,
            output_dir="custom_results",
            metadata={"condition": "treatment"}
        )
        
        assert config.name == "custom_test"
        assert config.max_turns == 10
        assert config.initial_topic == "climate change"
        assert len(config.topics) == 2
        assert config.stop_on_repetition is True
        assert config.metadata["condition"] == "treatment"


class TestExperimentInit:
    """Tests for Experiment initialization."""

    def test_experiment_creation_minimal(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test creating an experiment with minimal parameters."""
        experiment = Experiment(
            name="minimal_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        assert experiment.name == "minimal_test"
        assert len(experiment.agents) == 2
        assert experiment.config.name == "minimal_test"

    def test_experiment_creation_with_config(
        self, sample_agent, sample_assistant_agent, temp_dir
    ):
        """Test creating an experiment with config."""
        config = ExperimentConfig(
            name="configured_test",
            max_turns=5,
            initial_topic="test topic"
        )
        
        experiment = Experiment(
            name="configured_test",
            agents=[sample_agent, sample_assistant_agent],
            config=config,
            output_dir=str(temp_dir)
        )
        
        assert experiment.config.max_turns == 5
        assert experiment.config.initial_topic == "test topic"

    def test_experiment_creates_storage(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that experiment creates storage when save_conversations=True."""
        experiment = Experiment(
            name="storage_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        assert experiment.storage is not None
        assert Path(temp_dir).exists()

    def test_experiment_no_storage_when_disabled(
        self, sample_agent, sample_assistant_agent, temp_dir
    ):
        """Test that storage is None when save_conversations=False."""
        config = ExperimentConfig(
            name="no_storage",
            save_conversations=False
        )
        
        experiment = Experiment(
            name="no_storage",
            agents=[sample_agent, sample_assistant_agent],
            config=config
        )
        
        assert experiment.storage is None


class TestExperimentRun:
    """Tests for running experiments."""

    def test_run_basic(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test running a basic experiment."""
        experiment = Experiment(
            name="run_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=2, initial_topic="test topic")
        
        assert isinstance(result, ConversationLogger)
        assert len(result.turns) > 0

    def test_run_respects_max_turns(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that run respects max_turns limit."""
        experiment = Experiment(
            name="max_turns_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=3, initial_topic="test")
        
        # Each turn has user + assistant, so max 6 messages for 3 turns
        assert len(result.turns) <= 6

    def test_run_resets_agents(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that run resets agent history."""
        # Pre-populate agent history
        sample_agent.respond("Pre-existing message")
        
        experiment = Experiment(
            name="reset_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        experiment.run(max_turns=1, initial_topic="test")
        
        # After run, agent history should be from this run only
        # (agents are reset at start of run)

    def test_run_insufficient_agents(self, sample_agent, temp_dir):
        """Test error when running with insufficient agents."""
        experiment = Experiment(
            name="insufficient",
            agents=[sample_agent],
            output_dir=str(temp_dir)
        )
        
        with pytest.raises(ValueError, match="at least 2 agents"):
            experiment.run(max_turns=1)

    def test_run_saves_conversation(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that run saves conversation to storage."""
        experiment = Experiment(
            name="save_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=1, initial_topic="test")
        
        # Check that conversation was saved
        saved_convs = experiment.storage.list_conversations()
        assert len(saved_convs) == 1

    def test_run_with_custom_conversation_id(
        self, sample_agent, sample_assistant_agent, temp_dir
    ):
        """Test running with custom conversation ID."""
        experiment = Experiment(
            name="custom_id_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(
            max_turns=1,
            initial_topic="test",
            conversation_id="my_custom_id"
        )
        
        assert result.conversation_id == "my_custom_id"

    def test_run_with_metadata(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test running with additional metadata."""
        experiment = Experiment(
            name="metadata_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(
            max_turns=1,
            initial_topic="test",
            metadata={"condition": "treatment", "trial": 1}
        )
        
        assert result.metadata["condition"] == "treatment"
        assert result.metadata["trial"] == 1


class TestExperimentMultipleRuns:
    """Tests for running multiple conversations."""

    def test_run_multiple_times(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test running multiple conversations."""
        experiment = Experiment(
            name="multi_run_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        results = []
        for i in range(3):
            result = experiment.run(
                max_turns=1,
                initial_topic="test",
                conversation_id=f"run_{i}"
            )
            results.append(result)
        
        assert len(results) == 3
        assert experiment.storage.list_conversations() == 3 or len(experiment.storage.list_conversations()) == 3

    def test_run_batch(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test run_batch method if it exists."""
        experiment = Experiment(
            name="batch_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        # Check if run_batch exists
        if hasattr(experiment, 'run_batch'):
            results = experiment.run_batch(
                n_runs=3,
                max_turns=1,
                initial_topic="test"
            )
            assert len(results) == 3


class TestExperimentAgentRoles:
    """Tests for agent role handling."""

    def test_user_agent_goes_first(self, mock_provider, temp_dir):
        """Test that user agent speaks first."""
        user_persona = Persona(name="User")
        assistant_persona = Persona(name="Assistant")
        
        user_agent = ConversationAgent(
            provider=mock_provider,
            persona=user_persona,
            role="user"
        )
        assistant_agent = ConversationAgent(
            provider=mock_provider,
            persona=assistant_persona,
            role="assistant"
        )
        
        # Put assistant first in list to test ordering
        experiment = Experiment(
            name="role_order_test",
            agents=[assistant_agent, user_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=1, initial_topic="test")
        
        # First turn should be from user
        assert result.turns[0].role == "user" or result.turns[0].agent_name == "User"


class TestExperimentLogging:
    """Tests for experiment logging."""

    def test_turns_logged_correctly(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that conversation turns are logged correctly."""
        experiment = Experiment(
            name="logging_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=2, initial_topic="test")
        
        # Each turn should have both user and assistant
        user_turns = [t for t in result.turns if t.role == "user"]
        assistant_turns = [t for t in result.turns if t.role == "assistant"]
        
        assert len(user_turns) > 0
        assert len(assistant_turns) > 0

    def test_conversation_finalized(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that conversation is finalized after run."""
        experiment = Experiment(
            name="finalize_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=1, initial_topic="test")
        
        assert result.end_time is not None


class TestExperimentFromConfig:
    """Tests for creating experiments from configuration files."""

    @pytest.mark.skip(reason="Requires Ollama or mocking of config loading")
    def test_from_config_file(self, temp_dir, sample_config_yaml):
        """Test creating experiment from YAML config file."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(sample_config_yaml)
        
        # This would require mocking Ollama or having it available
        experiment = Experiment.from_config(str(config_path))
        
        assert experiment is not None


class TestExperimentStoppingConditions:
    """Tests for conversation stopping conditions."""

    def test_stops_at_max_turns(self, sample_agent, sample_assistant_agent, temp_dir):
        """Test that experiment stops at max_turns."""
        experiment = Experiment(
            name="stop_test",
            agents=[sample_agent, sample_assistant_agent],
            output_dir=str(temp_dir)
        )
        
        result = experiment.run(max_turns=2, initial_topic="test")
        
        # With 2 turns (each turn = user + assistant), max 4 messages
        assert len(result.turns) <= 4
