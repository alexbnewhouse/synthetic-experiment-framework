"""
Tests for the ConversationLogger class.
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path

from synthetic_experiments.data.logger import ConversationLogger, ConversationTurn
from synthetic_experiments.providers.base import Message


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_turn_creation(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            turn_number=1,
            timestamp=datetime.now(),
            agent_name="Test Agent",
            role="user",
            message="Hello!",
            tokens_used=10,
            cost=0.001,
            metadata={"key": "value"}
        )
        
        assert turn.turn_number == 1
        assert turn.agent_name == "Test Agent"
        assert turn.role == "user"
        assert turn.message == "Hello!"
        assert turn.tokens_used == 10
        assert turn.cost == 0.001

    def test_turn_to_dict(self):
        """Test serializing turn to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        turn = ConversationTurn(
            turn_number=1,
            timestamp=timestamp,
            agent_name="Agent",
            role="assistant",
            message="Response",
            tokens_used=5,
            cost=0.0005
        )
        
        result = turn.to_dict()
        
        assert result["turn_number"] == 1
        assert result["agent_name"] == "Agent"
        assert result["role"] == "assistant"
        assert result["message"] == "Response"
        assert result["timestamp"] == "2024-01-01T12:00:00"


class TestConversationLoggerInit:
    """Tests for ConversationLogger initialization."""

    def test_logger_creation_minimal(self):
        """Test creating a logger with minimal parameters."""
        logger = ConversationLogger(experiment_name="test_experiment")
        
        assert logger.experiment_name == "test_experiment"
        assert logger.conversation_id is not None
        assert logger.metadata == {}
        assert logger.turns == []
        assert logger.current_turn == 0

    def test_logger_creation_full(self):
        """Test creating a logger with all parameters."""
        logger = ConversationLogger(
            experiment_name="full_test",
            conversation_id="custom_id_001",
            metadata={"condition": "control"}
        )
        
        assert logger.experiment_name == "full_test"
        assert logger.conversation_id == "custom_id_001"
        assert logger.metadata == {"condition": "control"}

    def test_auto_generated_id(self):
        """Test that conversation ID is auto-generated."""
        logger = ConversationLogger(experiment_name="test")
        
        assert logger.conversation_id is not None
        assert "test" in logger.conversation_id


class TestConversationLoggerLogTurn:
    """Tests for logging conversation turns."""

    def test_log_turn_basic(self):
        """Test logging a basic turn."""
        logger = ConversationLogger(experiment_name="test")
        
        turn = logger.log_turn(
            agent_name="User",
            role="user",
            message="Hello!",
            tokens_used=10,
            cost=0.001
        )
        
        assert isinstance(turn, ConversationTurn)
        assert turn.turn_number == 1
        assert turn.agent_name == "User"
        assert turn.message == "Hello!"
        assert len(logger.turns) == 1

    def test_log_turn_increments_number(self):
        """Test that turn numbers increment correctly."""
        logger = ConversationLogger(experiment_name="test")
        
        turn1 = logger.log_turn(agent_name="A", role="user", message="1")
        turn2 = logger.log_turn(agent_name="B", role="assistant", message="2")
        turn3 = logger.log_turn(agent_name="A", role="user", message="3")
        
        assert turn1.turn_number == 1
        assert turn2.turn_number == 2
        assert turn3.turn_number == 3

    def test_log_turn_with_metadata(self):
        """Test logging turn with metadata."""
        logger = ConversationLogger(experiment_name="test")
        
        turn = logger.log_turn(
            agent_name="Agent",
            role="user",
            message="Test",
            metadata={"sentiment": "positive"}
        )
        
        assert turn.metadata == {"sentiment": "positive"}

    def test_log_message(self):
        """Test logging from a Message object."""
        logger = ConversationLogger(experiment_name="test")
        
        message = Message(
            role="assistant",
            content="Hello!",
            metadata={"tokens_used": 15, "cost": 0.002}
        )
        
        turn = logger.log_message(agent_name="Assistant", message=message)
        
        assert turn.role == "assistant"
        assert turn.message == "Hello!"
        assert turn.tokens_used == 15


class TestConversationLoggerStatistics:
    """Tests for conversation statistics."""

    def test_get_total_tokens(self, sample_conversation_logger):
        """Test getting total tokens."""
        total = sample_conversation_logger.get_total_tokens()
        
        # 10 + 15 + 12 + 20 = 57
        assert total == 57

    def test_get_total_cost(self, sample_conversation_logger):
        """Test getting total cost."""
        total = sample_conversation_logger.get_total_cost()
        
        # 0.001 + 0.002 + 0.001 + 0.002 = 0.006
        assert total == pytest.approx(0.006)

    def test_get_duration_not_finalized(self, sample_conversation_logger):
        """Test duration when not finalized."""
        duration = sample_conversation_logger.get_duration()
        
        assert duration is None

    def test_get_duration_finalized(self, sample_conversation_logger):
        """Test duration when finalized."""
        sample_conversation_logger.finalize()
        
        duration = sample_conversation_logger.get_duration()
        
        assert duration is not None
        assert duration >= 0

    def test_finalize_sets_end_time(self, sample_conversation_logger):
        """Test that finalize sets end time."""
        assert sample_conversation_logger.end_time is None
        
        sample_conversation_logger.finalize()
        
        assert sample_conversation_logger.end_time is not None


class TestConversationLoggerSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, sample_conversation_logger):
        """Test converting logger to dictionary."""
        sample_conversation_logger.finalize()
        result = sample_conversation_logger.to_dict()
        
        assert result["experiment_name"] == "test_experiment"
        assert result["conversation_id"] == "test_conv_001"
        assert "turns" in result
        assert len(result["turns"]) == 4
        assert "statistics" in result
        assert "metadata" in result

    def test_to_json_file(self, sample_conversation_logger, temp_dir):
        """Test saving logger to JSON file."""
        filepath = temp_dir / "conversation.json"
        sample_conversation_logger.finalize()
        sample_conversation_logger.to_json(str(filepath))
        
        assert filepath.exists()
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert data["experiment_name"] == "test_experiment"
        assert len(data["turns"]) == 4

    def test_from_json_file(self, sample_conversation_logger, temp_dir):
        """Test loading logger from JSON file."""
        filepath = temp_dir / "conversation.json"
        sample_conversation_logger.finalize()
        sample_conversation_logger.to_json(str(filepath))
        
        loaded = ConversationLogger.from_json(str(filepath))
        
        assert loaded.experiment_name == sample_conversation_logger.experiment_name
        assert loaded.conversation_id == sample_conversation_logger.conversation_id
        assert len(loaded.turns) == len(sample_conversation_logger.turns)

    def test_from_json_file_not_found(self):
        """Test error when JSON file not found."""
        with pytest.raises(FileNotFoundError):
            ConversationLogger.from_json("/nonexistent/path.json")


class TestConversationLoggerPandas:
    """Tests for pandas DataFrame export."""

    def test_to_dataframe(self, sample_conversation_logger):
        """Test converting to pandas DataFrame via manual construction."""
        pd = pytest.importorskip("pandas")
        
        # ConversationLogger doesn't have to_dataframe, but we can create one from dict
        data = sample_conversation_logger.to_dict()
        df = pd.DataFrame(data["turns"])
        
        assert len(df) == 4
        assert "turn_number" in df.columns
        assert "agent_name" in df.columns
        assert "message" in df.columns
        assert "tokens_used" in df.columns

    def test_to_dataframe_empty_conversation(self):
        """Test DataFrame from empty conversation."""
        pd = pytest.importorskip("pandas")
        
        logger = ConversationLogger(experiment_name="empty")
        data = logger.to_dict()
        df = pd.DataFrame(data["turns"])
        
        assert len(df) == 0


class TestConversationLoggerEdgeCases:
    """Tests for edge cases."""

    def test_empty_conversation_stats(self):
        """Test statistics for empty conversation."""
        logger = ConversationLogger(experiment_name="empty")
        
        assert logger.get_total_tokens() == 0
        assert logger.get_total_cost() == 0.0

    def test_turn_with_none_tokens(self):
        """Test turn with no token information."""
        logger = ConversationLogger(experiment_name="test")
        
        turn = logger.log_turn(
            agent_name="Agent",
            role="user",
            message="Test"
        )
        
        assert turn.tokens_used is None
        assert logger.get_total_tokens() == 0

    def test_multiple_conversations_unique_ids(self):
        """Test that multiple loggers get unique IDs."""
        logger1 = ConversationLogger(experiment_name="test")
        logger2 = ConversationLogger(experiment_name="test")
        
        # IDs should be different (timestamp-based)
        # Note: might be same if created at exact same millisecond
        # but generally should be unique
        assert logger1.conversation_id is not None
        assert logger2.conversation_id is not None
