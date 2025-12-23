"""
Tests for the ExperimentStorage class.
"""

import pytest
import json
from pathlib import Path

from synthetic_experiments.data.storage import ExperimentStorage
from synthetic_experiments.data.logger import ConversationLogger


class TestExperimentStorageInit:
    """Tests for ExperimentStorage initialization."""

    def test_storage_creation(self, temp_dir):
        """Test creating experiment storage."""
        storage = ExperimentStorage(
            base_dir=str(temp_dir / "experiment_results"),
            metadata={"experiment_type": "test"}
        )
        
        assert storage.base_dir.exists()
        assert storage.conversations_dir.exists()

    def test_storage_creates_directories(self, temp_dir):
        """Test that storage creates required directories."""
        base = temp_dir / "new_experiment" / "nested"
        storage = ExperimentStorage(base_dir=str(base))
        
        assert base.exists()
        assert (base / "conversations").exists()

    def test_storage_saves_metadata(self, temp_dir):
        """Test that storage saves experiment metadata."""
        storage = ExperimentStorage(
            base_dir=str(temp_dir / "with_metadata"),
            metadata={"condition": "treatment", "version": "1.0"}
        )
        
        metadata_path = storage.base_dir / "metadata.json"
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        assert data["condition"] == "treatment"
        assert data["version"] == "1.0"
        assert "created_at" in data


class TestExperimentStorageSave:
    """Tests for saving conversations."""

    def test_save_conversation(self, temp_dir, sample_conversation_logger):
        """Test saving a conversation."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "save_test"))
        sample_conversation_logger.finalize()
        
        filepath = storage.save_conversation(sample_conversation_logger)
        
        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_save_conversation_correct_filename(self, temp_dir, sample_conversation_logger):
        """Test that saved file has correct name."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "filename_test"))
        sample_conversation_logger.finalize()
        
        filepath = storage.save_conversation(sample_conversation_logger)
        
        expected_name = f"{sample_conversation_logger.conversation_id}.json"
        assert filepath.name == expected_name

    def test_save_conversation_no_overwrite(self, temp_dir, sample_conversation_logger):
        """Test that saving same conversation twice raises error."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "no_overwrite"))
        sample_conversation_logger.finalize()
        
        storage.save_conversation(sample_conversation_logger)
        
        with pytest.raises(FileExistsError):
            storage.save_conversation(sample_conversation_logger)

    def test_save_conversation_with_overwrite(self, temp_dir, sample_conversation_logger):
        """Test saving with overwrite=True."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "overwrite"))
        sample_conversation_logger.finalize()
        
        storage.save_conversation(sample_conversation_logger)
        filepath = storage.save_conversation(sample_conversation_logger, overwrite=True)
        
        assert filepath.exists()


class TestExperimentStorageLoad:
    """Tests for loading conversations."""

    def test_load_conversation(self, temp_dir, sample_conversation_logger):
        """Test loading a saved conversation."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "load_test"))
        sample_conversation_logger.finalize()
        storage.save_conversation(sample_conversation_logger)
        
        loaded = storage.load_conversation(sample_conversation_logger.conversation_id)
        
        assert loaded.experiment_name == sample_conversation_logger.experiment_name
        assert len(loaded.turns) == len(sample_conversation_logger.turns)

    def test_load_conversation_not_found(self, temp_dir):
        """Test error when loading non-existent conversation."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "load_notfound"))
        
        with pytest.raises(FileNotFoundError):
            storage.load_conversation("nonexistent_id")


class TestExperimentStorageList:
    """Tests for listing conversations."""

    def test_list_conversations_empty(self, temp_dir):
        """Test listing conversations when empty."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "list_empty"))
        
        conversations = storage.list_conversations()
        
        assert conversations == []

    def test_list_conversations(self, temp_dir, sample_conversation_logger):
        """Test listing conversations."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "list_test"))
        sample_conversation_logger.finalize()
        
        storage.save_conversation(sample_conversation_logger)
        
        # Create and save another conversation
        logger2 = ConversationLogger(
            experiment_name="test",
            conversation_id="another_conv"
        )
        logger2.log_turn(agent_name="A", role="user", message="Test")
        logger2.finalize()
        storage.save_conversation(logger2)
        
        conversations = storage.list_conversations()
        
        assert len(conversations) == 2
        assert sample_conversation_logger.conversation_id in conversations
        assert "another_conv" in conversations


class TestExperimentStorageExport:
    """Tests for data export."""

    def test_export_summary_csv(self, temp_dir, sample_conversation_logger):
        """Test exporting summary to CSV."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "csv_test"))
        sample_conversation_logger.finalize()
        storage.save_conversation(sample_conversation_logger)
        
        csv_path = storage.export_summary_csv()
        
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

    def test_export_summary_csv_content(self, temp_dir, sample_conversation_logger):
        """Test CSV content."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        storage = ExperimentStorage(base_dir=str(temp_dir / "csv_content"))
        sample_conversation_logger.finalize()
        storage.save_conversation(sample_conversation_logger)
        
        csv_path = storage.export_summary_csv()
        df = pd.read_csv(csv_path)
        
        assert len(df) == 1
        assert "conversation_id" in df.columns
        assert "total_turns" in df.columns
        assert "total_tokens" in df.columns

    def test_export_summary_csv_empty(self, temp_dir):
        """Test exporting CSV when no conversations."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "csv_empty"))
        
        csv_path = storage.export_summary_csv()
        
        # Should still create file path but might be empty
        assert csv_path is not None

    def test_export_summary_csv_custom_path(self, temp_dir, sample_conversation_logger):
        """Test exporting CSV to custom path."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "csv_custom"))
        sample_conversation_logger.finalize()
        storage.save_conversation(sample_conversation_logger)
        
        custom_path = temp_dir / "custom_summary.csv"
        csv_path = storage.export_summary_csv(output_path=str(custom_path))
        
        assert csv_path == custom_path
        assert custom_path.exists()


class TestExperimentStorageMultiple:
    """Tests for handling multiple conversations."""

    def test_multiple_conversations(self, temp_dir):
        """Test saving and loading multiple conversations."""
        storage = ExperimentStorage(base_dir=str(temp_dir / "multi"))
        
        conversations = []
        for i in range(5):
            logger = ConversationLogger(
                experiment_name="multi_test",
                conversation_id=f"conv_{i}",
                metadata={"index": i}
            )
            logger.log_turn(
                agent_name="Agent",
                role="user",
                message=f"Message {i}",
                tokens_used=10 * i
            )
            logger.finalize()
            storage.save_conversation(logger)
            conversations.append(logger.conversation_id)
        
        # List should return all
        listed = storage.list_conversations()
        assert len(listed) == 5
        
        # Each should be loadable
        for conv_id in conversations:
            loaded = storage.load_conversation(conv_id)
            assert loaded.conversation_id == conv_id

    def test_summary_csv_multiple(self, temp_dir):
        """Test CSV summary with multiple conversations."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        storage = ExperimentStorage(base_dir=str(temp_dir / "multi_csv"))
        
        for i in range(3):
            logger = ConversationLogger(
                experiment_name="test",
                conversation_id=f"conv_{i}"
            )
            logger.log_turn(
                agent_name="Agent",
                role="user",
                message=f"Message {i}",
                tokens_used=100
            )
            logger.finalize()
            storage.save_conversation(logger)
        
        csv_path = storage.export_summary_csv()
        df = pd.read_csv(csv_path)
        
        assert len(df) == 3


class TestExperimentStorageMetadata:
    """Tests for metadata handling."""

    def test_metadata_in_summary(self, temp_dir):
        """Test that conversation metadata appears in summary."""
        pytest.importorskip("pandas")
        import pandas as pd
        
        storage = ExperimentStorage(base_dir=str(temp_dir / "meta_summary"))
        
        logger = ConversationLogger(
            experiment_name="test",
            conversation_id="meta_test",
            metadata={"condition": "treatment", "participant_id": 42}
        )
        logger.log_turn(agent_name="A", role="user", message="Test")
        logger.finalize()
        storage.save_conversation(logger)
        
        csv_path = storage.export_summary_csv()
        df = pd.read_csv(csv_path)
        
        # Metadata should be prefixed with meta_
        assert "meta_condition" in df.columns or "condition" in df.columns
