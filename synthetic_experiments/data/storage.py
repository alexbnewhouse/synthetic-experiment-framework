"""
Data storage and management for experiment results.

This module provides tools for saving experimental data in formats that are
easy to analyze with standard social science tools (pandas, R, SPSS, etc.).

Example:
    >>> from synthetic_experiments.data import ExperimentStorage
    >>>
    >>> storage = ExperimentStorage(base_dir="results/my_experiment")
    >>> storage.save_conversation(conversation_logger)
    >>> storage.export_summary_csv()
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv
import logging
from datetime import datetime

from synthetic_experiments.data.logger import ConversationLogger

logger = logging.getLogger(__name__)


class ExperimentStorage:
    """
    Manages storage of experiment data.

    Organizes conversation data in a structured directory format:
    - conversations/: Individual conversation JSON files
    - summary.csv: High-level statistics for all conversations
    - metadata.json: Experiment-level metadata

    Attributes:
        base_dir: Root directory for experiment data
        conversations_dir: Directory for individual conversations
        metadata: Experiment-level metadata
    """

    def __init__(
        self,
        base_dir: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment storage.

        Args:
            base_dir: Base directory for storing data
            metadata: Optional experiment-level metadata
        """
        self.base_dir = Path(base_dir)
        self.conversations_dir = self.base_dir / "conversations"
        self.metadata = metadata or {}

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)

        # Save metadata
        self._save_metadata()

        logger.info(f"Experiment storage initialized at {self.base_dir}")

    def _save_metadata(self) -> None:
        """Save experiment metadata to file."""
        metadata_path = self.base_dir / "metadata.json"

        # Add creation time if not present
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_conversation(
        self,
        conversation: ConversationLogger,
        overwrite: bool = False
    ) -> Path:
        """
        Save a conversation to storage.

        Args:
            conversation: ConversationLogger to save
            overwrite: Whether to overwrite existing file

        Returns:
            Path to saved conversation file

        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        filename = f"{conversation.conversation_id}.json"
        filepath = self.conversations_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(
                f"Conversation file already exists: {filepath}. "
                f"Use overwrite=True to replace it."
            )

        conversation.to_json(str(filepath))
        logger.info(f"Saved conversation: {conversation.conversation_id}")

        return filepath

    def load_conversation(self, conversation_id: str) -> ConversationLogger:
        """
        Load a conversation from storage.

        Args:
            conversation_id: ID of conversation to load

        Returns:
            ConversationLogger instance

        Raises:
            FileNotFoundError: If conversation file doesn't exist
        """
        filepath = self.conversations_dir / f"{conversation_id}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Conversation not found: {conversation_id}")

        return ConversationLogger.from_json(str(filepath))

    def list_conversations(self) -> List[str]:
        """
        List all conversation IDs in storage.

        Returns:
            List of conversation IDs
        """
        return [
            f.stem for f in self.conversations_dir.glob("*.json")
        ]

    def export_summary_csv(
        self,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Export summary statistics to CSV file.

        Creates a CSV file with one row per conversation, containing
        high-level statistics that are easy to analyze in R, pandas, etc.

        Args:
            output_path: Optional custom output path (defaults to summary.csv)

        Returns:
            Path to CSV file
        """
        if output_path is None:
            output_path = self.base_dir / "summary.csv"
        else:
            output_path = Path(output_path)

        conversations = self.list_conversations()

        if not conversations:
            logger.warning("No conversations to export")
            return output_path

        # Collect summary data
        rows = []
        for conv_id in conversations:
            try:
                conv = self.load_conversation(conv_id)
                data = conv.to_dict()

                # Extract key statistics
                row = {
                    "conversation_id": conv_id,
                    "experiment_name": data["experiment_name"],
                    "start_time": data["start_time"],
                    "end_time": data["end_time"],
                    "duration_seconds": data["duration_seconds"],
                    "total_turns": data["statistics"]["total_turns"],
                    "total_tokens": data["statistics"]["total_tokens"],
                    "total_cost": data["statistics"]["total_cost"],
                }

                # Add metadata fields
                for key, value in data.get("metadata", {}).items():
                    # Flatten simple metadata
                    if isinstance(value, (str, int, float, bool)):
                        row[f"meta_{key}"] = value

                rows.append(row)

            except Exception as e:
                logger.error(f"Error processing conversation {conv_id}: {e}")
                continue

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Exported summary CSV to {output_path}")

        return Path(output_path)

    def export_turns_csv(
        self,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Export all conversation turns to a CSV file.

        Creates a CSV with one row per turn, useful for turn-level analysis
        (sentiment analysis, linguistic features, etc.).

        Args:
            output_path: Optional custom output path (defaults to turns.csv)

        Returns:
            Path to CSV file
        """
        if output_path is None:
            output_path = self.base_dir / "turns.csv"
        else:
            output_path = Path(output_path)

        conversations = self.list_conversations()

        if not conversations:
            logger.warning("No conversations to export")
            return output_path

        # Collect all turns
        rows = []
        for conv_id in conversations:
            try:
                conv = self.load_conversation(conv_id)
                data = conv.to_dict()

                for turn in data.get("turns", []):
                    row = {
                        "conversation_id": conv_id,
                        "experiment_name": data["experiment_name"],
                        "turn_number": turn["turn_number"],
                        "timestamp": turn["timestamp"],
                        "agent_name": turn["agent_name"],
                        "role": turn["role"],
                        "message": turn["message"],
                        "tokens_used": turn.get("tokens_used"),
                        "cost": turn.get("cost"),
                    }

                    # Add turn metadata
                    for key, value in turn.get("metadata", {}).items():
                        if isinstance(value, (str, int, float, bool)):
                            row[f"meta_{key}"] = value

                    rows.append(row)

            except Exception as e:
                logger.error(f"Error processing conversation {conv_id}: {e}")
                continue

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Exported turns CSV to {output_path}")

        return Path(output_path)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall experiment statistics.

        Returns:
            Dictionary with experiment-level statistics
        """
        conversations = self.list_conversations()

        total_turns = 0
        total_tokens = 0
        total_cost = 0.0
        total_duration = 0.0

        for conv_id in conversations:
            try:
                conv = self.load_conversation(conv_id)
                total_turns += len(conv.turns)
                total_tokens += conv.get_total_tokens()
                total_cost += conv.get_total_cost()
                duration = conv.get_duration()
                if duration:
                    total_duration += duration
            except Exception as e:
                logger.error(f"Error processing {conv_id}: {e}")

        return {
            "total_conversations": len(conversations),
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_duration_seconds": total_duration,
            "avg_turns_per_conversation": total_turns / len(conversations) if conversations else 0,
            "avg_cost_per_conversation": total_cost / len(conversations) if conversations else 0,
        }

    def __repr__(self) -> str:
        return f"ExperimentStorage(dir='{self.base_dir}', conversations={len(self.list_conversations())})"
