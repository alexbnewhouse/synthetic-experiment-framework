"""
Conversation logging system for experiments.

This module provides tools for logging conversation data during experiments,
enabling detailed analysis and reproducibility of research findings.

Example:
    >>> from synthetic_experiments.data import ConversationLogger
    >>>
    >>> logger = ConversationLogger(experiment_name="political_study")
    >>> logger.log_turn(
    ...     agent_name="User",
    ...     message="What's your view on climate policy?",
    ...     metadata={"turn": 1}
    ... )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

from synthetic_experiments.providers.base import Message

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """
    Represents a single turn in a conversation.

    Attributes:
        turn_number: Sequential turn number in conversation
        timestamp: When the turn occurred
        agent_name: Name of the speaking agent
        role: Agent's role ('user' or 'assistant')
        message: The message content
        tokens_used: Number of tokens in generation
        cost: Cost of generation
        metadata: Additional turn-specific data
    """
    turn_number: int
    timestamp: datetime
    agent_name: str
    role: str
    message: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "role": self.role,
            "message": self.message,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "metadata": self.metadata
        }


class ConversationLogger:
    """
    Logs conversation data during experiments.

    The logger records all conversation turns with metadata, enabling
    detailed post-experiment analysis. Data is stored both in memory
    and written to disk for persistence.

    Attributes:
        experiment_name: Name of the experiment
        conversation_id: Unique identifier for this conversation
        turns: List of conversation turns
        metadata: Experiment-level metadata
        start_time: When conversation started
    """

    def __init__(
        self,
        experiment_name: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversation logger.

        Args:
            experiment_name: Name of the experiment
            conversation_id: Optional unique ID (auto-generated if not provided)
            metadata: Optional experiment-level metadata
        """
        self.experiment_name = experiment_name
        self.conversation_id = conversation_id or self._generate_id()
        self.metadata = metadata or {}
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

        self.turns: List[ConversationTurn] = []
        self.current_turn = 0

    def _generate_id(self) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}"

    def log_turn(
        self,
        agent_name: str,
        role: str,
        message: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Log a conversation turn.

        Args:
            agent_name: Name of the speaking agent
            role: Agent's role ('user' or 'assistant')
            message: The message content
            tokens_used: Number of tokens used
            cost: Cost of generation
            metadata: Additional turn metadata

        Returns:
            ConversationTurn object
        """
        self.current_turn += 1

        turn = ConversationTurn(
            turn_number=self.current_turn,
            timestamp=datetime.now(),
            agent_name=agent_name,
            role=role,
            message=message,
            tokens_used=tokens_used,
            cost=cost,
            metadata=metadata or {}
        )

        self.turns.append(turn)
        return turn

    def log_message(
        self,
        agent_name: str,
        message: Message
    ) -> ConversationTurn:
        """
        Log a message from a conversation agent.

        Args:
            agent_name: Name of the agent
            message: Message object

        Returns:
            ConversationTurn object
        """
        return self.log_turn(
            agent_name=agent_name,
            role=message.role,
            message=message.content,
            tokens_used=message.metadata.get("tokens_used"),
            cost=message.metadata.get("cost"),
            metadata=message.metadata
        )

    def finalize(self) -> None:
        """Mark conversation as complete."""
        self.end_time = datetime.now()

    def get_duration(self) -> Optional[float]:
        """
        Get conversation duration in seconds.

        Returns:
            Duration in seconds, or None if not finalized
        """
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_total_tokens(self) -> int:
        """Get total tokens used in conversation."""
        return sum(turn.tokens_used or 0 for turn in self.turns)

    def get_total_cost(self) -> float:
        """Get total cost of conversation."""
        return sum(turn.cost or 0.0 for turn in self.turns)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert logger data to dictionary.

        Returns:
            Complete conversation data as dictionary
        """
        return {
            "experiment_name": self.experiment_name,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.get_duration(),
            "turns": [turn.to_dict() for turn in self.turns],
            "statistics": {
                "total_turns": len(self.turns),
                "total_tokens": self.get_total_tokens(),
                "total_cost": self.get_total_cost()
            }
        }

    def to_json(self, file_path: Optional[str] = None) -> str:
        """
        Convert to JSON string or save to file.

        Args:
            file_path: Optional path to save JSON file

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if file_path:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(json_str)
            logger.info(f"Conversation saved to {file_path}")

        return json_str

    @classmethod
    def from_json(cls, file_path: str) -> "ConversationLogger":
        """
        Load conversation from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            ConversationLogger instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        logger_obj = cls(
            experiment_name=data["experiment_name"],
            conversation_id=data["conversation_id"],
            metadata=data.get("metadata", {})
        )

        # Restore turns
        for turn_data in data.get("turns", []):
            turn_data_copy = turn_data.copy()
            turn_data_copy["timestamp"] = datetime.fromisoformat(turn_data["timestamp"])
            turn = ConversationTurn(**turn_data_copy)
            logger_obj.turns.append(turn)

        logger_obj.current_turn = len(logger_obj.turns)

        if data.get("start_time"):
            logger_obj.start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            logger_obj.end_time = datetime.fromisoformat(data["end_time"])

        return logger_obj

    def __repr__(self) -> str:
        return (
            f"ConversationLogger(experiment='{self.experiment_name}', "
            f"id='{self.conversation_id}', turns={len(self.turns)})"
        )
