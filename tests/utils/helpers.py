"""
Helper utilities for testing the synthetic_experiments package.
"""

from typing import List, Dict, Any
from unittest.mock import MagicMock

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult
)
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.data.logger import ConversationLogger


def create_mock_conversation(
    turns: int = 5,
    include_tokens: bool = True,
    include_cost: bool = True
) -> ConversationLogger:
    """
    Create a mock conversation logger with specified number of turns.
    
    Args:
        turns: Number of turns to generate
        include_tokens: Whether to include token counts
        include_cost: Whether to include cost information
    
    Returns:
        ConversationLogger with mock data
    """
    logger = ConversationLogger(
        experiment_name="mock_experiment",
        metadata={"mock": True}
    )
    
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        agent_name = "User" if role == "user" else "Assistant"
        
        logger.log_turn(
            agent_name=agent_name,
            role=role,
            message=f"Mock message {i + 1} from {agent_name}",
            tokens_used=10 * (i + 1) if include_tokens else None,
            cost=0.001 * (i + 1) if include_cost else None,
            metadata={"turn_index": i}
        )
    
    return logger


def create_political_conversation(
    orientation: str = "liberal",
    turns: int = 4
) -> ConversationLogger:
    """
    Create a mock political conversation for testing analysis.
    
    Args:
        orientation: 'liberal', 'conservative', or 'neutral'
        turns: Number of turns
    
    Returns:
        ConversationLogger with political content
    """
    messages = {
        "liberal": [
            "Climate change requires immediate government action.",
            "Universal healthcare is a human right.",
            "We need stronger social justice policies.",
            "Renewable energy investment is crucial."
        ],
        "conservative": [
            "Free market solutions work better than government intervention.",
            "Limited government protects individual freedom.",
            "Traditional values form the foundation of society.",
            "Personal responsibility is key to success."
        ],
        "neutral": [
            "There are multiple perspectives on this issue.",
            "Both sides make valid points.",
            "We should consider different viewpoints.",
            "Finding common ground is important."
        ]
    }
    
    selected_messages = messages.get(orientation, messages["neutral"])
    
    logger = ConversationLogger(
        experiment_name=f"political_{orientation}",
        metadata={"orientation": orientation}
    )
    
    for i in range(min(turns, len(selected_messages))):
        role = "user" if i % 2 == 0 else "assistant"
        logger.log_turn(
            agent_name=f"{orientation.capitalize()} {role.capitalize()}",
            role=role,
            message=selected_messages[i],
            tokens_used=20
        )
    
    return logger


class ResponseSequenceMock:
    """
    Helper class to create a sequence of predefined responses.
    """
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.index = 0
    
    def get_next(self) -> str:
        """Get the next response in sequence."""
        response = self.responses[self.index % len(self.responses)]
        self.index += 1
        return response
    
    def reset(self):
        """Reset the sequence."""
        self.index = 0


def assert_conversation_valid(logger: ConversationLogger):
    """
    Assert that a conversation logger has valid structure.
    
    Args:
        logger: ConversationLogger to validate
    
    Raises:
        AssertionError if validation fails
    """
    assert logger.experiment_name, "Experiment name must be set"
    assert logger.conversation_id, "Conversation ID must be set"
    assert logger.start_time is not None, "Start time must be set"
    
    for i, turn in enumerate(logger.turns):
        assert turn.turn_number == i + 1, f"Turn number mismatch at index {i}"
        assert turn.role in ["user", "assistant", "system"], f"Invalid role: {turn.role}"
        assert turn.message, "Turn message must not be empty"


def assert_metrics_valid(metrics: Dict[str, Any]):
    """
    Assert that metrics dictionary has expected structure.
    
    Args:
        metrics: Metrics dictionary to validate
    
    Raises:
        AssertionError if validation fails
    """
    required_keys = [
        "conversation_id",
        "total_turns",
        "total_tokens",
        "total_cost"
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"
    
    assert metrics["total_turns"] >= 0, "Total turns must be non-negative"
    assert metrics["total_tokens"] >= 0, "Total tokens must be non-negative"
    assert metrics["total_cost"] >= 0, "Total cost must be non-negative"
