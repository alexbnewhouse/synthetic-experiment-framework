"""
Experiment orchestration for multi-agent conversations.

This module provides the core Experiment class that orchestrates conversations
between multiple agents, manages data collection, and controls experimental flow.

Example:
    >>> from synthetic_experiments import Experiment
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.agents import ConversationAgent, Persona
    >>>
    >>> # Create agents
    >>> user_agent = ConversationAgent(
    ...     provider=OllamaProvider("llama2"),
    ...     persona=Persona(name="User", background="Curious person"),
    ...     role="user"
    ... )
    >>> advisor_agent = ConversationAgent(
    ...     provider=OllamaProvider("llama2"),
    ...     persona=Persona(name="Advisor", background="Helpful assistant"),
    ...     role="assistant"
    ... )
    >>>
    >>> # Run experiment
    >>> experiment = Experiment(
    ...     name="basic_conversation",
    ...     agents=[user_agent, advisor_agent]
    ... )
    >>> results = experiment.run(max_turns=10, initial_topic="climate change")
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.data.logger import ConversationLogger
from synthetic_experiments.data.storage import ExperimentStorage

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment.

    Attributes:
        name: Experiment name
        max_turns: Maximum conversation turns
        initial_topic: Starting topic for conversation
        topics: List of topics to cover (optional)
        stop_on_repetition: Stop if responses become repetitive
        repetition_threshold: Similarity threshold for repetition detection
        save_conversations: Whether to save conversation data
        output_dir: Directory for saving results
    """
    name: str
    max_turns: int = 20
    initial_topic: str = ""
    topics: List[str] = field(default_factory=list)
    stop_on_repetition: bool = False
    repetition_threshold: float = 0.9
    save_conversations: bool = True
    output_dir: str = "results"
    metadata: Dict[str, Any] = field(default_factory=dict)


class Experiment:
    """
    Orchestrates multi-agent conversation experiments.

    An Experiment manages the interaction between two or more agents,
    controls conversation flow, logs data, and can run multiple replicates
    of the same experimental condition.

    Attributes:
        name: Experiment name
        agents: List of conversation agents
        config: Experiment configuration
        storage: Data storage manager
        logger: Conversation logger (created per run)
    """

    def __init__(
        self,
        name: str,
        agents: List[ConversationAgent],
        config: Optional[ExperimentConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize an experiment.

        Args:
            name: Name of the experiment
            agents: List of conversation agents (typically 2)
            config: Optional experiment configuration
            output_dir: Optional output directory for results
        """
        self.name = name
        self.agents = agents
        self.config = config or ExperimentConfig(name=name)

        if output_dir:
            self.config.output_dir = output_dir

        # Initialize storage
        if self.config.save_conversations:
            self.storage = ExperimentStorage(
                base_dir=self.config.output_dir,
                metadata={
                    "experiment_name": self.name,
                    **self.config.metadata
                }
            )
        else:
            self.storage = None

        self.current_logger: Optional[ConversationLogger] = None

        logger.info(f"Experiment '{self.name}' initialized with {len(agents)} agents")

    def run(
        self,
        max_turns: Optional[int] = None,
        initial_topic: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationLogger:
        """
        Run a single conversation.

        Args:
            max_turns: Override max_turns from config
            initial_topic: Override initial_topic from config
            conversation_id: Optional custom conversation ID
            metadata: Additional metadata for this conversation

        Returns:
            ConversationLogger with full conversation data

        Raises:
            ValueError: If agents list is invalid
        """
        if len(self.agents) < 2:
            raise ValueError("Experiment requires at least 2 agents")

        # Use config values or overrides
        max_turns = max_turns or self.config.max_turns
        initial_topic = initial_topic or self.config.initial_topic

        # Reset agents
        for agent in self.agents:
            agent.reset_conversation()

        # Create logger
        conv_metadata = {
            **self.config.metadata,
            **(metadata or {}),
            "agents": [agent.name for agent in self.agents],
            "initial_topic": initial_topic
        }

        self.current_logger = ConversationLogger(
            experiment_name=self.name,
            conversation_id=conversation_id,
            metadata=conv_metadata
        )

        logger.info(
            f"Starting conversation: {self.current_logger.conversation_id} "
            f"(max_turns={max_turns})"
        )

        # Run conversation
        try:
            self._run_conversation(max_turns, initial_topic)
        except Exception as e:
            logger.error(f"Error during conversation: {e}")
            raise
        finally:
            # Finalize logger
            self.current_logger.finalize()

            # Save if configured
            if self.storage:
                self.storage.save_conversation(self.current_logger)

        logger.info(
            f"Conversation completed: {len(self.current_logger.turns)} turns, "
            f"{self.current_logger.get_total_tokens()} tokens"
        )

        return self.current_logger

    def _run_conversation(self, max_turns: int, initial_topic: str) -> None:
        """
        Execute the conversation between agents.

        Args:
            max_turns: Maximum number of turns
            initial_topic: Initial conversation topic
        """
        # Determine turn order (user agent goes first)
        user_agent = next((a for a in self.agents if a.role == "user"), self.agents[0])
        other_agents = [a for a in self.agents if a != user_agent]

        if not other_agents:
            raise ValueError("Need at least one assistant agent")

        assistant_agent = other_agents[0]

        # Start conversation with initial topic
        current_prompt = self._create_initial_prompt(initial_topic)

        for turn in range(max_turns):
            # User agent responds
            try:
                user_message = user_agent.respond(current_prompt)
                self.current_logger.log_message(user_agent.name, user_message)

                logger.debug(f"Turn {turn + 1}/{max_turns} - {user_agent.name}: {user_message.content[:100]}...")

                # Check for stopping conditions
                if self._should_stop(user_message.content):
                    logger.info(f"Stopping condition met at turn {turn + 1}")
                    break

                # Assistant agent responds
                assistant_message = assistant_agent.respond(user_message.content)
                self.current_logger.log_message(assistant_agent.name, assistant_message)

                logger.debug(f"Turn {turn + 1}/{max_turns} - {assistant_agent.name}: {assistant_message.content[:100]}...")

                # Prepare next prompt (assistant's response becomes the prompt)
                current_prompt = assistant_message.content

                # Check for stopping conditions
                if self._should_stop(assistant_message.content):
                    logger.info(f"Stopping condition met at turn {turn + 1}")
                    break

            except Exception as e:
                logger.error(f"Error at turn {turn + 1}: {e}")
                raise

    def _create_initial_prompt(self, initial_topic: str) -> str:
        """
        Create the initial conversation prompt.

        Args:
            initial_topic: The topic to start with

        Returns:
            Initial prompt string
        """
        if initial_topic:
            return f"Let's discuss {initial_topic}. What are your thoughts?"
        else:
            return "Hello! What would you like to talk about?"

    def _should_stop(self, message: str) -> bool:
        """
        Check if conversation should stop based on stopping conditions.

        Args:
            message: Latest message content

        Returns:
            True if conversation should stop
        """
        # Could implement various stopping conditions:
        # - Message length too short (agent giving up)
        # - Repetition detection
        # - Specific keywords
        # For now, keep it simple
        return len(message.strip()) < 10

    def run_multiple(
        self,
        num_conversations: int,
        **run_kwargs
    ) -> List[ConversationLogger]:
        """
        Run multiple conversation replicates.

        Args:
            num_conversations: Number of conversations to run
            **run_kwargs: Arguments to pass to run()

        Returns:
            List of ConversationLogger objects
        """
        results = []

        logger.info(f"Running {num_conversations} conversation replicates")

        for i in range(num_conversations):
            logger.info(f"Running replicate {i + 1}/{num_conversations}")

            # Add replicate number to metadata
            metadata = run_kwargs.get("metadata", {})
            metadata["replicate"] = i + 1
            run_kwargs["metadata"] = metadata

            result = self.run(**run_kwargs)
            results.append(result)

        logger.info(f"Completed {num_conversations} replicates")

        # Export summary if storage is available
        if self.storage:
            self.storage.export_summary_csv()
            self.storage.export_turns_csv()

        return results

    @classmethod
    def from_config_file(cls, config_path: str) -> "Experiment":
        """
        Create experiment from YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Experiment instance
        """
        from synthetic_experiments.experiments.config import load_experiment_config
        return load_experiment_config(config_path)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get experiment statistics.

        Returns:
            Dictionary with experiment-level statistics
        """
        if self.storage:
            return self.storage.get_statistics()
        else:
            return {
                "experiment_name": self.name,
                "agents": [a.name for a in self.agents],
                "storage_enabled": False
            }

    def __repr__(self) -> str:
        return (
            f"Experiment(name='{self.name}', agents={len(self.agents)}, "
            f"max_turns={self.config.max_turns})"
        )
