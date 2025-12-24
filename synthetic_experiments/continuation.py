"""
Conversation continuation and checkpointing support.

This module provides utilities for saving and resuming conversations,
enabling interruption recovery and conversation branching.

Example:
    >>> from synthetic_experiments.continuation import (
    ...     save_conversation_state,
    ...     load_conversation_state,
    ...     ContinuableExperiment
    ... )
    >>> 
    >>> # Save state mid-conversation
    >>> save_conversation_state(experiment, "checkpoint.json")
    >>> 
    >>> # Resume later
    >>> experiment = load_conversation_state("checkpoint.json")
    >>> experiment.continue_conversation(additional_turns=10)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """
    Serializable state of a conversation in progress.
    
    Attributes:
        experiment_name: Name of the experiment
        conversation_id: Unique conversation identifier
        turn_count: Number of completed turns
        max_turns: Maximum turns configured
        agent_histories: Conversation history for each agent
        agent_configs: Configuration for recreating agents
        current_prompt: The prompt for the next turn
        metadata: Additional metadata
        created_at: When the state was created
        last_updated: When last updated
    """
    experiment_name: str
    conversation_id: str
    turn_count: int
    max_turns: int
    agent_histories: List[List[Dict[str, str]]]
    agent_configs: List[Dict[str, Any]]
    current_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConversationState":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    def save(self, filepath: str):
        """Save state to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved conversation state to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ConversationState":
        """Load state from file."""
        with open(filepath) as f:
            return cls.from_json(f.read())


class ContinuableExperiment:
    """
    Experiment wrapper that supports interruption and continuation.
    
    This wraps a standard Experiment and adds the ability to:
    - Save conversation state at any point
    - Resume interrupted conversations
    - Continue completed conversations with more turns
    
    Example:
        >>> from synthetic_experiments import Experiment
        >>> 
        >>> # Create continuable experiment
        >>> base_exp = Experiment(name="my_exp", agents=[user, advisor])
        >>> experiment = ContinuableExperiment(base_exp, checkpoint_dir="checkpoints")
        >>> 
        >>> # Run with auto-checkpointing
        >>> result = experiment.run(max_turns=20, checkpoint_every=5)
        >>> 
        >>> # If interrupted, resume later
        >>> experiment = ContinuableExperiment.resume("checkpoints/my_exp_latest.json")
        >>> result = experiment.continue_conversation(additional_turns=10)
    """
    
    def __init__(
        self,
        experiment=None,
        checkpoint_dir: str = "checkpoints",
        auto_checkpoint: bool = True,
        checkpoint_every: int = 5
    ):
        """
        Initialize continuable experiment.
        
        Args:
            experiment: Base Experiment instance (or None if resuming)
            checkpoint_dir: Directory for checkpoint files
            auto_checkpoint: Automatically save checkpoints during run
            checkpoint_every: Save checkpoint every N turns
        """
        self._experiment = experiment
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_every = checkpoint_every
        
        self._state: Optional[ConversationState] = None
        self._current_turn = 0
    
    @property
    def name(self):
        return self._experiment.name if self._experiment else self._state.experiment_name
    
    @property
    def agents(self):
        return self._experiment.agents if self._experiment else []
    
    @classmethod
    def resume(cls, checkpoint_path: str) -> "ContinuableExperiment":
        """
        Resume an experiment from a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            ContinuableExperiment ready to continue
        """
        state = ConversationState.load(checkpoint_path)
        
        # Recreate the experiment from state
        instance = cls(checkpoint_dir=str(Path(checkpoint_path).parent))
        instance._state = state
        instance._current_turn = state.turn_count
        
        # Recreate experiment and agents
        instance._recreate_experiment_from_state()
        
        logger.info(f"Resumed experiment '{state.experiment_name}' at turn {state.turn_count}")
        return instance
    
    def _recreate_experiment_from_state(self):
        """Recreate experiment from saved state."""
        from synthetic_experiments import Experiment
        from synthetic_experiments.agents import ConversationAgent, Persona
        from synthetic_experiments.providers import OllamaProvider, ClaudeProvider, OpenAIProvider
        
        state = self._state
        
        # Map provider types to classes
        provider_classes = {
            'ollama': OllamaProvider,
            'claude': ClaudeProvider,
            'openai': OpenAIProvider
        }
        
        # Recreate agents
        agents = []
        for i, agent_config in enumerate(state.agent_configs):
            # Create provider
            provider_type = agent_config.get('provider_type', 'ollama')
            provider_model = agent_config.get('provider_model', 'llama2')
            ProviderClass = provider_classes.get(provider_type, OllamaProvider)
            provider = ProviderClass(model_name=provider_model)
            
            # Create persona
            persona = Persona(**agent_config.get('persona', {'name': f'Agent{i}'}))
            
            # Create agent
            agent = ConversationAgent(
                provider=provider,
                persona=persona,
                role=agent_config.get('role', 'user')
            )
            
            # Restore conversation history
            if i < len(state.agent_histories):
                for msg in state.agent_histories[i]:
                    agent.conversation_history.append(msg)
            
            agents.append(agent)
        
        # Create experiment
        self._experiment = Experiment(
            name=state.experiment_name,
            agents=agents
        )
    
    def _extract_agent_config(self, agent) -> Dict[str, Any]:
        """Extract serializable config from an agent."""
        config = {
            'role': agent.role,
            'name': agent.name,
            'persona': {
                'name': agent.persona.name,
                'background': agent.persona.background,
                'political_orientation': agent.persona.political_orientation,
                'communication_style': agent.persona.communication_style,
                'goals': agent.persona.goals,
                'beliefs': agent.persona.beliefs,
            }
        }
        
        # Extract provider info
        provider = agent.provider
        provider_type = type(provider).__name__.lower().replace('provider', '')
        config['provider_type'] = provider_type
        config['provider_model'] = provider.model_name
        
        return config
    
    def _save_checkpoint(self, current_prompt: str = ""):
        """Save current state to checkpoint."""
        if not self._experiment:
            return
        
        # Extract agent histories
        agent_histories = []
        for agent in self._experiment.agents:
            history = [
                {'role': msg.get('role', 'user'), 'content': msg.get('content', '')}
                for msg in agent.conversation_history
            ]
            agent_histories.append(history)
        
        # Extract agent configs
        agent_configs = [
            self._extract_agent_config(agent)
            for agent in self._experiment.agents
        ]
        
        # Create state
        self._state = ConversationState(
            experiment_name=self._experiment.name,
            conversation_id=self._experiment.current_logger.conversation_id if self._experiment.current_logger else "",
            turn_count=self._current_turn,
            max_turns=self._experiment.config.max_turns,
            agent_histories=agent_histories,
            agent_configs=agent_configs,
            current_prompt=current_prompt,
            metadata=self._experiment.config.metadata
        )
        
        # Save to file
        checkpoint_path = self.checkpoint_dir / f"{self._experiment.name}_latest.json"
        self._state.save(str(checkpoint_path))
        
        # Also save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = self.checkpoint_dir / f"{self._experiment.name}_{timestamp}.json"
        self._state.save(str(timestamped_path))
    
    def run(
        self,
        max_turns: Optional[int] = None,
        initial_topic: Optional[str] = None,
        **kwargs
    ):
        """
        Run conversation with checkpointing.
        
        Args:
            max_turns: Maximum turns
            initial_topic: Starting topic
            **kwargs: Additional run arguments
            
        Returns:
            ConversationLogger with results
        """
        from synthetic_experiments.data.logger import ConversationLogger
        
        max_turns = max_turns or self._experiment.config.max_turns
        initial_topic = initial_topic or self._experiment.config.initial_topic
        
        # Reset agents
        for agent in self._experiment.agents:
            agent.reset_conversation()
        
        # Create logger
        self._experiment.current_logger = ConversationLogger(
            experiment_name=self._experiment.name,
            metadata=kwargs.get('metadata', {})
        )
        
        self._current_turn = 0
        
        # Run with checkpointing
        self._run_with_checkpoints(max_turns, initial_topic)
        
        # Finalize
        self._experiment.current_logger.finalize()
        
        if self._experiment.storage:
            self._experiment.storage.save_conversation(self._experiment.current_logger)
        
        # Clean up checkpoint on successful completion
        latest_checkpoint = self.checkpoint_dir / f"{self._experiment.name}_latest.json"
        if latest_checkpoint.exists():
            latest_checkpoint.unlink()
        
        return self._experiment.current_logger
    
    def _run_with_checkpoints(self, max_turns: int, initial_topic: str):
        """Run conversation with periodic checkpointing."""
        agent_order = self._experiment._get_agent_order()
        current_prompt = self._experiment._create_initial_prompt(initial_topic)
        last_speaker = None
        
        for turn in range(max_turns):
            self._current_turn = turn + 1
            
            # Auto-checkpoint
            if self.auto_checkpoint and turn > 0 and turn % self.checkpoint_every == 0:
                self._save_checkpoint(current_prompt)
                logger.debug(f"Checkpoint saved at turn {turn}")
            
            # Get next agent
            if self._experiment.config.turn_order == "random":
                import random
                available = [a for a in self._experiment.agents if a != last_speaker]
                current_agent = random.choice(available) if available else self._experiment.agents[0]
            else:
                current_agent = agent_order[turn % len(agent_order)]
            
            try:
                message = current_agent.respond(current_prompt)
                self._experiment.current_logger.log_message(current_agent.name, message)
                
                if self._experiment._should_stop(message.content):
                    break
                
                if len(self._experiment.agents) > 2:
                    current_prompt = f"[{current_agent.name}]: {message.content}"
                else:
                    current_prompt = message.content
                
                last_speaker = current_agent
                
            except Exception as e:
                # Save checkpoint on error
                self._save_checkpoint(current_prompt)
                raise
    
    def continue_conversation(
        self,
        additional_turns: int = 10,
        new_prompt: Optional[str] = None
    ):
        """
        Continue an interrupted or completed conversation.
        
        Args:
            additional_turns: Number of additional turns to run
            new_prompt: Optional new prompt to inject
            
        Returns:
            ConversationLogger with updated results
        """
        if not self._state:
            raise ValueError("No saved state to continue from")
        
        logger.info(f"Continuing conversation from turn {self._state.turn_count}")
        
        # Use saved prompt or new one
        current_prompt = new_prompt or self._state.current_prompt
        
        # Create/update logger
        if not self._experiment.current_logger:
            from synthetic_experiments.data.logger import ConversationLogger
            self._experiment.current_logger = ConversationLogger(
                experiment_name=self._experiment.name,
                conversation_id=self._state.conversation_id
            )
        
        # Continue running
        self._run_with_checkpoints(
            max_turns=additional_turns,
            initial_topic=""  # Already have prompt
        )
        
        self._experiment.current_logger.finalize()
        
        return self._experiment.current_logger
    
    def get_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        latest = self.checkpoint_dir / f"{self.name}_latest.json"
        return latest if latest.exists() else None


def save_conversation_state(
    experiment,
    filepath: str,
    current_prompt: str = ""
):
    """
    Save the current state of an experiment for later resumption.
    
    Args:
        experiment: Experiment instance
        filepath: Path to save state
        current_prompt: Current prompt in conversation
    """
    cont = ContinuableExperiment(experiment)
    cont._current_turn = len(experiment.current_logger.turns) if experiment.current_logger else 0
    cont._save_checkpoint(current_prompt)
    
    # Copy to requested path
    import shutil
    latest = cont.checkpoint_dir / f"{experiment.name}_latest.json"
    shutil.copy(latest, filepath)
    
    logger.info(f"Saved conversation state to {filepath}")


def load_conversation_state(filepath: str) -> ContinuableExperiment:
    """
    Load a saved conversation state.
    
    Args:
        filepath: Path to saved state
        
    Returns:
        ContinuableExperiment ready to continue
    """
    return ContinuableExperiment.resume(filepath)
