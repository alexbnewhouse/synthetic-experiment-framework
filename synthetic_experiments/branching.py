"""
Conversation branching support.

This module enables forking conversations to explore different paths,
useful for counterfactual analysis and A/B testing response strategies.

Example:
    >>> from synthetic_experiments.branching import (
    ...     ConversationTree,
    ...     BranchingExperiment,
    ...     fork_conversation
    ... )
    >>> 
    >>> # Run initial conversation
    >>> result = experiment.run(max_turns=5)
    >>> 
    >>> # Fork at turn 3 to try different responses
    >>> tree = ConversationTree(result)
    >>> branch = tree.fork_at(turn=3, new_message="What if we consider...")
    >>> branch_result = branch.continue_conversation(max_turns=5)
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
import copy
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Branch:
    """
    A branch in the conversation tree.
    
    Attributes:
        id: Unique branch identifier
        parent_id: ID of parent branch (None for root)
        fork_turn: Turn number where this branch forked
        messages: Messages in this branch
        metadata: Branch metadata
        children: Child branch IDs
    """
    id: str
    parent_id: Optional[str]
    fork_turn: int
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @property
    def turn_count(self) -> int:
        return len(self.messages)
    
    def get_message(self, turn: int) -> Optional[Dict[str, str]]:
        """Get message at specific turn."""
        if 0 <= turn < len(self.messages):
            return self.messages[turn]
        return None


class ConversationTree:
    """
    Tree structure for managing conversation branches.
    
    Enables exploring multiple conversation paths from the same starting point.
    
    Example:
        >>> # Create tree from initial conversation
        >>> tree = ConversationTree(conversation_logger)
        >>> 
        >>> # Fork at turn 5
        >>> branch1 = tree.fork_at(turn=5, new_message="Let's explore option A")
        >>> branch2 = tree.fork_at(turn=5, new_message="Let's explore option B")
        >>> 
        >>> # Continue each branch independently
        >>> result1 = tree.continue_branch(branch1.id, max_turns=5)
        >>> result2 = tree.continue_branch(branch2.id, max_turns=5)
        >>> 
        >>> # Compare branches
        >>> tree.compare_branches([branch1.id, branch2.id])
    """
    
    def __init__(self, logger=None, name: str = "conversation_tree"):
        """
        Initialize conversation tree.
        
        Args:
            logger: ConversationLogger with initial conversation
            name: Tree name for identification
        """
        self.name = name
        self.branches: Dict[str, Branch] = {}
        self.root_id: Optional[str] = None
        self._experiment = None
        
        if logger:
            self._initialize_from_logger(logger)
    
    def _initialize_from_logger(self, logger):
        """Create root branch from conversation logger."""
        messages = [
            {'agent': turn.get('agent', ''), 'content': turn.get('content', '')}
            for turn in logger.turns
        ]
        
        root = Branch(
            id="root",
            parent_id=None,
            fork_turn=0,
            messages=messages,
            metadata={
                'experiment_name': logger.experiment_name,
                'original': True
            }
        )
        
        self.branches[root.id] = root
        self.root_id = root.id
    
    def set_experiment(self, experiment):
        """
        Set the experiment for continuing conversations.
        
        Args:
            experiment: Experiment instance
        """
        self._experiment = experiment
    
    def fork_at(
        self,
        turn: int,
        new_message: Optional[str] = None,
        branch_id: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Branch:
        """
        Fork conversation at specified turn.
        
        Args:
            turn: Turn number to fork at (0-indexed)
            new_message: Optional replacement message for the fork point
            branch_id: Branch to fork from (default: root)
            metadata: Additional metadata for new branch
            
        Returns:
            New Branch instance
        """
        source_id = branch_id or self.root_id
        source = self.branches.get(source_id)
        
        if not source:
            raise ValueError(f"Branch {source_id} not found")
        
        if turn >= len(source.messages):
            raise ValueError(f"Turn {turn} exceeds branch length ({len(source.messages)})")
        
        # Copy messages up to fork point
        forked_messages = copy.deepcopy(source.messages[:turn])
        
        # Add new message if provided
        if new_message:
            # Determine speaker based on turn pattern
            if forked_messages:
                last_agent = forked_messages[-1].get('agent', '')
                # Alternate speakers
                agents = list(set(m.get('agent', '') for m in source.messages))
                agents = [a for a in agents if a != last_agent] or [last_agent]
                next_agent = agents[0]
            else:
                next_agent = source.messages[0].get('agent', 'User') if source.messages else 'User'
            
            forked_messages.append({
                'agent': next_agent,
                'content': new_message
            })
        
        # Create new branch
        new_branch = Branch(
            id=str(uuid.uuid4())[:8],
            parent_id=source_id,
            fork_turn=turn,
            messages=forked_messages,
            metadata=metadata or {}
        )
        
        # Register branch
        self.branches[new_branch.id] = new_branch
        source.children.append(new_branch.id)
        
        logger.info(f"Created branch {new_branch.id} forking from {source_id} at turn {turn}")
        
        return new_branch
    
    def continue_branch(
        self,
        branch_id: str,
        max_turns: int = 10,
        initial_prompt: Optional[str] = None
    ) -> Branch:
        """
        Continue a branch with additional turns.
        
        Args:
            branch_id: Branch to continue
            max_turns: Maximum additional turns
            initial_prompt: Optional prompt to continue with
            
        Returns:
            Updated Branch
        """
        if not self._experiment:
            raise ValueError("No experiment set. Call set_experiment() first.")
        
        branch = self.branches.get(branch_id)
        if not branch:
            raise ValueError(f"Branch {branch_id} not found")
        
        # Restore agent state from branch messages
        for agent in self._experiment.agents:
            agent.reset_conversation()
            for msg in branch.messages:
                if msg.get('agent') == agent.name:
                    agent.conversation_history.append({
                        'role': 'assistant',
                        'content': msg.get('content', '')
                    })
                else:
                    agent.conversation_history.append({
                        'role': 'user',
                        'content': msg.get('content', '')
                    })
        
        # Determine starting prompt
        if initial_prompt:
            current_prompt = initial_prompt
        elif branch.messages:
            current_prompt = branch.messages[-1].get('content', '')
        else:
            current_prompt = "Continue the conversation."
        
        # Run additional turns
        agent_order = self._experiment._get_agent_order()
        
        for turn in range(max_turns):
            current_agent = agent_order[(len(branch.messages) + turn) % len(agent_order)]
            
            try:
                message = current_agent.respond(current_prompt)
                branch.messages.append({
                    'agent': current_agent.name,
                    'content': message.content
                })
                
                if self._experiment._should_stop(message.content):
                    break
                
                current_prompt = message.content
                
            except Exception as e:
                logger.error(f"Error continuing branch: {e}")
                break
        
        return branch
    
    def get_branch_path(self, branch_id: str) -> List[str]:
        """
        Get path from root to branch.
        
        Args:
            branch_id: Target branch ID
            
        Returns:
            List of branch IDs from root to target
        """
        path = []
        current_id = branch_id
        
        while current_id:
            path.append(current_id)
            branch = self.branches.get(current_id)
            current_id = branch.parent_id if branch else None
        
        return list(reversed(path))
    
    def get_full_conversation(self, branch_id: str) -> List[Dict[str, str]]:
        """
        Get complete conversation history for a branch.
        
        Args:
            branch_id: Branch ID
            
        Returns:
            Full message history
        """
        branch = self.branches.get(branch_id)
        return copy.deepcopy(branch.messages) if branch else []
    
    def compare_branches(
        self,
        branch_ids: List[str],
        metric_func: Callable[[List[Dict]], float] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple branches using a metric.
        
        Args:
            branch_ids: Branches to compare
            metric_func: Function to compute metric from messages
            
        Returns:
            Comparison results
        """
        results = {}
        
        for branch_id in branch_ids:
            branch = self.branches.get(branch_id)
            if not branch:
                continue
            
            results[branch_id] = {
                'turn_count': len(branch.messages),
                'fork_turn': branch.fork_turn,
                'parent': branch.parent_id
            }
            
            if metric_func:
                results[branch_id]['metric'] = metric_func(branch.messages)
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        return {
            'name': self.name,
            'root_id': self.root_id,
            'branches': {
                bid: {
                    'id': b.id,
                    'parent_id': b.parent_id,
                    'fork_turn': b.fork_turn,
                    'messages': b.messages,
                    'metadata': b.metadata,
                    'children': b.children,
                    'created_at': b.created_at
                }
                for bid, b in self.branches.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTree":
        """Deserialize tree from dictionary."""
        tree = cls(name=data.get('name', 'conversation_tree'))
        tree.root_id = data.get('root_id')
        
        for bid, bdata in data.get('branches', {}).items():
            tree.branches[bid] = Branch(**bdata)
        
        return tree
    
    def visualize(self) -> str:
        """
        Create text visualization of tree structure.
        
        Returns:
            ASCII tree representation
        """
        lines = [f"ConversationTree: {self.name}"]
        lines.append("=" * 40)
        
        def draw_branch(branch_id: str, prefix: str = "", is_last: bool = True):
            branch = self.branches.get(branch_id)
            if not branch:
                return
            
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}[{branch.id}] (turns: {len(branch.messages)}, fork: {branch.fork_turn})")
            
            extension = "    " if is_last else "│   "
            children = branch.children
            for i, child_id in enumerate(children):
                draw_branch(child_id, prefix + extension, i == len(children) - 1)
        
        if self.root_id:
            root = self.branches[self.root_id]
            lines.append(f"[{root.id}] ROOT (turns: {len(root.messages)})")
            for i, child_id in enumerate(root.children):
                draw_branch(child_id, "", i == len(root.children) - 1)
        
        return "\n".join(lines)


class BranchingExperiment:
    """
    Experiment wrapper that enables branching exploration.
    
    Example:
        >>> from synthetic_experiments import Experiment
        >>> 
        >>> # Create branching experiment
        >>> base_exp = Experiment(name="my_exp", agents=[user, advisor])
        >>> branching_exp = BranchingExperiment(base_exp)
        >>> 
        >>> # Run initial conversation
        >>> result = branching_exp.run(max_turns=10)
        >>> 
        >>> # Fork and explore alternatives
        >>> alt1 = branching_exp.fork_and_continue(
        ...     turn=5,
        ...     new_message="What if we tried a different approach?",
        ...     additional_turns=5
        ... )
        >>> 
        >>> # Compare outcomes
        >>> branching_exp.compare_outcomes()
    """
    
    def __init__(self, experiment):
        """
        Initialize branching experiment.
        
        Args:
            experiment: Base Experiment instance
        """
        self._experiment = experiment
        self.tree = ConversationTree(name=experiment.name)
        self.tree.set_experiment(experiment)
        self._current_branch_id: Optional[str] = None
    
    def run(self, max_turns: int = 20, initial_topic: str = None, **kwargs):
        """
        Run initial conversation (creates root branch).
        
        Args:
            max_turns: Maximum turns
            initial_topic: Starting topic
            **kwargs: Additional run arguments
            
        Returns:
            ConversationLogger with results
        """
        result = self._experiment.run(
            max_turns=max_turns,
            initial_topic=initial_topic,
            **kwargs
        )
        
        # Initialize tree from result
        self.tree = ConversationTree(result, name=self._experiment.name)
        self.tree.set_experiment(self._experiment)
        self._current_branch_id = self.tree.root_id
        
        return result
    
    def fork_and_continue(
        self,
        turn: int,
        new_message: str,
        additional_turns: int = 10,
        branch_id: str = None
    ) -> Branch:
        """
        Fork at a turn and continue with new path.
        
        Args:
            turn: Turn to fork at
            new_message: New message to inject
            additional_turns: Turns to continue
            branch_id: Branch to fork from (default: current)
            
        Returns:
            New branch with completed conversation
        """
        source_id = branch_id or self._current_branch_id
        
        # Create fork
        new_branch = self.tree.fork_at(
            turn=turn,
            new_message=new_message,
            branch_id=source_id
        )
        
        # Continue the branch
        self.tree.continue_branch(new_branch.id, max_turns=additional_turns)
        
        self._current_branch_id = new_branch.id
        
        return new_branch
    
    def explore_alternatives(
        self,
        turn: int,
        alternatives: List[str],
        additional_turns: int = 10
    ) -> List[Branch]:
        """
        Explore multiple alternative paths from same fork point.
        
        Args:
            turn: Turn to fork at
            alternatives: List of alternative messages
            additional_turns: Turns to continue each
            
        Returns:
            List of completed branches
        """
        branches = []
        
        for alt_message in alternatives:
            branch = self.fork_and_continue(
                turn=turn,
                new_message=alt_message,
                additional_turns=additional_turns,
                branch_id=self.tree.root_id  # Always fork from root
            )
            branches.append(branch)
        
        return branches
    
    def compare_outcomes(
        self,
        metric_func: Callable[[List[Dict]], float] = None
    ) -> Dict[str, Any]:
        """
        Compare all branches using a metric.
        
        Args:
            metric_func: Function to compute metric from messages
            
        Returns:
            Comparison results
        """
        return self.tree.compare_branches(
            list(self.tree.branches.keys()),
            metric_func
        )
    
    def get_tree(self) -> ConversationTree:
        """Get the conversation tree."""
        return self.tree
    
    def visualize_tree(self) -> str:
        """Get tree visualization."""
        return self.tree.visualize()


# Convenience functions
def fork_conversation(
    logger,
    turn: int,
    new_message: str,
    experiment=None,
    additional_turns: int = 10
) -> Tuple[ConversationTree, Branch]:
    """
    Quick fork of a conversation.
    
    Args:
        logger: ConversationLogger with original conversation
        turn: Turn to fork at
        new_message: New message to inject
        experiment: Experiment for continuation (optional)
        additional_turns: Turns to add after fork
        
    Returns:
        (ConversationTree, new Branch)
    """
    tree = ConversationTree(logger)
    
    branch = tree.fork_at(turn=turn, new_message=new_message)
    
    if experiment:
        tree.set_experiment(experiment)
        tree.continue_branch(branch.id, max_turns=additional_turns)
    
    return tree, branch


def explore_counterfactuals(
    logger,
    turn: int,
    alternatives: List[str],
    experiment=None,
    additional_turns: int = 10
) -> ConversationTree:
    """
    Explore counterfactual conversation paths.
    
    Args:
        logger: Original conversation
        turn: Turn to branch at
        alternatives: Alternative messages to try
        experiment: Experiment for continuation
        additional_turns: Turns per branch
        
    Returns:
        ConversationTree with all branches
    """
    tree = ConversationTree(logger)
    
    if experiment:
        tree.set_experiment(experiment)
    
    for alt in alternatives:
        branch = tree.fork_at(turn=turn, new_message=alt)
        if experiment:
            tree.continue_branch(branch.id, max_turns=additional_turns)
    
    return tree
