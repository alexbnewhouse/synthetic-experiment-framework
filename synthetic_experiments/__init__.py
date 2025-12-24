"""
Synthetic Experiments Framework
================================

A Python framework for social scientists to conduct experimental research
on conversations with LLM chatbots.

Key Features:
- Modular LLM provider system (Ollama, Claude, OpenAI)
- Flexible persona and prompt management
- Multi-agent conversation orchestration
- Comprehensive data logging and analysis tools
- Cost estimation and parallel execution
- Conversation branching and continuation
- Built for social science research workflows

Basic Usage:
    >>> from synthetic_experiments import Experiment
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.agents import ConversationAgent, Persona
    >>>
    >>> # Create providers and agents
    >>> provider = OllamaProvider(model="llama2")
    >>> persona = Persona(name="User", background="Political moderate")
    >>> agent = ConversationAgent(provider=provider, persona=persona)
    >>>
    >>> # Run an experiment
    >>> experiment = Experiment.from_config("config.yaml")
    >>> results = experiment.run()

Advanced Features:
    >>> # Cost estimation
    >>> from synthetic_experiments.costs import CostEstimator
    >>> estimator = CostEstimator()
    >>> cost = estimator.estimate_experiment(model="gpt-4", num_turns=20)
    >>>
    >>> # Parallel execution
    >>> from synthetic_experiments.parallel import ParallelRunner
    >>> runner = ParallelRunner(max_workers=4)
    >>> results = runner.run_batch(experiments)
    >>>
    >>> # Conversation branching
    >>> from synthetic_experiments.branching import BranchingExperiment
    >>> branching = BranchingExperiment(experiment)
    >>> branching.fork_and_continue(turn=5, new_message="What if...")
"""

__version__ = "0.1.0"
__author__ = "Alex Newhouse"

# Import key classes for easy access
from synthetic_experiments.providers.base import LLMProvider
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.experiments.experiment import Experiment

# Import new modules
from synthetic_experiments.costs import CostEstimator, estimate_cost
from synthetic_experiments.parallel import ParallelRunner, BatchRunner
from synthetic_experiments.comparison import ModelComparator
from synthetic_experiments.streaming import StreamingExperiment, stream_conversation
from synthetic_experiments.continuation import ContinuableExperiment, save_conversation_state, load_conversation_state
from synthetic_experiments.stopping import StoppingConditionManager, create_default_stopping_conditions
from synthetic_experiments.export import ConversationDataExporter, export_to_csv, export_for_analysis
from synthetic_experiments.rate_limiting import RateLimiter, RateLimitedProvider
from synthetic_experiments.branching import ConversationTree, BranchingExperiment, fork_conversation

__all__ = [
    # Core classes
    "LLMProvider",
    "ConversationAgent",
    "Persona",
    "Experiment",
    # Cost estimation
    "CostEstimator",
    "estimate_cost",
    # Parallel execution
    "ParallelRunner",
    "BatchRunner",
    # Model comparison
    "ModelComparator",
    # Streaming
    "StreamingExperiment",
    "stream_conversation",
    # Continuation
    "ContinuableExperiment",
    "save_conversation_state",
    "load_conversation_state",
    # Stopping conditions
    "StoppingConditionManager",
    "create_default_stopping_conditions",
    # Export
    "ConversationDataExporter",
    "export_to_csv",
    "export_for_analysis",
    # Rate limiting
    "RateLimiter",
    "RateLimitedProvider",
    # Branching
    "ConversationTree",
    "BranchingExperiment",
    "fork_conversation",
]
