"""
Data logging and storage for synthetic experiments.

This module provides tools for recording and analyzing experimental data,
with formats optimized for social science research workflows.

Example:
    >>> from synthetic_experiments.data import ConversationLogger, ExperimentStorage
    >>>
    >>> # Log a conversation
    >>> logger = ConversationLogger(experiment_name="political_study")
    >>> logger.log_turn("User", "user", "Hello!")
    >>>
    >>> # Save to storage
    >>> storage = ExperimentStorage(base_dir="results/political_study")
    >>> storage.save_conversation(logger)
    >>> storage.export_summary_csv()
"""

from synthetic_experiments.data.logger import ConversationLogger, ConversationTurn
from synthetic_experiments.data.storage import ExperimentStorage

__all__ = [
    "ConversationLogger",
    "ConversationTurn",
    "ExperimentStorage",
]
