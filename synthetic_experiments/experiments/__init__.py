"""
Experiment orchestration and configuration.

This module provides tools for setting up and running multi-agent conversation
experiments, including configuration management and experimental control.

Example:
    >>> from synthetic_experiments.experiments import Experiment, load_experiment_config
    >>>
    >>> # Load from config file
    >>> experiment = load_experiment_config("config.yaml")
    >>> results = experiment.run()
    >>>
    >>> # Or create programmatically
    >>> experiment = Experiment(name="my_experiment", agents=[agent1, agent2])
    >>> results = experiment.run(max_turns=20)
"""

from synthetic_experiments.experiments.experiment import Experiment, ExperimentConfig
from synthetic_experiments.experiments.config import (
    load_experiment_config,
    save_experiment_config,
    ConfigurationError
)

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
    "ConfigurationError",
]
