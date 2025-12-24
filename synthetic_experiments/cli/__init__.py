"""
Command-line interface for the Synthetic Experiments Framework.

Provides the `synthetic-exp` command for running experiments, 
administering surveys, and analyzing results.

Commands:
    synthetic-exp run CONFIG      Run experiment from YAML config
    synthetic-exp survey          Administer standalone survey
    synthetic-exp analyze DIR     Analyze experiment results
    synthetic-exp list-surveys    List available survey instruments
    synthetic-exp init            Initialize new experiment directory
    synthetic-exp validate-config Validate a config file
    synthetic-exp validate-persona Validate a persona file
"""

from synthetic_experiments.cli.main import cli

__all__ = ['cli']
