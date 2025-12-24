"""
Configuration management for experiments.

This module handles loading and validating experiment configurations from
YAML files, making it easy for social scientists to set up experiments
without writing Python code.

Example YAML:
    experiment:
      name: "political_polarization_test"
      max_turns: 20
      initial_topic: "climate policy"

    agents:
      - name: "user_persona"
        role: "user"
        provider:
          type: "ollama"
          model: "llama2"
        persona:
          file: "personas/conservative_user.yaml"

      - name: "advisor"
        role: "assistant"
        provider:
          type: "ollama"
          model: "llama2"
        persona:
          name: "Neutral Advisor"
          background: "Balanced AI assistant"
"""

from pathlib import Path
from typing import Dict, Any, List
import yaml
import os
import logging

from synthetic_experiments.providers.ollama import OllamaProvider
from synthetic_experiments.providers.claude import ClaudeProvider
from synthetic_experiments.providers.openai import OpenAIProvider
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.providers.base import GenerationConfig
from synthetic_experiments.experiments.experiment import Experiment, ExperimentConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigurationError: If YAML is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate experiment configuration without creating objects.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Check experiment section
    if "experiment" not in config:
        raise ConfigurationError("Configuration must have 'experiment' section")
    
    exp_config = config["experiment"]
    if not exp_config.get("name"):
        raise ConfigurationError("Experiment must have a 'name'")
    
    # Check agents section
    if "agents" not in config or not config["agents"]:
        raise ConfigurationError("Configuration must have 'agents' section with at least 2 agents")
    
    if len(config["agents"]) < 2:
        raise ConfigurationError("Experiment requires at least 2 agents")
    
    # Validate each agent
    for i, agent_config in enumerate(config["agents"]):
        if "provider" not in agent_config:
            raise ConfigurationError(f"Agent {i+1} must have 'provider' configuration")
        if "persona" not in agent_config:
            raise ConfigurationError(f"Agent {i+1} must have 'persona' configuration")
        
        provider = agent_config["provider"]
        if "type" not in provider:
            raise ConfigurationError(f"Agent {i+1} provider must have 'type'")
        if provider["type"].lower() not in ["ollama", "claude", "openai"]:
            raise ConfigurationError(f"Agent {i+1} provider type must be ollama, claude, or openai")
        if "model" not in provider:
            raise ConfigurationError(f"Agent {i+1} provider must have 'model'")
    
    return True


def create_provider_from_config(provider_config: Dict[str, Any]):
    """
    Create an LLM provider from configuration.

    Args:
        provider_config: Provider configuration dictionary

    Returns:
        LLMProvider instance

    Raises:
        ConfigurationError: If provider config is invalid
    """
    provider_type = provider_config.get("type", "").lower()
    model = provider_config.get("model")

    if not model:
        raise ConfigurationError("Provider 'model' is required")

    try:
        if provider_type == "ollama":
            return OllamaProvider(
                model_name=model,
                base_url=provider_config.get("base_url", "http://localhost:11434"),
                auto_pull=provider_config.get("auto_pull", True)
            )

        elif provider_type == "claude":
            api_key = provider_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            return ClaudeProvider(
                model_name=model,
                api_key=api_key
            )

        elif provider_type == "openai":
            api_key = provider_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            return OpenAIProvider(
                model_name=model,
                api_key=api_key
            )

        else:
            raise ConfigurationError(
                f"Unknown provider type: {provider_type}. "
                f"Must be one of: ollama, claude, openai"
            )

    except Exception as e:
        raise ConfigurationError(f"Failed to create provider: {e}")


def create_persona_from_config(
    persona_config: Dict[str, Any],
    config_dir: Path
) -> Persona:
    """
    Create a persona from configuration.

    Args:
        persona_config: Persona configuration dictionary
        config_dir: Directory containing the config file (for resolving relative paths)

    Returns:
        Persona instance

    Raises:
        ConfigurationError: If persona config is invalid
    """
    try:
        # If persona specified via file
        if "file" in persona_config:
            persona_file = persona_config["file"]

            # Resolve relative paths relative to config file
            if not os.path.isabs(persona_file):
                persona_file = config_dir / persona_file

            return Persona.from_yaml(str(persona_file))

        # Otherwise create from inline config
        else:
            return Persona(**persona_config)

    except Exception as e:
        raise ConfigurationError(f"Failed to create persona: {e}")


def create_agent_from_config(
    agent_config: Dict[str, Any],
    config_dir: Path
) -> ConversationAgent:
    """
    Create a conversation agent from configuration.

    Args:
        agent_config: Agent configuration dictionary
        config_dir: Directory containing the config file

    Returns:
        ConversationAgent instance

    Raises:
        ConfigurationError: If agent config is invalid
    """
    try:
        # Create provider
        if "provider" not in agent_config:
            raise ConfigurationError("Agent must have 'provider' configuration")

        provider = create_provider_from_config(agent_config["provider"])

        # Create persona
        if "persona" not in agent_config:
            raise ConfigurationError("Agent must have 'persona' configuration")

        persona = create_persona_from_config(agent_config["persona"], config_dir)

        # Get role (default to user)
        role = agent_config.get("role", "user")

        # Get name (defaults to persona name)
        name = agent_config.get("name")

        # Create generation config if specified
        generation_config = None
        if "generation" in agent_config:
            gen_cfg = agent_config["generation"]
            generation_config = GenerationConfig(
                temperature=gen_cfg.get("temperature", 0.7),
                max_tokens=gen_cfg.get("max_tokens", 1000),
                top_p=gen_cfg.get("top_p"),
                seed=gen_cfg.get("seed")
            )

        # Create agent
        agent = ConversationAgent(
            provider=provider,
            persona=persona,
            role=role,
            name=name,
            generation_config=generation_config
        )

        return agent

    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Failed to create agent: {e}")


def load_experiment_config(config_path: str) -> Experiment:
    """
    Load a complete experiment from a YAML configuration file.

    This is the main entry point for creating experiments from configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Experiment instance ready to run

    Raises:
        ConfigurationError: If configuration is invalid

    Example:
        >>> experiment = load_experiment_config("config.yaml")
        >>> results = experiment.run()
    """
    config_path = Path(config_path)
    config_dir = config_path.parent

    # Load YAML
    config = load_yaml_config(str(config_path))

    # Extract experiment config
    if "experiment" not in config:
        raise ConfigurationError("Configuration must have 'experiment' section")

    exp_config = config["experiment"]
    experiment_name = exp_config.get("name")

    if not experiment_name:
        raise ConfigurationError("Experiment must have a 'name'")

    # Create experiment configuration
    experiment_config = ExperimentConfig(
        name=experiment_name,
        max_turns=exp_config.get("max_turns", 20),
        initial_topic=exp_config.get("initial_topic", ""),
        topics=exp_config.get("topics", []),
        stop_on_repetition=exp_config.get("stop_on_repetition", False),
        repetition_threshold=exp_config.get("repetition_threshold", 0.9),
        save_conversations=exp_config.get("save_conversations", True),
        output_dir=exp_config.get("output_dir", "results"),
        metadata=exp_config.get("metadata", {}),
        turn_order=exp_config.get("turn_order", "round_robin")
    )

    # Resolve output_dir relative to config file if relative
    if not os.path.isabs(experiment_config.output_dir):
        experiment_config.output_dir = str(config_dir / experiment_config.output_dir)

    # Create agents
    if "agents" not in config or not config["agents"]:
        raise ConfigurationError("Configuration must have 'agents' section with at least 2 agents")

    agents = []
    for i, agent_config in enumerate(config["agents"]):
        try:
            agent = create_agent_from_config(agent_config, config_dir)
            agents.append(agent)
            logger.info(f"Created agent: {agent.name}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create agent {i + 1}: {e}")

    if len(agents) < 2:
        raise ConfigurationError("Experiment requires at least 2 agents")

    # Create experiment
    experiment = Experiment(
        name=experiment_name,
        agents=agents,
        config=experiment_config
    )

    logger.info(f"Loaded experiment '{experiment_name}' from {config_path}")

    return experiment


def save_experiment_config(
    experiment: Experiment,
    output_path: str
) -> None:
    """
    Save an experiment configuration to a YAML file.

    Args:
        experiment: Experiment instance to save
        output_path: Path where YAML file should be saved
    """
    config = {
        "experiment": {
            "name": experiment.name,
            "max_turns": experiment.config.max_turns,
            "initial_topic": experiment.config.initial_topic,
            "topics": experiment.config.topics,
            "output_dir": experiment.config.output_dir,
            "metadata": experiment.config.metadata,
        },
        "agents": []
    }

    for agent in experiment.agents:
        agent_config = {
            "name": agent.name,
            "role": agent.role,
            "provider": {
                "type": agent.provider.__class__.__name__.replace("Provider", "").lower(),
                "model": agent.provider.model_name
            },
            "persona": agent.persona.to_dict()
        }
        config["agents"].append(agent_config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved experiment configuration to {output_path}")
