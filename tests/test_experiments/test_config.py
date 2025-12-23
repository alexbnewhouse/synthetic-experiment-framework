"""
Tests for experiment configuration loading.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from synthetic_experiments.experiments.config import (
    load_yaml_config,
    create_provider_from_config,
    create_persona_from_config,
    create_agent_from_config,
    ConfigurationError
)
from synthetic_experiments.agents.persona import Persona


class TestLoadYamlConfig:
    """Tests for YAML configuration loading."""

    def test_load_valid_yaml(self, temp_dir, sample_config_yaml):
        """Test loading a valid YAML config file."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(sample_config_yaml)
        
        config = load_yaml_config(str(config_path))
        
        assert isinstance(config, dict)
        assert "experiment" in config
        assert "agents" in config

    def test_load_missing_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self, temp_dir):
        """Test error with invalid YAML."""
        config_path = temp_dir / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [[[")
        
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_yaml_config(str(config_path))

    def test_load_non_dict_yaml(self, temp_dir):
        """Test error when YAML is not a dictionary."""
        config_path = temp_dir / "list.yaml"
        with open(config_path, 'w') as f:
            f.write("- item1\n- item2")
        
        with pytest.raises(ConfigurationError, match="must be a dictionary"):
            load_yaml_config(str(config_path))


class TestCreateProviderFromConfig:
    """Tests for provider creation from config."""

    @patch('synthetic_experiments.experiments.config.OllamaProvider')
    def test_create_ollama_provider(self, mock_ollama):
        """Test creating Ollama provider from config."""
        config = {
            "type": "ollama",
            "model": "llama2",
            "base_url": "http://localhost:11434"
        }
        
        provider = create_provider_from_config(config)
        
        mock_ollama.assert_called_once()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch('synthetic_experiments.experiments.config.ClaudeProvider')
    def test_create_claude_provider(self, mock_claude):
        """Test creating Claude provider from config."""
        config = {
            "type": "claude",
            "model": "claude-3-opus-20240229"
        }
        
        provider = create_provider_from_config(config)
        
        mock_claude.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('synthetic_experiments.experiments.config.OpenAIProvider')
    def test_create_openai_provider(self, mock_openai):
        """Test creating OpenAI provider from config."""
        config = {
            "type": "openai",
            "model": "gpt-4"
        }
        
        provider = create_provider_from_config(config)
        
        mock_openai.assert_called_once()

    def test_create_unknown_provider_type(self):
        """Test error for unknown provider type."""
        config = {
            "type": "unknown",
            "model": "some-model"
        }
        
        with pytest.raises(ConfigurationError, match="Unknown provider type"):
            create_provider_from_config(config)

    def test_create_provider_missing_model(self):
        """Test error when model is missing."""
        config = {
            "type": "ollama"
        }
        
        with pytest.raises(ConfigurationError, match="'model' is required"):
            create_provider_from_config(config)


class TestCreatePersonaFromConfig:
    """Tests for persona creation from config."""

    def test_create_persona_inline(self, temp_dir):
        """Test creating persona from inline config."""
        config = {
            "name": "Test User",
            "background": "Test background",
            "political_orientation": "moderate"
        }
        
        persona = create_persona_from_config(config, temp_dir)
        
        assert isinstance(persona, Persona)
        assert persona.name == "Test User"
        assert persona.background == "Test background"

    def test_create_persona_from_file(self, temp_dir, sample_yaml_persona_content):
        """Test creating persona from YAML file."""
        # Create persona file
        personas_dir = temp_dir / "personas"
        personas_dir.mkdir()
        persona_path = personas_dir / "test_persona.yaml"
        with open(persona_path, 'w') as f:
            f.write(sample_yaml_persona_content)
        
        config = {
            "file": "personas/test_persona.yaml"
        }
        
        persona = create_persona_from_config(config, temp_dir)
        
        assert isinstance(persona, Persona)
        assert persona.name == "YAML Test User"

    def test_create_persona_file_not_found(self, temp_dir):
        """Test error when persona file not found."""
        config = {
            "file": "nonexistent.yaml"
        }
        
        with pytest.raises(ConfigurationError, match="Failed to create persona"):
            create_persona_from_config(config, temp_dir)


class TestCreateAgentFromConfig:
    """Tests for agent creation from config."""

    @patch('synthetic_experiments.experiments.config.create_provider_from_config')
    def test_create_agent(self, mock_create_provider, temp_dir):
        """Test creating agent from config."""
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        
        config = {
            "name": "test_agent",
            "role": "user",
            "provider": {
                "type": "ollama",
                "model": "llama2"
            },
            "persona": {
                "name": "Test User",
                "background": "Test"
            }
        }
        
        agent = create_agent_from_config(config, temp_dir)
        
        assert agent is not None
        assert agent.name == "test_agent"
        assert agent.role == "user"

    @patch('synthetic_experiments.experiments.config.create_provider_from_config')
    def test_create_agent_missing_provider(self, mock_create_provider, temp_dir):
        """Test error when provider config missing."""
        config = {
            "name": "test_agent",
            "role": "user",
            "persona": {
                "name": "Test User"
            }
        }
        
        with pytest.raises(ConfigurationError, match="'provider' configuration"):
            create_agent_from_config(config, temp_dir)

    @patch('synthetic_experiments.experiments.config.create_provider_from_config')
    def test_create_agent_missing_persona(self, mock_create_provider, temp_dir):
        """Test error when persona config missing."""
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        
        config = {
            "name": "test_agent",
            "role": "user",
            "provider": {
                "type": "ollama",
                "model": "llama2"
            }
        }
        
        with pytest.raises(ConfigurationError, match="'persona' configuration"):
            create_agent_from_config(config, temp_dir)


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_configuration_error_message(self):
        """Test ConfigurationError message."""
        error = ConfigurationError("Test error message")
        
        assert str(error) == "Test error message"

    def test_configuration_error_is_exception(self):
        """Test that ConfigurationError is an Exception."""
        assert issubclass(ConfigurationError, Exception)
