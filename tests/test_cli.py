"""
Tests for the CLI module.
"""

import pytest
from pathlib import Path
from click.testing import CliRunner

from synthetic_experiments.cli.main import cli
from synthetic_experiments.experiments.config import validate_config, load_yaml_config


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample experiment config file."""
        config_content = """
experiment:
  name: "test_experiment"
  max_turns: 5
  initial_topic: "testing"
  output_dir: "results"

agents:
  - name: "test_user"
    role: "user"
    provider:
      type: "ollama"
      model: "llama3.2"
    persona:
      name: "Test User"
      background: "A test user persona"
      traits:
        - "curious"
  
  - name: "test_assistant"
    role: "assistant"
    provider:
      type: "ollama"
      model: "llama3.2"
    persona:
      name: "Test Assistant"
      background: "A test assistant persona"
"""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text(config_content)
        return config_path

    @pytest.fixture
    def sample_persona(self, temp_dir):
        """Create a sample persona file."""
        persona_content = """
name: "Sample Persona"
background: "A sample background"
communication_style: "friendly"
beliefs:
  test_belief: "value"
"""
        persona_path = temp_dir / "sample_persona.yaml"
        persona_path.write_text(persona_content)
        return persona_path

    def test_cli_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'version' in result.output.lower() or '0.' in result.output

    def test_cli_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'run' in result.output
        assert 'validate-config' in result.output

    def test_validate_config_valid(self, runner, sample_config):
        """Test validate-config with valid config."""
        result = runner.invoke(cli, ['validate-config', str(sample_config)])
        assert result.exit_code == 0
        assert 'valid' in result.output.lower() or 'Configuration valid' in result.output

    def test_validate_config_missing_file(self, runner, temp_dir):
        """Test validate-config with missing file."""
        result = runner.invoke(cli, ['validate-config', str(temp_dir / 'nonexistent.yaml')])
        assert result.exit_code != 0

    def test_validate_persona_valid(self, runner, sample_persona):
        """Test validate-persona with valid persona."""
        result = runner.invoke(cli, ['validate-persona', str(sample_persona)])
        assert result.exit_code == 0

    def test_validate_persona_missing_file(self, runner, temp_dir):
        """Test validate-persona with missing file."""
        result = runner.invoke(cli, ['validate-persona', str(temp_dir / 'nonexistent.yaml')])
        assert result.exit_code != 0

    def test_list_surveys(self, runner):
        """Test list-surveys command."""
        result = runner.invoke(cli, ['list-surveys'])
        assert result.exit_code == 0
        assert 'bail_2018' in result.output.lower() or 'Available' in result.output

    def test_init_command(self, runner, temp_dir):
        """Test init command creates files."""
        import os
        
        # Change to temp dir before running init
        original_dir = os.getcwd()
        try:
            os.chdir(str(temp_dir))
            result = runner.invoke(cli, ['init'])
            
            assert result.exit_code == 0
            
            # Check files were created
            assert (temp_dir / 'config.yaml').exists()
            assert (temp_dir / 'personas').is_dir()
        finally:
            os.chdir(original_dir)

    def test_run_command_missing_config(self, runner):
        """Test run command without config."""
        result = runner.invoke(cli, ['run'])
        assert result.exit_code != 0  # Should fail without config


class TestValidateConfigFunction:
    """Tests for the validate_config function."""

    def test_validate_valid_config(self, temp_dir):
        """Test validation with valid config."""
        config_content = """
experiment:
  name: "test"
  max_turns: 10

agents:
  - name: "user"
    role: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Test"
      background: "Test background"
  - name: "assistant"
    role: "assistant"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Assistant"
      background: "Assistant background"
"""
        config_path = temp_dir / "valid_config.yaml"
        config_path.write_text(config_content)
        
        # Should not raise
        config = load_yaml_config(str(config_path))
        result = validate_config(config)
        assert result is True

    def test_validate_missing_experiment(self, temp_dir):
        """Test validation with missing experiment section."""
        config_content = """
agents:
  - name: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Test"
"""
        config_path = temp_dir / "invalid_config.yaml"
        config_path.write_text(config_content)
        
        config = load_yaml_config(str(config_path))
        with pytest.raises(Exception):  # ConfigurationError
            validate_config(config)

    def test_validate_single_agent(self, temp_dir):
        """Test validation with only one agent."""
        config_content = """
experiment:
  name: "test"

agents:
  - name: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Test"
      background: "Test"
"""
        config_path = temp_dir / "single_agent.yaml"
        config_path.write_text(config_content)
        
        config = load_yaml_config(str(config_path))
        with pytest.raises(Exception):  # ConfigurationError
            validate_config(config)


class TestCLIIntegration:
    """Integration tests for CLI (may require actual providers)."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.mark.skip(reason="Requires Ollama running")
    def test_run_experiment_integration(self, runner, sample_config, temp_dir):
        """Test running an actual experiment."""
        result = runner.invoke(cli, [
            'run',
            str(sample_config),
            '--max-turns', '2',
            '--output-dir', str(temp_dir / 'output')
        ])
        
        # Check output files created
        output_dir = temp_dir / 'output'
        if result.exit_code == 0:
            assert output_dir.exists()
