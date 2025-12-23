"""
Tests for the Persona class and PersonaFactory.
"""

import pytest
import tempfile
from pathlib import Path

from synthetic_experiments.agents.persona import Persona, PersonaFactory


class TestPersonaBasic:
    """Tests for basic Persona functionality."""

    def test_persona_creation_minimal(self):
        """Test creating a persona with minimal parameters."""
        persona = Persona(name="Test User")
        
        assert persona.name == "Test User"
        assert persona.background == ""
        assert persona.political_orientation is None
        assert persona.communication_style == "conversational"
        assert persona.goals == ""
        assert persona.beliefs == {}
        assert persona.attributes == {}
        assert persona.system_prompt is None

    def test_persona_creation_full(self, sample_persona):
        """Test creating a persona with all parameters."""
        assert sample_persona.name == "Test User"
        assert sample_persona.background == "A test persona for unit testing"
        assert sample_persona.political_orientation == "moderate"
        assert sample_persona.communication_style == "analytical and precise"
        assert sample_persona.goals == "Provide predictable responses for testing"
        assert sample_persona.beliefs == {
            "testing": "Unit tests are important",
            "quality": "Code quality matters"
        }
        assert sample_persona.attributes == {"test_attribute": "test_value"}

    def test_persona_to_dict(self, sample_persona):
        """Test serializing persona to dictionary."""
        result = sample_persona.to_dict()
        
        assert result["name"] == "Test User"
        assert result["background"] == "A test persona for unit testing"
        assert result["political_orientation"] == "moderate"
        assert "beliefs" in result
        assert "attributes" in result

    def test_persona_from_dict(self):
        """Test creating persona from dictionary."""
        data = {
            "name": "Dict User",
            "background": "Created from dict",
            "political_orientation": "liberal",
            "communication_style": "friendly"
        }
        
        persona = Persona.from_dict(data)
        
        assert persona.name == "Dict User"
        assert persona.background == "Created from dict"
        assert persona.political_orientation == "liberal"

    def test_persona_repr(self, sample_persona):
        """Test persona string representation."""
        result = repr(sample_persona)
        
        assert "Test User" in result
        assert "moderate" in result


class TestPersonaSystemPrompt:
    """Tests for Persona.to_system_prompt method."""

    def test_system_prompt_minimal(self):
        """Test system prompt with minimal persona."""
        persona = Persona(name="Simple User")
        prompt = persona.to_system_prompt()
        
        assert "You are Simple User" in prompt

    def test_system_prompt_full(self, sample_persona):
        """Test system prompt with full persona."""
        prompt = sample_persona.to_system_prompt()
        
        assert "You are Test User" in prompt
        assert "Background:" in prompt
        assert "Political orientation:" in prompt
        assert "Communication style:" in prompt
        assert "Your goals:" in prompt
        assert "Your beliefs:" in prompt

    def test_system_prompt_custom_override(self):
        """Test that custom system_prompt overrides generated one."""
        custom_prompt = "You are a custom assistant with specific behaviors."
        persona = Persona(
            name="Custom User",
            background="Some background",
            system_prompt=custom_prompt
        )
        
        result = persona.to_system_prompt()
        
        assert result == custom_prompt
        assert "Background:" not in result

    def test_system_prompt_includes_beliefs(self):
        """Test that beliefs are included in system prompt."""
        persona = Persona(
            name="Belief User",
            beliefs={
                "environment": "Climate change is real",
                "economy": "Fair taxation is important"
            }
        )
        
        prompt = persona.to_system_prompt()
        
        assert "Your beliefs:" in prompt
        assert "environment" in prompt or "Climate change" in prompt

    def test_system_prompt_includes_attributes(self):
        """Test that string attributes are included."""
        persona = Persona(
            name="Attr User",
            attributes={
                "specialty": "Data science",
                "numeric_attr": 42  # Non-string, should be skipped
            }
        )
        
        prompt = persona.to_system_prompt()
        
        assert "specialty:" in prompt
        assert "Data science" in prompt


class TestPersonaYAML:
    """Tests for YAML serialization/deserialization."""

    def test_save_yaml(self, sample_persona, temp_dir):
        """Test saving persona to YAML file."""
        filepath = temp_dir / "test_persona.yaml"
        sample_persona.save_yaml(str(filepath))
        
        assert filepath.exists()
        
        # Verify content
        import yaml
        with open(filepath) as f:
            data = yaml.safe_load(f)
        
        assert data["name"] == "Test User"
        assert data["political_orientation"] == "moderate"

    def test_load_yaml(self, temp_dir, sample_yaml_persona_content):
        """Test loading persona from YAML file."""
        filepath = temp_dir / "persona.yaml"
        with open(filepath, 'w') as f:
            f.write(sample_yaml_persona_content)
        
        persona = Persona.from_yaml(str(filepath))
        
        assert persona.name == "YAML Test User"
        assert persona.political_orientation == "liberal"
        assert persona.beliefs["environment"] == "Sustainability is important"

    def test_load_yaml_file_not_found(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Persona.from_yaml("/nonexistent/path/persona.yaml")

    def test_load_yaml_invalid_format(self, temp_dir):
        """Test that invalid YAML format raises error."""
        filepath = temp_dir / "invalid.yaml"
        with open(filepath, 'w') as f:
            f.write("just a string, not a dict")
        
        with pytest.raises(ValueError, match="Invalid persona file format"):
            Persona.from_yaml(str(filepath))

    def test_round_trip_yaml(self, sample_persona, temp_dir):
        """Test saving and loading produces equivalent persona."""
        filepath = temp_dir / "roundtrip.yaml"
        
        sample_persona.save_yaml(str(filepath))
        loaded = Persona.from_yaml(str(filepath))
        
        assert loaded.name == sample_persona.name
        assert loaded.background == sample_persona.background
        assert loaded.political_orientation == sample_persona.political_orientation
        assert loaded.beliefs == sample_persona.beliefs


class TestPersonaFactory:
    """Tests for PersonaFactory class."""

    def test_create_liberal_user(self):
        """Test creating a liberal user persona."""
        persona = PersonaFactory.create_political_user("liberal", "moderate")
        
        assert "liberal" in persona.name.lower() or "Liberal" in persona.name
        assert persona.political_orientation == "liberal"
        assert len(persona.beliefs) > 0

    def test_create_conservative_user(self):
        """Test creating a conservative user persona."""
        persona = PersonaFactory.create_political_user("conservative", "moderate")
        
        assert "conservative" in persona.name.lower() or "Conservative" in persona.name
        assert persona.political_orientation == "conservative"
        assert len(persona.beliefs) > 0

    def test_create_moderate_user(self):
        """Test creating a moderate user persona."""
        persona = PersonaFactory.create_political_user("moderate", "moderate")
        
        assert "moderate" in persona.name.lower() or "Moderate" in persona.name
        assert persona.political_orientation == "moderate"

    def test_create_political_user_invalid_orientation(self):
        """Test that invalid orientation raises error."""
        with pytest.raises(ValueError, match="Invalid orientation"):
            PersonaFactory.create_political_user("anarchist", "moderate")

    def test_create_political_user_intensity_levels(self):
        """Test different intensity levels."""
        for intensity in ["low", "moderate", "high"]:
            persona = PersonaFactory.create_political_user("liberal", intensity)
            assert persona is not None
            assert intensity in persona.name.lower() or intensity in persona.communication_style.lower()

    def test_create_neutral_advisor(self):
        """Test creating a neutral advisor."""
        persona = PersonaFactory.create_neutral_advisor()
        
        assert "Neutral" in persona.name
        assert persona.political_orientation is None
        assert "neutral" in persona.communication_style.lower()

    def test_create_empathetic_advisor(self):
        """Test creating an empathetic advisor."""
        persona = PersonaFactory.create_empathetic_advisor()
        
        assert "Empathetic" in persona.name
        assert "empathetic" in persona.communication_style.lower() or "warm" in persona.communication_style.lower()

    def test_create_challenging_advisor(self):
        """Test creating a challenging advisor."""
        persona = PersonaFactory.create_challenging_advisor()
        
        assert "Challenging" in persona.name
        assert "question" in persona.communication_style.lower() or "devil" in persona.communication_style.lower()


class TestPersonaComparison:
    """Tests for persona equality and hashing."""

    def test_personas_with_same_name_different_attrs(self):
        """Test that personas with same name but different attrs are different."""
        p1 = Persona(name="User", background="Background 1")
        p2 = Persona(name="User", background="Background 2")
        
        # They should have different backgrounds
        assert p1.background != p2.background

    def test_persona_dict_round_trip(self, sample_persona):
        """Test that dict conversion preserves all data."""
        data = sample_persona.to_dict()
        restored = Persona.from_dict(data)
        
        assert restored.name == sample_persona.name
        assert restored.background == sample_persona.background
        assert restored.political_orientation == sample_persona.political_orientation
        assert restored.communication_style == sample_persona.communication_style
        assert restored.goals == sample_persona.goals
        assert restored.beliefs == sample_persona.beliefs
        assert restored.attributes == sample_persona.attributes
