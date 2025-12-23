"""
Persona system for defining agent characteristics and behaviors.

Personas define the background, personality, goals, and communication style
of conversation agents. They can be defined programmatically or loaded from
YAML files for easy configuration by social scientists.

Example:
    >>> from synthetic_experiments.agents import Persona
    >>>
    >>> # Create persona programmatically
    >>> persona = Persona(
    ...     name="Conservative User",
    ...     background="Middle-aged suburban voter",
    ...     political_orientation="conservative",
    ...     communication_style="direct and skeptical"
    ... )
    >>>
    >>> # Or load from YAML file
    >>> persona = Persona.from_yaml("personas/conservative_user.yaml")
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class Persona:
    """
    Defines the characteristics and behavior of a conversation agent.

    Personas are flexible containers for agent attributes. The framework
    provides common fields for social science research, but you can add
    custom fields via the 'attributes' dictionary.

    Attributes:
        name: Display name of the persona
        background: Background description (demographics, context, etc.)
        political_orientation: Political leaning (if relevant to research)
        communication_style: How the persona communicates
        goals: What the persona wants to achieve in conversations
        beliefs: Key beliefs or values held by the persona
        attributes: Additional custom attributes (flexible dictionary)
        system_prompt: Optional explicit system prompt override
    """
    name: str
    background: str = ""
    political_orientation: Optional[str] = None
    communication_style: str = "conversational"
    goals: str = ""
    beliefs: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None

    def to_system_prompt(self) -> str:
        """
        Convert persona to a system prompt for LLMs.

        This generates a natural language description of the persona
        that can be used as a system message to guide LLM behavior.

        Returns:
            System prompt string
        """
        if self.system_prompt:
            return self.system_prompt

        prompt_parts = [f"You are {self.name}."]

        if self.background:
            prompt_parts.append(f"Background: {self.background}")

        if self.political_orientation:
            prompt_parts.append(
                f"Political orientation: {self.political_orientation}"
            )

        if self.communication_style:
            prompt_parts.append(
                f"Communication style: {self.communication_style}"
            )

        if self.goals:
            prompt_parts.append(f"Your goals: {self.goals}")

        if self.beliefs:
            beliefs_str = ", ".join([f"{k}: {v}" for k, v in self.beliefs.items()])
            prompt_parts.append(f"Your beliefs: {beliefs_str}")

        # Add any custom attributes
        for key, value in self.attributes.items():
            if isinstance(value, str):
                prompt_parts.append(f"{key}: {value}")

        return " ".join(prompt_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        """Create persona from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, file_path: str) -> "Persona":
        """
        Load persona from YAML file.

        YAML format example:
            name: "Conservative User"
            background: "Middle-aged suburban voter"
            political_orientation: "conservative"
            communication_style: "direct and skeptical"
            goals: "Seek confirmation of existing beliefs"
            beliefs:
              economy: "Free market solutions are best"
              government: "Limited government intervention"

        Args:
            file_path: Path to YAML file

        Returns:
            Persona instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Persona file not found: {file_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid persona file format: {file_path}")

        return cls.from_dict(data)

    def save_yaml(self, file_path: str) -> None:
        """
        Save persona to YAML file.

        Args:
            file_path: Path where YAML file should be saved
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Persona(name='{self.name}', orientation={self.political_orientation})"


class PersonaFactory:
    """
    Factory for creating common persona types.

    This class provides pre-built personas for common research scenarios,
    particularly for political polarization studies.
    """

    @staticmethod
    def create_political_user(
        orientation: str,
        intensity: str = "moderate"
    ) -> Persona:
        """
        Create a political user persona.

        Args:
            orientation: 'liberal', 'conservative', or 'moderate'
            intensity: 'low', 'moderate', or 'high' - how strongly held are beliefs

        Returns:
            Persona configured with political characteristics
        """
        orientations = {
            "liberal": {
                "background": "Urban professional, college-educated, values social progress",
                "beliefs": {
                    "climate": "Climate change requires immediate government action",
                    "healthcare": "Universal healthcare should be a right",
                    "equality": "Systemic inequalities must be addressed",
                },
                "style": {
                    "low": "open-minded and curious",
                    "moderate": "progressive but willing to discuss",
                    "high": "passionate and advocacy-oriented"
                }
            },
            "conservative": {
                "background": "Suburban or rural, values tradition and stability",
                "beliefs": {
                    "economy": "Free market solutions work best",
                    "government": "Limited government is better government",
                    "tradition": "Traditional values are important",
                },
                "style": {
                    "low": "pragmatic and open to discussion",
                    "moderate": "principled but respectful",
                    "high": "strong convictions, skeptical of change"
                }
            },
            "moderate": {
                "background": "Centrist, considers multiple perspectives",
                "beliefs": {
                    "approach": "Most issues require balanced solutions",
                    "compromise": "Finding common ground is important",
                },
                "style": {
                    "low": "neutral and analytical",
                    "moderate": "balanced and thoughtful",
                    "high": "actively seeks middle ground"
                }
            }
        }

        if orientation not in orientations:
            raise ValueError(
                f"Invalid orientation: {orientation}. "
                f"Must be one of: {list(orientations.keys())}"
            )

        config = orientations[orientation]
        style = config.get("style", {}).get(intensity, "conversational")

        return Persona(
            name=f"{orientation.capitalize()} User ({intensity} intensity)",
            background=config["background"],
            political_orientation=orientation,
            communication_style=style,
            goals=f"Engage in political discussions from a {orientation} perspective",
            beliefs=config.get("beliefs", {})
        )

    @staticmethod
    def create_neutral_advisor() -> Persona:
        """
        Create a neutral advisor persona (for the assistant role).

        Returns:
            Persona configured as a neutral, helpful advisor
        """
        return Persona(
            name="Neutral Advisor",
            background="AI assistant trained to provide balanced, helpful information",
            political_orientation=None,
            communication_style="neutral, informative, and balanced",
            goals="Help users explore topics and perspectives thoughtfully",
            beliefs={
                "approach": "Present multiple perspectives fairly",
                "neutrality": "Avoid taking political sides",
            }
        )

    @staticmethod
    def create_empathetic_advisor() -> Persona:
        """
        Create an empathetic advisor persona.

        Returns:
            Persona configured as an empathetic, supportive advisor
        """
        return Persona(
            name="Empathetic Advisor",
            background="AI assistant focused on understanding and validation",
            communication_style="warm, empathetic, and validating",
            goals="Understand user perspectives and provide supportive guidance",
            beliefs={
                "approach": "Validate user feelings while offering perspective",
                "empathy": "Understanding comes before advice",
            }
        )

    @staticmethod
    def create_challenging_advisor() -> Persona:
        """
        Create a challenging advisor persona (Socratic approach).

        Returns:
            Persona configured to ask probing questions
        """
        return Persona(
            name="Challenging Advisor",
            background="AI assistant using Socratic method to deepen thinking",
            communication_style="questioning, thought-provoking, devil's advocate",
            goals="Challenge assumptions and encourage critical thinking",
            beliefs={
                "approach": "Ask probing questions to test reasoning",
                "method": "Play devil's advocate to strengthen arguments",
            }
        )
