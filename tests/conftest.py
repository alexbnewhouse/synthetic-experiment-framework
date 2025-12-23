"""
Pytest configuration and shared fixtures for synthetic_experiments tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, Mock

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    ModelInfo,
    GenerationConfig,
    GenerationResult
)
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.data.logger import ConversationLogger


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing without real API calls.
    """
    
    def __init__(
        self,
        model_name: str = "mock-model",
        responses: list = None,
        **config
    ):
        self._responses = responses or ["This is a mock response."]
        self._response_index = 0
        self._call_count = 0
        super().__init__(model_name, **config)
    
    def validate_config(self) -> None:
        """Mock validation always passes."""
        pass
    
    def generate(
        self,
        messages: list,
        generation_config: GenerationConfig = None
    ) -> GenerationResult:
        """Generate a mock response."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        self._call_count += 1
        
        # Cycle through responses
        response_text = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1
        
        return GenerationResult(
            message=Message(
                role="assistant",
                content=response_text
            ),
            tokens_used=len(response_text.split()) * 2,  # Rough estimate
            finish_reason="stop",
            cost=0.001
        )
    
    def get_model_info(self) -> ModelInfo:
        """Return mock model info."""
        return ModelInfo(
            provider="mock",
            model_name=self.model_name,
            context_window=4096,
            supports_streaming=False
        )
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    def reset(self):
        """Reset the mock provider state."""
        self._response_index = 0
        self._call_count = 0


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_provider_with_responses():
    """Factory fixture to create mock provider with custom responses."""
    def _create(responses: list):
        return MockLLMProvider(responses=responses)
    return _create


@pytest.fixture
def sample_persona():
    """Create a sample persona for testing."""
    return Persona(
        name="Test User",
        background="A test persona for unit testing",
        political_orientation="moderate",
        communication_style="analytical and precise",
        goals="Provide predictable responses for testing",
        beliefs={
            "testing": "Unit tests are important",
            "quality": "Code quality matters"
        },
        attributes={
            "test_attribute": "test_value"
        }
    )


@pytest.fixture
def sample_assistant_persona():
    """Create a sample assistant persona."""
    return Persona(
        name="Test Assistant",
        background="A helpful testing assistant",
        communication_style="helpful and concise",
        goals="Assist with testing"
    )


@pytest.fixture
def sample_agent(mock_provider, sample_persona):
    """Create a sample conversation agent."""
    return ConversationAgent(
        provider=mock_provider,
        persona=sample_persona,
        role="user",
        name="Test Agent"
    )


@pytest.fixture
def sample_assistant_agent(mock_provider, sample_assistant_persona):
    """Create a sample assistant agent."""
    return ConversationAgent(
        provider=mock_provider,
        persona=sample_assistant_persona,
        role="assistant",
        name="Test Assistant"
    )


@pytest.fixture
def sample_message():
    """Create a sample message."""
    return Message(
        role="user",
        content="Hello, this is a test message.",
        metadata={"test": True}
    )


@pytest.fixture
def sample_conversation_logger():
    """Create a sample conversation logger with some turns."""
    logger = ConversationLogger(
        experiment_name="test_experiment",
        conversation_id="test_conv_001",
        metadata={"test": True}
    )
    
    # Add some sample turns
    logger.log_turn(
        agent_name="User",
        role="user",
        message="Hello, how are you?",
        tokens_used=10,
        cost=0.001
    )
    
    logger.log_turn(
        agent_name="Assistant",
        role="assistant",
        message="I'm doing well, thank you! How can I help you today?",
        tokens_used=15,
        cost=0.002
    )
    
    logger.log_turn(
        agent_name="User",
        role="user",
        message="I'd like to discuss climate change.",
        tokens_used=12,
        cost=0.001
    )
    
    logger.log_turn(
        agent_name="Assistant",
        role="assistant",
        message="Climate change is an important topic. What specific aspects would you like to explore?",
        tokens_used=20,
        cost=0.002
    )
    
    return logger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file-based tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_yaml_persona_content():
    """Sample YAML content for persona file."""
    return """
name: "YAML Test User"
background: "A persona loaded from YAML for testing"
political_orientation: "liberal"
communication_style: "friendly and engaging"
goals: "Test YAML loading functionality"
beliefs:
  environment: "Sustainability is important"
  technology: "AI should be ethical"
attributes:
  source: "yaml_file"
"""


@pytest.fixture
def sample_config_yaml():
    """Sample experiment configuration YAML."""
    return """
experiment:
  name: "test_experiment"
  max_turns: 5
  initial_topic: "test topic"
  output_dir: "test_results"

agents:
  - name: "test_user"
    role: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Test User"
      background: "Test background"
      communication_style: "direct"

  - name: "test_assistant"
    role: "assistant"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Test Assistant"
      background: "Test assistant background"
"""


@pytest.fixture
def generation_config():
    """Create a sample generation config."""
    return GenerationConfig(
        temperature=0.7,
        max_tokens=500,
        top_p=0.9,
        stop_sequences=["END"],
        seed=42
    )


@pytest.fixture
def political_text_liberal():
    """Sample liberal-leaning political text."""
    return """
    Climate change requires immediate government action. We need to invest in
    renewable energy and sustainability initiatives. Universal healthcare should
    be a right for all citizens. Social justice and equality are fundamental values.
    We must address systemic inequalities through progressive policies.
    """


@pytest.fixture
def political_text_conservative():
    """Sample conservative-leaning political text."""
    return """
    The free market is the best way to grow our economy. Limited government
    allows for more personal responsibility and freedom. We need strong borders
    and national security. Traditional values and family values are the
    foundation of a strong society. Religious freedom must be protected.
    """


@pytest.fixture
def political_text_neutral():
    """Sample neutral political text."""
    return """
    There are many perspectives on this issue. Some people believe in one
    approach while others prefer different solutions. It's important to
    consider multiple viewpoints and find common ground. Both sides make
    valid points worth considering.
    """
