"""
Tests for the ConversationAgent class.
"""

import pytest
from copy import deepcopy

from synthetic_experiments.agents.agent import ConversationAgent
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.providers.base import (
    Message,
    GenerationConfig,
    GenerationResult
)


class TestConversationAgentInit:
    """Tests for ConversationAgent initialization."""

    def test_agent_creation_minimal(self, mock_provider, sample_persona):
        """Test creating an agent with minimal parameters."""
        agent = ConversationAgent(
            provider=mock_provider,
            persona=sample_persona
        )
        
        assert agent.provider == mock_provider
        assert agent.persona == sample_persona
        assert agent.role == "user"
        assert agent.name == sample_persona.name
        assert agent.conversation_history == []
        assert agent.total_tokens_used == 0
        assert agent.total_cost == 0.0
        assert agent.response_count == 0

    def test_agent_creation_full(self, mock_provider, sample_persona, generation_config):
        """Test creating an agent with all parameters."""
        agent = ConversationAgent(
            provider=mock_provider,
            persona=sample_persona,
            role="assistant",
            name="Custom Name",
            generation_config=generation_config
        )
        
        assert agent.role == "assistant"
        assert agent.name == "Custom Name"
        assert agent.generation_config == generation_config

    def test_agent_default_name_from_persona(self, mock_provider, sample_persona):
        """Test that agent name defaults to persona name."""
        agent = ConversationAgent(
            provider=mock_provider,
            persona=sample_persona
        )
        
        assert agent.name == sample_persona.name


class TestConversationAgentRespond:
    """Tests for ConversationAgent.respond method."""

    def test_respond_basic(self, sample_agent):
        """Test basic response generation."""
        response = sample_agent.respond("Hello!")
        
        assert isinstance(response, Message)
        assert response.role == "user"  # Agent's role
        assert response.content == "This is a mock response."

    def test_respond_adds_to_history(self, sample_agent):
        """Test that respond adds messages to history."""
        assert len(sample_agent.conversation_history) == 0
        
        sample_agent.respond("First message")
        
        # Should have both the prompt and the response
        assert len(sample_agent.conversation_history) == 2
        
        # First should be the prompt (as assistant since agent is user)
        assert sample_agent.conversation_history[0].role == "assistant"
        assert sample_agent.conversation_history[0].content == "First message"
        
        # Second should be the response
        assert sample_agent.conversation_history[1].role == "user"

    def test_respond_without_prompt(self, sample_agent):
        """Test responding without a new prompt (continuation)."""
        # First add some history
        sample_agent.respond("Initial message")
        initial_history_len = len(sample_agent.conversation_history)
        
        # Respond without prompt
        response = sample_agent.respond()
        
        # Should add only the response
        assert len(sample_agent.conversation_history) == initial_history_len + 1

    def test_respond_updates_statistics(self, sample_agent):
        """Test that respond updates agent statistics."""
        assert sample_agent.response_count == 0
        assert sample_agent.total_tokens_used == 0
        
        sample_agent.respond("Test message")
        
        assert sample_agent.response_count == 1
        assert sample_agent.total_tokens_used > 0

    def test_respond_with_custom_config(self, sample_agent):
        """Test responding with custom generation config."""
        config = GenerationConfig(temperature=0.1, max_tokens=50)
        
        response = sample_agent.respond("Test", generation_config=config)
        
        assert response is not None

    def test_respond_message_metadata(self, sample_agent):
        """Test that response message includes metadata."""
        response = sample_agent.respond("Test")
        
        assert "agent_name" in response.metadata
        assert response.metadata["agent_name"] == sample_agent.name

    def test_respond_multiple_turns(self, mock_provider_with_responses):
        """Test multiple conversation turns."""
        responses = ["Response 1", "Response 2", "Response 3"]
        provider = mock_provider_with_responses(responses)
        persona = Persona(name="Multi-turn Agent")
        agent = ConversationAgent(provider=provider, persona=persona)
        
        for i, expected in enumerate(responses):
            response = agent.respond(f"Message {i}")
            assert response.content == expected
        
        # History should have 6 messages (3 prompts + 3 responses)
        assert len(agent.conversation_history) == 6


class TestConversationAgentHistory:
    """Tests for conversation history management."""

    def test_get_history_returns_copy(self, sample_agent):
        """Test that get_history returns a copy, not the original."""
        sample_agent.respond("Test")
        
        history = sample_agent.get_history()
        original_len = len(history)
        
        # Modify the returned history
        history.append(Message(role="user", content="Extra"))
        
        # Original should be unchanged
        assert len(sample_agent.conversation_history) == original_len

    def test_reset_conversation(self, sample_agent):
        """Test resetting conversation history."""
        sample_agent.respond("Message 1")
        sample_agent.respond("Message 2")
        
        assert len(sample_agent.conversation_history) > 0
        
        sample_agent.reset_conversation()
        
        assert len(sample_agent.conversation_history) == 0

    def test_reset_preserves_statistics(self, sample_agent):
        """Test that reset preserves usage statistics."""
        sample_agent.respond("Test")
        
        tokens_before = sample_agent.total_tokens_used
        count_before = sample_agent.response_count
        
        sample_agent.reset_conversation()
        
        assert sample_agent.total_tokens_used == tokens_before
        assert sample_agent.response_count == count_before


class TestConversationAgentPersona:
    """Tests for persona management."""

    def test_update_persona(self, sample_agent):
        """Test updating agent's persona."""
        new_persona = Persona(
            name="Updated User",
            background="New background"
        )
        
        old_name = sample_agent.persona.name
        sample_agent.update_persona(new_persona)
        
        assert sample_agent.persona.name == "Updated User"
        assert sample_agent.persona.name != old_name

    def test_persona_affects_system_prompt(self, mock_provider):
        """Test that persona is used in system prompt for generation."""
        persona = Persona(
            name="Specific User",
            background="Very specific background for testing"
        )
        
        agent = ConversationAgent(provider=mock_provider, persona=persona)
        
        # The system prompt should include persona info
        messages = agent._prepare_messages_for_generation()
        
        assert len(messages) >= 1
        assert messages[0].role == "system"
        assert "Specific User" in messages[0].content


class TestConversationAgentStatistics:
    """Tests for agent statistics tracking."""

    def test_get_statistics(self, sample_agent):
        """Test getting agent statistics."""
        sample_agent.respond("Test 1")
        sample_agent.respond("Test 2")
        
        stats = sample_agent.get_statistics()
        
        assert stats["agent_name"] == sample_agent.name
        assert stats["role"] == sample_agent.role
        assert stats["response_count"] == 2
        assert stats["total_tokens_used"] > 0
        assert stats["messages_in_history"] == 4  # 2 prompts + 2 responses

    def test_to_dict(self, sample_agent):
        """Test serializing agent to dictionary."""
        sample_agent.respond("Test")
        
        result = sample_agent.to_dict()
        
        assert result["name"] == sample_agent.name
        assert result["role"] == sample_agent.role
        assert "persona" in result
        assert "provider" in result
        assert "history" in result
        assert "statistics" in result


class TestConversationAgentRoles:
    """Tests for different agent roles."""

    def test_user_role(self, mock_provider, sample_persona):
        """Test agent with user role."""
        agent = ConversationAgent(
            provider=mock_provider,
            persona=sample_persona,
            role="user"
        )
        
        response = agent.respond("Hello")
        
        # Response should have agent's role (user)
        assert response.role == "user"
        
        # Prompt should be added as opposite role (assistant)
        assert agent.conversation_history[0].role == "assistant"

    def test_assistant_role(self, mock_provider, sample_persona):
        """Test agent with assistant role."""
        agent = ConversationAgent(
            provider=mock_provider,
            persona=sample_persona,
            role="assistant"
        )
        
        response = agent.respond("Hello")
        
        # Response should have agent's role (assistant)
        assert response.role == "assistant"
        
        # Prompt should be added as opposite role (user)
        assert agent.conversation_history[0].role == "user"


class TestConversationAgentPrepareMessages:
    """Tests for message preparation for LLM."""

    def test_prepare_messages_includes_system(self, sample_agent):
        """Test that prepared messages include system prompt."""
        messages = sample_agent._prepare_messages_for_generation()
        
        assert len(messages) >= 1
        assert messages[0].role == "system"

    def test_prepare_messages_includes_history(self, sample_agent):
        """Test that prepared messages include conversation history."""
        sample_agent.respond("First message")
        
        messages = sample_agent._prepare_messages_for_generation()
        
        # Should have system + history
        assert len(messages) >= 3  # system + prompt + response

    def test_prepare_messages_order(self, sample_agent):
        """Test that messages are in correct order."""
        sample_agent.respond("Message 1")
        
        messages = sample_agent._prepare_messages_for_generation()
        
        # First should be system
        assert messages[0].role == "system"
        
        # Rest should be conversation history in order
        for i, hist_msg in enumerate(sample_agent.conversation_history):
            assert messages[i + 1].role == hist_msg.role


class TestConversationAgentErrorHandling:
    """Tests for error handling in agent."""

    def test_respond_provider_error(self, mock_provider, sample_persona):
        """Test handling of provider errors."""
        # Make provider raise an error
        mock_provider.generate = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Provider error")
        )
        
        agent = ConversationAgent(provider=mock_provider, persona=sample_persona)
        
        with pytest.raises(RuntimeError, match="Response generation failed"):
            agent.respond("Test")


class TestConversationAgentRepr:
    """Tests for string representation."""

    def test_repr(self, sample_agent):
        """Test agent string representation."""
        result = repr(sample_agent)
        
        assert "ConversationAgent" in result
        assert sample_agent.name in result
        assert sample_agent.role in result
