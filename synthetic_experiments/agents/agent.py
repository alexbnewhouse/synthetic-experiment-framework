"""
Conversation agent that combines an LLM provider with a persona.

A ConversationAgent represents one participant in a conversation, with its own
personality (persona), memory (conversation history), and language model (provider).

Example:
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.agents import Persona, ConversationAgent
    >>>
    >>> provider = OllamaProvider(model_name="llama2")
    >>> persona = Persona(name="Friendly User", background="Curious learner")
    >>> agent = ConversationAgent(provider=provider, persona=persona, role="user")
    >>>
    >>> response = agent.respond("What's your view on renewable energy?")
    >>> print(response.content)
"""

from typing import List, Optional, Dict, Any
from copy import deepcopy
import logging

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    GenerationConfig,
    GenerationResult
)
from synthetic_experiments.agents.persona import Persona

logger = logging.getLogger(__name__)


class ConversationAgent:
    """
    An agent that can participate in conversations.

    A ConversationAgent combines three key components:
    1. Provider: The LLM that generates responses
    2. Persona: The character/personality the agent embodies
    3. History: The conversation memory

    Attributes:
        provider: LLM provider for generating responses
        persona: Persona defining agent characteristics
        role: Agent's role in conversations ('user' or 'assistant')
        name: Display name (defaults to persona name)
        conversation_history: List of messages in conversation
        generation_config: Default generation parameters
    """

    def __init__(
        self,
        provider: LLMProvider,
        persona: Persona,
        role: str = "user",
        name: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize a conversation agent.

        Args:
            provider: LLM provider for generating responses
            persona: Persona defining agent behavior
            role: Agent role - 'user' or 'assistant' (for message formatting)
            name: Optional display name (defaults to persona name)
            generation_config: Default generation configuration
        """
        self.provider = provider
        self.persona = persona
        self.role = role
        self.name = name or persona.name
        self.conversation_history: List[Message] = []
        self.generation_config = generation_config or GenerationConfig()

        # Track statistics
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.response_count = 0

    def respond(
        self,
        prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> Message:
        """
        Generate a response to the current conversation state.

        This method generates a response based on the conversation history
        and the agent's persona. If a prompt is provided, it's added to
        the history as a message from the other participant first.

        Args:
            prompt: Optional new message to respond to
            generation_config: Override default generation config

        Returns:
            Message containing the agent's response

        Raises:
            RuntimeError: If response generation fails
        """
        # Add prompt to history if provided
        if prompt:
            other_role = "assistant" if self.role == "user" else "user"
            self.conversation_history.append(
                Message(role=other_role, content=prompt)
            )

        # Prepare messages for generation
        messages = self._prepare_messages_for_generation()

        # Use provided config or default
        config = generation_config or self.generation_config

        # Generate response
        try:
            result = self.provider.generate(messages, config)

            # Add response to history
            # The role in history is our agent's role
            history_message = Message(
                role=self.role,
                content=result.message.content,
                metadata={
                    "agent_name": self.name,
                    "tokens_used": result.tokens_used,
                    "cost": result.cost,
                }
            )
            self.conversation_history.append(history_message)

            # Update statistics
            if result.tokens_used:
                self.total_tokens_used += result.tokens_used
            if result.cost:
                self.total_cost += result.cost
            self.response_count += 1

            return history_message

        except Exception as e:
            logger.error(f"Agent {self.name} failed to generate response: {e}")
            raise RuntimeError(f"Response generation failed: {e}")

    def _prepare_messages_for_generation(self) -> List[Message]:
        """
        Prepare messages for LLM generation.

        Adds the persona's system prompt and formats conversation history
        appropriately for the LLM provider.

        Returns:
            List of messages ready for LLM generation
        """
        messages = []

        # Add system message with persona
        system_prompt = self.persona.to_system_prompt()
        messages.append(Message(role="system", content=system_prompt))

        # Add conversation history
        messages.extend(self.conversation_history)

        return messages

    def reset_conversation(self) -> None:
        """
        Clear conversation history.

        Useful for starting a new conversation with the same agent configuration.
        Statistics are preserved.
        """
        self.conversation_history = []
        logger.info(f"Agent {self.name} conversation history reset")

    def update_persona(self, persona: Persona) -> None:
        """
        Update the agent's persona.

        This allows dynamic persona changes during experiments, e.g., to
        simulate opinion shifts or role changes.

        Args:
            persona: New persona to adopt
        """
        old_name = self.persona.name
        self.persona = persona
        logger.info(f"Agent persona updated from '{old_name}' to '{persona.name}'")

    def get_history(self) -> List[Message]:
        """
        Get a copy of the conversation history.

        Returns:
            Deep copy of conversation history
        """
        return deepcopy(self.conversation_history)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with usage statistics (tokens, cost, response count)
        """
        return {
            "agent_name": self.name,
            "role": self.role,
            "persona": self.persona.name,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "response_count": self.response_count,
            "messages_in_history": len(self.conversation_history)
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize agent to dictionary.

        Returns:
            Dictionary representation of agent
        """
        return {
            "name": self.name,
            "role": self.role,
            "persona": self.persona.to_dict(),
            "provider": {
                "type": self.provider.__class__.__name__,
                "model": self.provider.model_name
            },
            "history": [msg.to_dict() for msg in self.conversation_history],
            "statistics": self.get_statistics()
        }

    def __repr__(self) -> str:
        return (
            f"ConversationAgent(name='{self.name}', role='{self.role}', "
            f"persona='{self.persona.name}', provider={self.provider})"
        )
