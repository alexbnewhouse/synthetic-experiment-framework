"""Integration-like tests for survey flow with context.

These tests use a dummy provider to verify that:
- Pre-survey uses a fresh provider instance.
- Post-survey uses the existing agent provider and includes conversation history
  in the messages passed to the provider.
"""

from typing import List, Optional

from synthetic_experiments.analysis.survey import SurveyAdministrator, calculate_polarization_delta
from synthetic_experiments.agents import ConversationAgent, Persona
from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    GenerationConfig,
    GenerationResult,
)


class DummyProvider(LLMProvider):
    """A minimal provider that returns a fixed numeric response and records calls."""

    instances_created = 0

    def __init__(self, model_name: str = "dummy-model", **config):
        super().__init__(model_name=model_name, **config)
        DummyProvider.instances_created += 1
        self.last_messages: List[Message] = []

    def validate_config(self) -> None:
        # No-op for dummy provider
        return None

    def get_model_info(self):
        from synthetic_experiments.providers.base import ModelInfo

        return ModelInfo(provider="dummy", model_name=self.model_name)

    def generate(
        self,
        messages: List[Message],
        generation_config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        self.last_messages = list(messages)
        # Always return the neutral midpoint (4) to keep scoring simple.
        return GenerationResult(message=Message(role="assistant", content="4"))


class TestSurveyIntegration:
    """Verifies pre/post survey flow with context handling."""

    def setup_method(self):
        DummyProvider.instances_created = 0

    def test_pre_and_post_survey_flow_uses_context(self):
        persona = Persona(name="Advisor", background="", communication_style="", beliefs={})

        admin = SurveyAdministrator(
            provider_class=DummyProvider,
            provider_kwargs={},
            persona=persona,
            seed=123,
            survey="default",
        )

        # Pre-survey should create a fresh provider
        pre_results = admin.administer_pre_survey()
        assert DummyProvider.instances_created == 1
        assert pre_results.survey_type == "pre"

        # Build an advisor agent with its own provider and a conversation history
        advisor_provider = DummyProvider()
        advisor_agent = ConversationAgent(provider=advisor_provider, persona=persona, role="assistant")
        advisor_agent.conversation_history.append(Message(role="user", content="Hello"))
        advisor_agent.conversation_history.append(Message(role="assistant", content="Hi there"))

        post_results = admin.administer_post_survey(advisor_agent)

        # Post-survey should reuse the existing advisor provider (no new instance)
        assert DummyProvider.instances_created == 2  # first for pre, second for advisor_agent
        assert post_results.survey_type == "post"

        # Ensure conversation history was included in the provider call
        assert len(advisor_provider.last_messages) >= 3  # system + history + question
        roles = [m.role for m in advisor_provider.last_messages]
        assert roles.count("system") == 1
        assert roles.count("user") >= 1  # the survey question is user role

        # Delta should be computable even with dummy responses
        delta = calculate_polarization_delta(pre_results, post_results)
        assert delta.overall_delta == post_results.overall_score - pre_results.overall_score
