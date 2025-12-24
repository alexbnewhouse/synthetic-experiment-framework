"""Minimal pre/post survey example using the Bail et al. (2018) instrument.

This script shows how to measure treatment effects where the treatment is the
conversation itself (context window). Requires an LLM provider (e.g., Ollama)
configured locally.
"""

from synthetic_experiments.analysis.survey import SurveyAdministrator, calculate_polarization_delta
from synthetic_experiments.agents import ConversationAgent, Persona
from synthetic_experiments.providers import OllamaProvider


def main():
    # 1) Define advisor persona
    advisor_persona = Persona(
        name="Neutral Advisor",
        background="General advisor",
        communication_style="balanced and calm",
        beliefs={},
    )

    # 2) Create survey administrator using Bail et al. (2018) survey
    admin = SurveyAdministrator(
        provider_class=OllamaProvider,
        provider_kwargs={"model_name": "llama3.2"},
        persona=advisor_persona,
        seed=42,
        survey="bail2018",
    )

    # 3) Pre-survey (fresh LLM - baseline)
    pre_results = admin.administer_pre_survey()
    print(f"Pre-survey ideological score: {pre_results.ideological_score:.3f}")

    # 4) Run a short conversation (treatment = context)
    advisor_agent = ConversationAgent(
        provider=OllamaProvider(model_name="llama3.2"),
        persona=advisor_persona,
        role="assistant",
    )
    advisor_agent.respond("Hi, I'm interested in energy policy. What's your view?")
    advisor_agent.respond("And what about immigration policy?")

    # 5) Post-survey (same agent, includes conversation history in context)
    post_results = admin.administer_post_survey(advisor_agent)
    print(f"Post-survey ideological score: {post_results.ideological_score:.3f}")
    print(f"Conversation turns in context: {post_results.metadata.get('conversation_turns_in_context', 0)}")

    # 6) Compute treatment effect
    delta = calculate_polarization_delta(pre_results, post_results)
    print("\n=== Treatment Effects ===")
    print(f"Ideological change: {delta.ideological_delta:+.3f}")
    print(f"Overall change: {delta.overall_delta:+.3f}")


if __name__ == "__main__":
    main()
