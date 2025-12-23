#!/usr/bin/env python3
"""
Run the political polarization experiment.

This script demonstrates how to run a full experimental design studying
political polarization in LLM conversations.

Usage:
    python run_experiment.py                    # Run single conversation
    python run_experiment.py --replicates 5     # Run 5 replicates
    python run_experiment.py --full-design      # Run full 3x3 factorial design
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthetic_experiments.experiments import load_experiment_config
from synthetic_experiments.agents import Persona

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_conversation(config_path: str, replicate: int = 1):
    """Run a single conversation from a configuration file."""
    logger.info(f"Loading experiment from {config_path}")

    experiment = load_experiment_config(config_path)

    logger.info(f"Running replicate {replicate}")
    logger.info(f"Agents: {[agent.name for agent in experiment.agents]}")
    logger.info(f"Max turns: {experiment.config.max_turns}")

    # Run the conversation
    result = experiment.run(metadata={"replicate": replicate})

    logger.info(f"\n{'='*60}")
    logger.info(f"CONVERSATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Conversation ID: {result.conversation_id}")
    logger.info(f"Total turns: {len(result.turns)}")
    logger.info(f"Total tokens: {result.get_total_tokens()}")
    logger.info(f"Total cost: ${result.get_total_cost():.4f}")
    logger.info(f"Duration: {result.get_duration():.2f} seconds")
    logger.info(f"{'='*60}\n")

    return result


def run_multiple_replicates(config_path: str, num_replicates: int):
    """Run multiple replicates of the same experimental condition."""
    logger.info(f"Running {num_replicates} replicates")

    experiment = load_experiment_config(config_path)

    results = experiment.run_multiple(
        num_conversations=num_replicates
    )

    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")

    stats = experiment.get_statistics()
    logger.info(f"Total conversations: {stats['total_conversations']}")
    logger.info(f"Total turns: {stats['total_turns']}")
    logger.info(f"Avg turns/conversation: {stats['avg_turns_per_conversation']:.1f}")
    logger.info(f"Total tokens: {stats['total_tokens']}")
    logger.info(f"Total cost: ${stats['total_cost']:.4f}")
    logger.info(f"Avg cost/conversation: ${stats['avg_cost_per_conversation']:.4f}")
    logger.info(f"{'='*60}\n")

    return results


def run_full_factorial_design():
    """
    Run full 3x3 factorial design.

    Independent Variables:
    - User persona: Conservative, Liberal, Moderate (3 levels)
    - Advisor type: Neutral, Empathetic, Challenging (3 levels)

    Total: 9 conditions × 5 replicates = 45 conversations
    """
    logger.info("Running full factorial design (3x3)")
    logger.info("This will run 9 conditions × 5 replicates = 45 conversations")
    logger.info("This may take a while...\n")

    user_personas = [
        "personas/conservative_user.yaml",
        "personas/liberal_user.yaml",
        "personas/moderate_user.yaml"
    ]

    advisor_personas = [
        "personas/neutral_advisor.yaml",
        "personas/empathetic_advisor.yaml",
        "personas/challenging_advisor.yaml"
    ]

    topics = ["climate policy", "healthcare reform", "immigration policy"]

    condition_num = 0
    total_conditions = len(user_personas) * len(advisor_personas) * len(topics)

    for user_persona_file in user_personas:
        for advisor_persona_file in advisor_personas:
            for topic in topics:
                condition_num += 1

                # Load personas to get names
                user_persona = Persona.from_yaml(user_persona_file)
                advisor_persona = Persona.from_yaml(advisor_persona_file)

                logger.info(f"\n{'='*60}")
                logger.info(f"Condition {condition_num}/{total_conditions}")
                logger.info(f"User: {user_persona.name}")
                logger.info(f"Advisor: {advisor_persona.name}")
                logger.info(f"Topic: {topic}")
                logger.info(f"{'='*60}\n")

                # Create experiment programmatically
                from synthetic_experiments import Experiment
                from synthetic_experiments.agents import ConversationAgent
                from synthetic_experiments.providers import OllamaProvider

                user_agent = ConversationAgent(
                    provider=OllamaProvider(model_name="llama2"),
                    persona=user_persona,
                    role="user"
                )

                advisor_agent = ConversationAgent(
                    provider=OllamaProvider(model_name="llama2"),
                    persona=advisor_persona,
                    role="assistant"
                )

                experiment = Experiment(
                    name=f"political_polarization_full_design",
                    agents=[user_agent, advisor_agent],
                    output_dir="results/full_factorial_design"
                )

                # Run 5 replicates for this condition
                experiment.run_multiple(
                    num_conversations=5,
                    initial_topic=topic,
                    metadata={
                        "condition": condition_num,
                        "user_persona": user_persona.name,
                        "advisor_persona": advisor_persona.name,
                        "topic": topic
                    }
                )

    logger.info(f"\n{'='*60}")
    logger.info(f"FULL FACTORIAL DESIGN COMPLETE")
    logger.info(f"Total conversations: {total_conditions * 5}")
    logger.info(f"Results saved to: results/full_factorial_design/")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run political polarization experiment"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of replicates to run"
    )
    parser.add_argument(
        "--full-design",
        action="store_true",
        help="Run full 3x3 factorial design (45 conversations)"
    )

    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)

    try:
        if args.full_design:
            run_full_factorial_design()
        elif args.replicates > 1:
            run_multiple_replicates(args.config, args.replicates)
        else:
            run_single_conversation(args.config)

    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running experiment: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
