"""
Command-line interface for the Synthetic Experiments Framework.

This CLI provides social scientists with easy access to common framework
operations without writing Python code.

Usage:
    synthetic-exp run config.yaml                    # Run experiment from config
    synthetic-exp run config.yaml --replicates 5    # Run 5 replicates
    synthetic-exp survey --provider ollama --model llama3.2 --survey bail2018
    synthetic-exp analyze results/                   # Analyze experiment results
    synthetic-exp list-surveys                       # List available survey instruments
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="synthetic-exp")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
def cli(verbose: bool, quiet: bool):
    """
    Synthetic Experiments Framework CLI.
    
    A tool for social scientists to run LLM conversation experiments.
    
    Examples:
    
        # Run an experiment from config file
        synthetic-exp run experiments/config.yaml
        
        # Run with 5 replicates
        synthetic-exp run experiments/config.yaml --replicates 5
        
        # Administer a standalone survey
        synthetic-exp survey --provider ollama --model llama3.2
        
        # Analyze results
        synthetic-exp analyze results/my_experiment/
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--replicates', '-r', default=1, help='Number of conversation replicates')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--max-turns', '-t', type=int, help='Override max turns from config')
@click.option('--topic', help='Override initial topic from config')
def run(config_file: str, replicates: int, output: Optional[str], 
        max_turns: Optional[int], topic: Optional[str]):
    """
    Run an experiment from a YAML configuration file.
    
    CONFIG_FILE: Path to the YAML experiment configuration.
    
    Examples:
    
        synthetic-exp run config.yaml
        synthetic-exp run config.yaml --replicates 10
        synthetic-exp run config.yaml -o results/my_study/ -t 30
    """
    try:
        from synthetic_experiments.experiments.config import load_experiment_config
        
        click.echo(f"Loading experiment from: {config_file}")
        experiment = load_experiment_config(config_file)
        
        if output:
            experiment.config.output_dir = output
            
        run_kwargs = {}
        if max_turns:
            run_kwargs['max_turns'] = max_turns
        if topic:
            run_kwargs['initial_topic'] = topic
            
        if replicates > 1:
            click.echo(f"Running {replicates} replicates...")
            results = experiment.run_multiple(replicates, **run_kwargs)
            click.echo(f"✓ Completed {len(results)} conversations")
            
            # Show summary
            total_turns = sum(len(r.turns) for r in results)
            total_tokens = sum(r.get_total_tokens() for r in results)
            click.echo(f"  Total turns: {total_turns}")
            click.echo(f"  Total tokens: {total_tokens}")
        else:
            click.echo("Running single conversation...")
            result = experiment.run(**run_kwargs)
            click.echo(f"✓ Completed: {len(result.turns)} turns, {result.get_total_tokens()} tokens")
            
        if experiment.storage:
            click.echo(f"Results saved to: {experiment.config.output_dir}")
            
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running experiment: {e}", err=True)
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--provider', '-p', type=click.Choice(['ollama', 'claude', 'openai']), 
              default='ollama', help='LLM provider to use')
@click.option('--model', '-m', default='llama3.2', help='Model name')
@click.option('--survey', '-s', type=click.Choice(['default', 'bail2018']), 
              default='default', help='Survey instrument to use')
@click.option('--persona', type=click.Path(exists=True), help='Path to persona YAML file')
@click.option('--seed', default=42, type=int, help='Random seed for reproducibility')
@click.option('--output', '-o', type=click.Path(), help='Output file for results (JSON)')
def survey(provider: str, model: str, survey: str, persona: Optional[str], 
           seed: int, output: Optional[str]):
    """
    Administer a standalone polarization survey to an LLM.
    
    This runs a pre-survey (baseline measurement) without any conversation context.
    Useful for establishing baselines or testing survey instruments.
    
    Examples:
    
        synthetic-exp survey --provider ollama --model llama3.2
        synthetic-exp survey --survey bail2018 --seed 123
        synthetic-exp survey --persona my_advisor.yaml -o results.json
    """
    try:
        from synthetic_experiments.analysis.survey import SurveyAdministrator
        from synthetic_experiments.agents import Persona
        
        # Get provider class
        if provider == 'ollama':
            from synthetic_experiments.providers import OllamaProvider
            provider_class = OllamaProvider
            provider_kwargs = {'model_name': model}
        elif provider == 'claude':
            from synthetic_experiments.providers import ClaudeProvider
            provider_class = ClaudeProvider
            provider_kwargs = {'model_name': model}
        elif provider == 'openai':
            from synthetic_experiments.providers import OpenAIProvider
            provider_class = OpenAIProvider
            provider_kwargs = {'model_name': model}
            
        # Load or create persona
        if persona:
            agent_persona = Persona.from_yaml(persona)
            click.echo(f"Using persona: {agent_persona.name}")
        else:
            agent_persona = Persona(
                name="Survey Respondent",
                background="An AI assistant responding to survey questions",
                communication_style="direct and honest",
                beliefs={}
            )
            
        click.echo(f"Administering {survey} survey using {provider}/{model}...")
        click.echo(f"Seed: {seed}")
        
        admin = SurveyAdministrator(
            provider_class=provider_class,
            provider_kwargs=provider_kwargs,
            persona=agent_persona,
            seed=seed,
            survey=survey
        )
        
        results = admin.administer_pre_survey()
        
        click.echo(f"\n=== Survey Results ===")
        click.echo(f"Affective score: {results.affective_score:.3f}")
        click.echo(f"Ideological score: {results.ideological_score:.3f}")
        click.echo(f"Overall score: {results.overall_score:.3f}")
        click.echo(f"Valid response rate: {results.valid_response_rate:.1%}")
        
        if output:
            import json
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            click.echo(f"\nResults saved to: {output}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command('list-surveys')
def list_surveys():
    """
    List available survey instruments.
    
    Shows all built-in survey types with descriptions.
    """
    from synthetic_experiments.analysis.survey import list_available_surveys
    
    surveys = list_available_surveys()
    
    click.echo("Available Survey Instruments:")
    click.echo("=" * 60)
    
    for key, description in surveys.items():
        if not key.endswith('_et_al') and key != 'bail':  # Skip aliases
            click.echo(f"\n{key}:")
            # Wrap description
            words = description.split()
            lines = []
            current_line = "  "
            for word in words:
                if len(current_line) + len(word) + 1 > 60:
                    lines.append(current_line)
                    current_line = "  " + word
                else:
                    current_line += " " + word if current_line != "  " else word
            lines.append(current_line)
            click.echo("\n".join(lines))


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--export-csv', is_flag=True, help='Export summary CSV')
@click.option('--export-turns', is_flag=True, help='Export turn-by-turn CSV')
def analyze(results_dir: str, export_csv: bool, export_turns: bool):
    """
    Analyze experiment results from a directory.
    
    RESULTS_DIR: Path to the experiment results directory.
    
    Examples:
    
        synthetic-exp analyze results/my_experiment/
        synthetic-exp analyze results/my_experiment/ --export-csv
    """
    try:
        from synthetic_experiments.data import ExperimentStorage
        from synthetic_experiments.analysis import calculate_basic_metrics
        from synthetic_experiments.analysis.political import analyze_conversation_polarization
        
        storage = ExperimentStorage(results_dir)
        conversations = storage.list_conversations()
        
        if not conversations:
            click.echo("No conversations found in directory.")
            return
            
        click.echo(f"Found {len(conversations)} conversation(s)")
        click.echo("=" * 50)
        
        total_turns = 0
        total_tokens = 0
        
        for conv_id in conversations:
            conv = storage.load_conversation(conv_id)
            metrics = calculate_basic_metrics(conv)
            
            click.echo(f"\n{conv_id}:")
            click.echo(f"  Turns: {metrics.total_turns}")
            click.echo(f"  Tokens: {metrics.total_tokens}")
            click.echo(f"  Avg message length: {metrics.avg_message_length:.1f} chars")
            
            total_turns += metrics.total_turns
            total_tokens += metrics.total_tokens
            
            # Try political analysis if applicable
            try:
                polarization = analyze_conversation_polarization(conv)
                avg_pol = polarization['overall_metrics']['avg_polarization']
                click.echo(f"  Avg polarization: {avg_pol:.3f}")
            except Exception:
                pass  # Skip if political analysis fails
                
        click.echo("\n" + "=" * 50)
        click.echo(f"Total: {total_turns} turns, {total_tokens} tokens")
        
        if export_csv:
            csv_path = storage.export_summary_csv()
            click.echo(f"\nExported summary to: {csv_path}")
            
        if export_turns:
            csv_path = storage.export_turns_csv()
            click.echo(f"Exported turns to: {csv_path}")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('persona_file', type=click.Path(exists=True))
def validate_persona(persona_file: str):
    """
    Validate a persona YAML file.
    
    Checks that the persona file is valid and shows its contents.
    """
    try:
        from synthetic_experiments.agents import Persona
        
        persona = Persona.from_yaml(persona_file)
        
        click.echo(f"✓ Valid persona: {persona.name}")
        click.echo(f"  Background: {persona.background[:50]}..." if len(persona.background) > 50 else f"  Background: {persona.background}")
        click.echo(f"  Political orientation: {persona.political_orientation or 'not specified'}")
        click.echo(f"  Communication style: {persona.communication_style}")
        click.echo(f"  Beliefs: {len(persona.beliefs)} defined")
        
    except Exception as e:
        click.echo(f"✗ Invalid persona: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate_config(config_file: str):
    """
    Validate an experiment configuration file.
    
    Checks that the config file is valid without running the experiment.
    """
    try:
        from synthetic_experiments.experiments.config import load_yaml_config, validate_config
        
        config = load_yaml_config(config_file)
        validate_config(config)
        
        exp_config = config.get('experiment', {})
        agents_config = config.get('agents', [])
        
        click.echo(f"✓ Valid configuration")
        click.echo(f"  Experiment: {exp_config.get('name', 'unnamed')}")
        click.echo(f"  Max turns: {exp_config.get('max_turns', 20)}")
        click.echo(f"  Agents: {len(agents_config)}")
        
        for i, agent in enumerate(agents_config):
            click.echo(f"    [{i+1}] {agent.get('name', 'unnamed')} ({agent.get('role', 'unknown')})")
            
    except Exception as e:
        click.echo(f"✗ Invalid configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
def init():
    """
    Initialize a new experiment in the current directory.
    
    Creates a basic config.yaml and personas/ directory structure.
    """
    import os
    
    # Create directories
    os.makedirs('personas', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Create sample config
    config_content = '''# Synthetic Experiment Configuration
experiment:
  name: "my_experiment"
  max_turns: 20
  initial_topic: "climate change"
  output_dir: "results"

agents:
  - name: "user"
    role: "user"
    provider:
      type: "ollama"
      model: "llama3.2"
    persona:
      file: "personas/user.yaml"

  - name: "advisor"
    role: "assistant"
    provider:
      type: "ollama"
      model: "llama3.2"
    persona:
      name: "Neutral Advisor"
      background: "A balanced, helpful AI assistant"
      communication_style: "calm and informative"
'''
    
    # Create sample persona
    persona_content = '''name: "Sample User"
background: "A curious person interested in learning about various topics"
political_orientation: "moderate"
communication_style: "inquisitive and open-minded"
goals: "Learn more and form well-informed opinions"
beliefs:
  general: "I try to see multiple perspectives on issues"
'''
    
    if not os.path.exists('config.yaml'):
        with open('config.yaml', 'w') as f:
            f.write(config_content)
        click.echo("✓ Created config.yaml")
    else:
        click.echo("  config.yaml already exists (skipped)")
        
    if not os.path.exists('personas/user.yaml'):
        with open('personas/user.yaml', 'w') as f:
            f.write(persona_content)
        click.echo("✓ Created personas/user.yaml")
    else:
        click.echo("  personas/user.yaml already exists (skipped)")
        
    click.echo("\nExperiment initialized! Next steps:")
    click.echo("  1. Edit config.yaml to customize your experiment")
    click.echo("  2. Edit personas/user.yaml or add more personas")
    click.echo("  3. Run: synthetic-exp run config.yaml")


if __name__ == '__main__':
    cli()
