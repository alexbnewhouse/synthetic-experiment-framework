# Getting Started with Synthetic Experiments Framework

Welcome to the Synthetic Experiments Framework! This guide will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai) installed and running (for local models)
- Optional: API keys for Claude or OpenAI (for cloud models)

### Install the Package

```bash
# Clone the repository
git clone https://github.com/alexnewhouse/synthetic-experiment-framework.git
cd synthetic-experiment-framework

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Install Ollama (for local models)

1. Download and install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull a model:
```bash
ollama pull llama2
```

## Quick Start: Your First Experiment

### Option 1: Using a Configuration File

Create a file called `my_experiment.yaml`:

```yaml
experiment:
  name: "my_first_experiment"
  max_turns: 10
  initial_topic: "renewable energy"

agents:
  - name: "curious_user"
    role: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Curious Student"
      background: "College student interested in learning"
      communication_style: "inquisitive and open-minded"

  - name: "helpful_advisor"
    role: "assistant"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      name: "Knowledgeable Advisor"
      background: "Expert on various topics"
      communication_style: "clear and educational"
```

Run the experiment:

```python
from synthetic_experiments import load_experiment_config

# Load and run
experiment = load_experiment_config("my_experiment.yaml")
results = experiment.run()

# View results
print(f"Conversation completed with {len(results.turns)} turns")
print(f"Total tokens: {results.get_total_tokens()}")
```

### Option 2: Programmatic API

```python
from synthetic_experiments import Experiment
from synthetic_experiments.providers import OllamaProvider
from synthetic_experiments.agents import ConversationAgent, Persona

# Create provider
provider = OllamaProvider(model_name="llama2")

# Create personas
user_persona = Persona(
    name="Curious Student",
    background="Interested in learning about science",
    communication_style="inquisitive"
)

advisor_persona = Persona(
    name="Science Teacher",
    background="Experienced educator",
    communication_style="clear and patient"
)

# Create agents
user = ConversationAgent(
    provider=provider,
    persona=user_persona,
    role="user"
)

advisor = ConversationAgent(
    provider=provider,
    persona=advisor_persona,
    role="assistant"
)

# Create and run experiment
experiment = Experiment(
    name="science_discussion",
    agents=[user, advisor]
)

results = experiment.run(
    max_turns=10,
    initial_topic="photosynthesis"
)

# Analyze results
from synthetic_experiments.analysis import calculate_basic_metrics

metrics = calculate_basic_metrics(results)
print(f"Average message length: {metrics.avg_message_length:.1f} characters")
```

## Running the Political Polarization Example

The framework comes with a complete political polarization experiment:

```bash
cd examples/political_polarization

# Run a single conversation
python run_experiment.py

# Run 5 replicates
python run_experiment.py --replicates 5

# Run the full factorial design (45 conversations - takes a while!)
python run_experiment.py --full-design
```

## Next Steps

- Read the [User Guide](user_guide.md) for detailed explanations
- Explore the [API Reference](api_reference.md) for all available features
- Follow the [Political Polarization Tutorial](political_polarization_tutorial.md) to understand the research workflow

## Troubleshooting

### Ollama Connection Issues

If you get connection errors:

1. Make sure Ollama is running: `ollama serve`
2. Check that a model is installed: `ollama list`
3. Pull a model if needed: `ollama pull llama2`

### Model Not Found

If a model isn't found, the framework will automatically pull it (if `auto_pull=True`). This may take a few minutes the first time.

### API Key Issues

For Claude or OpenAI, set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

Or pass it directly:

```python
from synthetic_experiments.providers import ClaudeProvider

provider = ClaudeProvider(
    model_name="claude-3-sonnet-20240229",
    api_key="your-key-here"
)
```

## Getting Help

- Check the documentation in the `docs/` folder
- Review example code in `examples/`
- Open an issue on GitHub for bugs or feature requests
