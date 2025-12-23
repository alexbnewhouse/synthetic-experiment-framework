# Synthetic Experiment Framework

A Python framework for social scientists to design and conduct experimental research on conversations with LLM chatbots.

## Overview

The Synthetic Experiment Framework enables rigorous social science research on LLM conversations through:

- **Modular architecture**: Swap LLM providers, personas, and experimental configurations easily
- **Local-first**: Prioritizes privacy and cost control with Ollama, supports Claude/OpenAI APIs
- **Research-ready**: Built-in data collection, metrics, and analysis tools for social science workflows
- **Readable code**: Clear documentation and examples designed for social scientists

### Primary Use Case: Political Polarization Research

Study how different user personas and chatbot framing affect political discourse through synthetic conversations between two LLMs, where one assumes a user persona and the other acts as an advisor.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull a model
# Visit https://ollama.ai for installation instructions
ollama pull llama2
```

### Run Your First Experiment

```python
from synthetic_experiments import Experiment
from synthetic_experiments.providers import OllamaProvider
from synthetic_experiments.agents import ConversationAgent, PersonaFactory

# Create agents with pre-built personas
provider = OllamaProvider(model_name="llama2")

user = ConversationAgent(
    provider=provider,
    persona=PersonaFactory.create_political_user("conservative", "moderate"),
    role="user"
)

advisor = ConversationAgent(
    provider=provider,
    persona=PersonaFactory.create_neutral_advisor(),
    role="assistant"
)

# Run experiment
experiment = Experiment(name="political_discourse", agents=[user, advisor])
results = experiment.run(max_turns=20, initial_topic="climate policy")

print(f"Completed {len(results.turns)} turns")
print(f"Total tokens: {results.get_total_tokens()}")
```

### Run the Political Polarization Example

```bash
cd examples/political_polarization

# Single conversation
python run_experiment.py

# Multiple replicates
python run_experiment.py --replicates 5

# Full factorial design (3x3, 45 conversations)
python run_experiment.py --full-design
```

## Key Features

### 1. Flexible LLM Provider System

Support for multiple LLM backends:

```python
# Local models via Ollama (primary)
from synthetic_experiments.providers import OllamaProvider
provider = OllamaProvider(model_name="llama2")

# Claude via Anthropic API
from synthetic_experiments.providers import ClaudeProvider
provider = ClaudeProvider(model_name="claude-3-sonnet-20240229")

# OpenAI GPT models
from synthetic_experiments.providers import OpenAIProvider
provider = OpenAIProvider(model_name="gpt-4")
```

### 2. Rich Persona System

Define agent characteristics programmatically or via YAML:

```yaml
# persona.yaml
name: "Conservative Voter"
background: "Middle-aged suburban voter"
political_orientation: "conservative"
communication_style: "direct and principled"
beliefs:
  economy: "Free market solutions work best"
  government: "Limited government is better"
```

Pre-built personas for political research:
- Conservative, Liberal, Moderate users
- Neutral, Empathetic, Challenging advisors

### 3. Comprehensive Data Collection

Automatic logging of:
- Full conversation transcripts (JSON)
- Turn-by-turn statistics (CSV)
- Token usage and costs
- Custom metadata

### 4. Built-in Analysis Tools

```python
from synthetic_experiments.analysis import calculate_basic_metrics
from synthetic_experiments.analysis.political import (
    detect_political_language,
    analyze_conversation_polarization,
    calculate_opinion_shift
)

# Basic conversation metrics
metrics = calculate_basic_metrics(conversation)

# Political language detection
political = detect_political_language(message)

# Track polarization over time
polarization = analyze_conversation_polarization(conversation)

# Measure opinion shift
shift = calculate_opinion_shift(conversation)
```

### 5. Pre/Post Survey System for Treatment Effects

Measure how conversations in context affect LLM "advisor" survey responses:

```python
from synthetic_experiments.analysis.survey import (
    SurveyAdministrator,
    calculate_polarization_delta
)
from synthetic_experiments.providers import OllamaProvider
from synthetic_experiments.agents import Persona, ConversationAgent

# Create survey administrator
admin = SurveyAdministrator(
    provider_class=OllamaProvider,
    provider_kwargs={"model_name": "llama3.2"},
    persona=Persona.from_yaml("advisor.yaml"),
    seed=42,
    survey="bail2018"  # Use Bail et al. (2018) PNAS survey
)

# Pre-survey (fresh LLM - baseline)
pre_results = admin.administer_pre_survey()

# Run conversation experiment with advisor agent
advisor_agent = ConversationAgent(provider=provider, persona=persona)
# ... conversation runs here ...

# Post-survey (SAME agent with conversation in context)
post_results = admin.administer_post_survey(advisor_agent)

# Calculate treatment effect
delta = calculate_polarization_delta(pre_results, post_results)
print(f"Ideological shift: {delta.ideological_delta:+.3f}")
```

**Key features:**
- Pre-survey: Fresh LLM (baseline measurement)
- Post-survey: Includes conversation history in context window
- Measures how compressed conversations affect LLM responses
- Built-in surveys: `"default"` (affective + ideological) or `"bail2018"` (PNAS replication)

### 6. Experimental Design Support

Run factorial designs, multiple replicates, and batch experiments:

```python
# Run multiple replicates
results = experiment.run_multiple(
    num_conversations=10,
    initial_topic="climate change",
    metadata={"condition": "baseline"}
)

# Or use YAML configuration
experiment = load_experiment_config("config.yaml")
results = experiment.run()
```

## Testing

The framework includes a comprehensive test suite with 260+ tests covering all modules:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=synthetic_experiments --cov-report=term-missing

# Run specific test modules
pytest tests/test_providers/      # Provider tests
pytest tests/test_agents/         # Agent and persona tests
pytest tests/test_experiments/    # Experiment orchestration tests
pytest tests/test_analysis/       # Metrics and analysis tests
pytest tests/test_data/           # Data logging/storage tests

# Skip integration tests (that require external services)
pytest tests/ -m "not integration"
```

Test coverage: ~79% across all modules.

## Project Structure

```
synthetic_experiments/
├── providers/         # LLM provider implementations (Ollama, Claude, OpenAI)
├── agents/           # Conversation agents and persona system
├── experiments/      # Experiment orchestration and configuration
├── data/            # Data logging and storage
└── analysis/        # Metrics and analysis tools

examples/
├── political_polarization/   # Complete political polarization study
│   ├── personas/            # Pre-built political personas
│   ├── config.yaml          # Experiment configuration
│   └── run_experiment.py    # Run script

tests/
├── conftest.py              # Shared fixtures and MockLLMProvider
├── test_providers/          # Provider unit tests
├── test_agents/             # Agent and persona tests
├── test_experiments/        # Experiment tests
├── test_data/               # Data logging/storage tests
├── test_analysis/           # Metrics and analysis tests
└── test_integration.py      # End-to-end integration tests

docs/
├── getting_started.md                  # Installation and first steps
├── user_guide.md                       # Complete feature guide
├── political_polarization_tutorial.md  # Research workflow tutorial
└── api_reference.md                    # API documentation
```

## Documentation

- **[Getting Started](docs/getting_started.md)**: Installation and quick start
- **[User Guide](docs/user_guide.md)**: Complete feature documentation
- **[Political Polarization Tutorial](docs/political_polarization_tutorial.md)**: Step-by-step research example
- **[API Reference](docs/api_reference.md)**: Detailed API documentation

## Example: Political Polarization Study

Research question: *How do user political personas and advisor framing affect political discourse dynamics?*

**Experimental Design:**
- Independent variables: User persona (3) × Advisor type (3)
- Dependent variables: Polarization metrics, opinion shift, engagement
- Design: 3×3 factorial, 5 replicates per condition = 45 conversations

**Running the study:**

```bash
cd examples/political_polarization
python run_experiment.py --full-design
```

**Results:**
- Full transcripts in JSON
- Summary statistics in CSV
- Automated political language analysis
- Opinion shift tracking

See the [Political Polarization Tutorial](docs/political_polarization_tutorial.md) for complete details.

## Use Cases Beyond Political Research

The framework is designed for flexibility:

- **Consumer behavior**: How do chatbots influence purchasing decisions?
- **Health communication**: Effects of chatbot framing on health beliefs
- **Education**: Learning outcomes from different tutoring personas
- **Misinformation**: How personas respond to corrective information
- **Persuasion**: Effectiveness of different rhetorical strategies

Simply adapt personas and metrics to your research domain.

## Architecture Principles

1. **Modularity**: Every component (provider, persona, agent, experiment) is independently configurable
2. **Readability**: Clear class names, extensive docstrings, type hints, examples throughout
3. **Social Science Focus**:
   - YAML configurations (familiar to researchers)
   - CSV/JSON output (easy import to R, pandas, SPSS)
   - Pre-built analysis for common research questions
   - Terminology aligned with experimental design

4. **Extensibility**:
   - Plugin architecture for custom metrics
   - Custom persona factories
   - Extensible conversation controllers
   - Easy to add new providers

## Requirements

- Python 3.9+
- Ollama (for local models)
- Optional: API keys for Claude or OpenAI

Core dependencies:
- `ollama`, `anthropic`, `openai` - LLM providers
- `pyyaml` - Configuration management
- `pandas` - Data analysis
- `pydantic` - Data validation

Development dependencies:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting

See `requirements.txt` for complete list.

## Contributing

This is a research framework under active development. Contributions welcome:

- Bug reports and feature requests via GitHub Issues
- Code contributions via Pull Requests
- Research use cases and examples
- Additional personas and metrics

## Citation

If you use this framework in your research, please cite:

```
Newhouse, A. (2025). Synthetic Experiment Framework: A Python toolkit for
social science research on LLM conversations.
https://github.com/alexnewhouse/synthetic-experiment-framework
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built for social scientists studying the societal impacts of LLM chatbots. Designed to make rigorous computational social science research accessible to researchers without extensive programming backgrounds.

## Support

- **Documentation**: See `docs/` directory
- **Examples**: Check `examples/` for complete research workflows
- **Issues**: Report bugs or request features on GitHub
- **Questions**: Open a discussion on GitHub Discussions
