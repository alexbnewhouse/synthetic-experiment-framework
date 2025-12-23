# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Synthetic Experiment Framework**: A Python framework for social scientists to design and conduct experimental research on LLM conversations. Primary focus is political polarization research, but designed for extensibility to other social science domains.

**Target audience**: Social scientists with basic Python knowledge
**Design principle**: Readability and clarity over cleverness

## Architecture

### Core Components (in dependency order)

1. **Providers** (`synthetic_experiments/providers/`)
   - Abstract base: `base.py` defines `LLMProvider`, `Message`, `GenerationConfig`, `GenerationResult`
   - Implementations: `ollama.py` (primary), `claude.py`, `openai.py`
   - Pattern: Abstract base class with concrete implementations for each LLM service

2. **Personas** (`synthetic_experiments/agents/persona.py`)
   - `Persona` dataclass: Defines agent characteristics (background, beliefs, communication style)
   - `PersonaFactory`: Pre-built personas for political research
   - Supports both programmatic creation and YAML file loading

3. **Agents** (`synthetic_experiments/agents/agent.py`)
   - `ConversationAgent`: Combines provider + persona + conversation history
   - Manages turn-taking and message generation
   - Tracks usage statistics (tokens, cost)

4. **Data Collection** (`synthetic_experiments/data/`)
   - `logger.py`: `ConversationLogger` records all conversation turns with timestamps
   - `storage.py`: `ExperimentStorage` manages file I/O, exports CSV/JSON

5. **Experiments** (`synthetic_experiments/experiments/`)
   - `experiment.py`: `Experiment` class orchestrates multi-agent conversations
   - `config.py`: YAML configuration loading and validation
   - Handles replicates, metadata, stopping conditions

6. **Analysis** (`synthetic_experiments/analysis/`)
   - `metrics.py`: Basic conversation metrics (length, turns, tokens)
   - `political.py`: Political language detection, polarization metrics, opinion shift
   - `survey.py`: Pre/post survey system for measuring polarization treatment effects
   - `MetricCalculator`: Extensible framework for custom metrics

## Development Commands

### Setup
```bash
pip install -e .  # Install in development mode
pip install -r requirements.txt
```

### Running Tests
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

# Skip integration tests (require Ollama running)
pytest tests/ -m "not integration"

# Run example experiment
cd examples/political_polarization
python run_experiment.py
```

### Running the Political Polarization Study
```bash
cd examples/political_polarization

# Single conversation
python run_experiment.py

# 5 replicates
python run_experiment.py --replicates 5

# Full factorial design (45 conversations)
python run_experiment.py --full-design
```

## Key Design Patterns

### Provider Pattern
All LLM providers implement the same interface (`LLMProvider`), making them interchangeable:
```python
# All providers work identically
provider = OllamaProvider("llama2")
# OR
provider = ClaudeProvider("claude-3-sonnet-20240229")
# OR
provider = OpenAIProvider("gpt-4")

# Same interface for all
result = provider.generate(messages, generation_config)
```

### Configuration-Driven Experiments
Experiments can be defined in YAML for reproducibility:
```yaml
experiment:
  name: "my_study"
  max_turns: 20

agents:
  - role: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      file: "personas/conservative.yaml"
```

Load and run:
```python
experiment = load_experiment_config("config.yaml")
results = experiment.run()
```

### Data Flow
```
Experiment → Agents → Provider → LLM
                ↓
          ConversationLogger → ExperimentStorage → JSON/CSV files
```

## File Organization

### Core Library (`synthetic_experiments/`)
- Organized by function: providers, agents, experiments, data, analysis
- Each module has `__init__.py` with clear exports
- Main `__init__.py` exports commonly-used classes for convenience

### Examples (`examples/`)
- Complete, runnable research workflows
- `political_polarization/`: Full factorial design example
- Each example includes personas, config, and run script

### Documentation (`docs/`)
- `getting_started.md`: Installation and first experiment
- `user_guide.md`: Complete feature walkthrough
- `political_polarization_tutorial.md`: Research workflow example
- `api_reference.md`: Detailed API docs (placeholder)

## Code Style and Conventions

### Docstrings
- All public classes, methods, and functions have comprehensive docstrings
- Format: Google-style docstrings with examples
- Example code in docstrings uses `>>>` format

### Type Hints
- Use type hints for all function signatures
- Import from `typing` module for complex types
- Helps both IDEs and social scientist users understand APIs

### Naming Conventions
- Classes: `PascalCase` (e.g., `ConversationAgent`)
- Functions/methods: `snake_case` (e.g., `calculate_basic_metrics`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `CONSERVATIVE_LANGUAGE`)
- Private methods: `_leading_underscore` (e.g., `_prepare_messages`)

### Imports
- Standard library first, then third-party, then local
- Explicit imports preferred over wildcards
- Example:
  ```python
  from typing import List, Dict, Any
  import logging

  from synthetic_experiments.providers.base import LLMProvider
  from synthetic_experiments.agents.persona import Persona
  ```

## Common Tasks

### Adding a New Provider
1. Create new file in `synthetic_experiments/providers/`
2. Inherit from `LLMProvider` (in `base.py`)
3. Implement required methods: `generate()`, `validate_config()`, `get_model_info()`
4. Add to `providers/__init__.py`
5. Update `config.py` to support YAML configuration

### Adding a New Metric
```python
from synthetic_experiments.analysis import MetricCalculator

calculator = MetricCalculator()

def my_custom_metric(turn):
    # Your logic here
    return some_value

calculator.register_metric("my_metric", my_custom_metric)
```

### Using the Survey System for Treatment Effects
The survey system measures how conversations (in context) affect LLM "advisor" survey responses:

```python
from synthetic_experiments.analysis.survey import (
    SurveyAdministrator,
    BailEtAlSurvey,
    calculate_polarization_delta,
    list_available_surveys
)
from synthetic_experiments.providers.ollama import OllamaProvider
from synthetic_experiments.agents import Persona, ConversationAgent

# Create survey administrator
# Survey options: "default" (12 questions) or "bail2018" (10-item PNAS replication)
admin = SurveyAdministrator(
    provider_class=OllamaProvider,
    provider_kwargs={"model_name": "llama3.2"},
    persona=Persona.from_yaml("advisor.yaml"),
    seed=42,
    survey="bail2018"  # Bail et al. (2018) PNAS 10-item policy scale
)

# Pre-survey (fresh LLM - baseline measurement)
pre_results = admin.administer_pre_survey()

# Run conversation experiment
advisor_agent = ConversationAgent(provider=..., persona=...)
# ... conversation runs, building advisor_agent.conversation_history ...

# Post-survey (with conversation in context!)
post_results = admin.administer_post_survey(advisor_agent)

# Calculate treatment effect
delta = calculate_polarization_delta(pre_results, post_results)
```

Key design: 
- Pre-survey: Fresh LLM instance (baseline, no context)
- Post-survey: Uses advisor agent WITH conversation history in context window
- This measures how conversation context affects LLM responses
- Two built-in surveys: "default" (affective + ideological) or "bail2018" (PNAS replication)

### Creating a New Persona
Option 1 - YAML file:
```yaml
name: "My Persona"
background: "Description"
communication_style: "Style"
beliefs:
  topic: "belief"
```

Option 2 - Programmatically:
```python
persona = Persona(
    name="My Persona",
    background="Description",
    communication_style="Style",
    beliefs={"topic": "belief"}
)
```

## Important Constraints

### Social Science Focus
- Prioritize readability over performance optimizations
- Use terminology familiar to social scientists (experiment, condition, replicate)
- Export data in formats social scientists use (CSV for R/pandas/SPSS)
- Provide clear examples in documentation

### Local-First Philosophy
- Ollama support is primary; API providers are secondary
- Default to privacy-preserving local models
- Document costs clearly when using API providers

### Extensibility
- Framework should support domains beyond political research
- Personas and metrics should be easily customizable
- Avoid hardcoding research-specific logic in core modules

## Testing and Validation

### Before Committing Changes
1. Run the test suite: `pytest tests/`
2. Check coverage: `pytest tests/ --cov=synthetic_experiments`
3. Ensure imports work: `python -c "import synthetic_experiments"`
4. Run example: `cd examples/political_polarization && python run_experiment.py`
5. Check docstrings are clear and include examples
6. Verify YAML configs still load: `load_experiment_config("config.yaml")`

### When Adding Features
- Add corresponding tests in `tests/` directory
- Add docstrings with examples
- Update relevant documentation in `docs/`
- Consider if feature needs example in `examples/`
- Think about social scientist user perspective

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures, MockLLMProvider
├── pytest.ini               # pytest configuration
├── utils/                   # Test utilities and helpers
├── test_providers/          # Tests for Ollama, Claude, OpenAI providers
├── test_agents/             # Tests for Persona, PersonaFactory, ConversationAgent
├── test_experiments/        # Tests for Experiment, config loading
├── test_data/               # Tests for ConversationLogger, ExperimentStorage
├── test_analysis/           # Tests for metrics, political analysis
└── test_integration.py      # End-to-end integration tests
```

## Dependencies

### Core (Required)
- `ollama`: Local LLM access (primary)
- `anthropic`: Claude API access
- `openai`: OpenAI API access
- `pyyaml`: YAML config parsing
- `pydantic`: Data validation
- `pandas`: Data export and analysis
- `click`: CLI (if needed)

### Optional (Analysis)
- `textblob`: More sophisticated sentiment analysis
- `matplotlib`, `seaborn`: Visualization
- `jupyter`: Interactive analysis

## Future Enhancements (Not Yet Implemented)

- CLI tool (`synthetic-exp` command)
- API reference documentation
- Streaming conversation support
- Real-time metrics dashboard
- More sophisticated stopping conditions
- Multi-agent (>2 participants) support
- Conversation continuation/interruption

## Implemented Features

- ✅ Unit tests (pytest) - 260+ tests with ~79% coverage
- ✅ Pre/post survey system for measuring polarization treatment effects

## Critical Files to Understand

1. `synthetic_experiments/providers/base.py` - Foundation of provider system
2. `synthetic_experiments/agents/agent.py` - Core agent logic
3. `synthetic_experiments/experiments/experiment.py` - Orchestration
4. `synthetic_experiments/experiments/config.py` - YAML configuration system
5. `synthetic_experiments/analysis/survey.py` - Pre/post survey measurement system
6. `examples/political_polarization/run_experiment.py` - Complete usage example
