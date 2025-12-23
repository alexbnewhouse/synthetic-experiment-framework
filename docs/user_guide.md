# User Guide

This guide explains the key concepts and features of the Synthetic Experiments Framework.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Providers](#providers)
3. [Personas](#personas)
4. [Agents](#agents)
5. [Experiments](#experiments)
6. [Data Collection](#data-collection)
7. [Analysis](#analysis)
8. [Best Practices](#best-practices)

## Core Concepts

### The Framework Architecture

The framework is built around four key components:

1. **Providers**: Interface to LLM models (Ollama, Claude, OpenAI)
2. **Personas**: Define agent characteristics and behavior
3. **Agents**: Combine providers + personas to create conversation participants
4. **Experiments**: Orchestrate multi-agent conversations and collect data

### Experimental Workflow

```
Design → Configure → Run → Analyze
```

1. **Design**: Define your research question and experimental conditions
2. **Configure**: Create personas and experiment configurations
3. **Run**: Execute conversations and collect data
4. **Analyze**: Extract metrics and insights from conversations

## Providers

Providers connect to different LLM services. The framework supports three types:

### Ollama (Local Models)

Best for: Privacy, cost control, rapid iteration

```python
from synthetic_experiments.providers import OllamaProvider

provider = OllamaProvider(
    model_name="llama2",  # or "mistral", "llama3", etc.
    base_url="http://localhost:11434",
    auto_pull=True  # Automatically download model if needed
)
```

### Claude (Anthropic API)

Best for: High-quality responses, latest models

```python
from synthetic_experiments.providers import ClaudeProvider
import os

provider = ClaudeProvider(
    model_name="claude-3-sonnet-20240229",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
```

### OpenAI (GPT Models)

Best for: GPT-specific research, comparison studies

```python
from synthetic_experiments.providers import OpenAIProvider

provider = OpenAIProvider(
    model_name="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)
```

### Generation Configuration

Control how models generate responses:

```python
from synthetic_experiments.providers.base import GenerationConfig

config = GenerationConfig(
    temperature=0.7,      # Higher = more random (0.0-2.0)
    max_tokens=1000,      # Maximum response length
    top_p=0.9,           # Nucleus sampling
    seed=42              # For reproducibility
)
```

## Personas

Personas define the characteristics, beliefs, and communication style of agents.

### Creating Personas Programmatically

```python
from synthetic_experiments.agents import Persona

persona = Persona(
    name="Climate Activist",
    background="Young urban professional concerned about environment",
    political_orientation="liberal",
    communication_style="passionate and data-driven",
    goals="Advocate for climate action",
    beliefs={
        "climate": "Climate change is an urgent crisis",
        "policy": "Government must take aggressive action"
    }
)
```

### Creating Personas from YAML

Create `my_persona.yaml`:

```yaml
name: "Climate Activist"
background: "Young urban professional concerned about environment"
political_orientation: "liberal"
communication_style: "passionate and data-driven"
goals: "Advocate for climate action"
beliefs:
  climate: "Climate change is an urgent crisis"
  policy: "Government must take aggressive action"
```

Load it:

```python
persona = Persona.from_yaml("my_persona.yaml")
```

### Using Pre-built Personas

The framework provides factory methods for common persona types:

```python
from synthetic_experiments.agents import PersonaFactory

# Political user personas
conservative = PersonaFactory.create_political_user("conservative", intensity="moderate")
liberal = PersonaFactory.create_political_user("liberal", intensity="high")
moderate = PersonaFactory.create_political_user("moderate")

# Advisor personas
neutral = PersonaFactory.create_neutral_advisor()
empathetic = PersonaFactory.create_empathetic_advisor()
challenging = PersonaFactory.create_challenging_advisor()
```

## Agents

Agents combine providers and personas to participate in conversations.

### Creating Agents

```python
from synthetic_experiments.agents import ConversationAgent
from synthetic_experiments.providers import OllamaProvider
from synthetic_experiments.agents import Persona

provider = OllamaProvider("llama2")
persona = Persona(name="Researcher", background="Curious scientist")

agent = ConversationAgent(
    provider=provider,
    persona=persona,
    role="user",  # or "assistant"
    generation_config=GenerationConfig(temperature=0.8)
)
```

### Agent Roles

- **user**: Represents the human user in conversations
- **assistant**: Represents the chatbot/advisor

For synthetic conversations (two LLMs talking), one LLM takes the "user" role and acts as a persona, while the other takes the "assistant" role.

### Using Agents

```python
# Generate a response
response = agent.respond("What do you think about solar energy?")
print(response.content)

# View conversation history
history = agent.get_history()

# Reset for a new conversation
agent.reset_conversation()

# Get statistics
stats = agent.get_statistics()
print(f"Total tokens used: {stats['total_tokens_used']}")
```

## Experiments

Experiments orchestrate multi-agent conversations.

### Creating Experiments

```python
from synthetic_experiments import Experiment

experiment = Experiment(
    name="energy_discussion",
    agents=[user_agent, advisor_agent],
    output_dir="results/energy_study"
)
```

### Running Experiments

```python
# Single conversation
result = experiment.run(
    max_turns=20,
    initial_topic="renewable energy transition",
    metadata={"condition": "baseline"}
)

# Multiple replicates
results = experiment.run_multiple(
    num_conversations=10,
    max_turns=20,
    initial_topic="renewable energy"
)
```

### Experiment Configuration

Use YAML for reproducible experiments:

```yaml
experiment:
  name: "energy_study"
  max_turns: 20
  initial_topic: "renewable energy"
  output_dir: "results/energy_study"
  metadata:
    research_question: "How do personas affect energy policy discussions?"

agents:
  - name: "user"
    role: "user"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      file: "personas/environmentalist.yaml"

  - name: "advisor"
    role: "assistant"
    provider:
      type: "ollama"
      model: "llama2"
    persona:
      file: "personas/neutral_advisor.yaml"
```

Load and run:

```python
from synthetic_experiments import load_experiment_config

experiment = load_experiment_config("config.yaml")
results = experiment.run()
```

## Data Collection

The framework automatically collects comprehensive conversation data.

### Data Structure

Results are stored in two formats:

1. **JSON**: Full conversation data (one file per conversation)
2. **CSV**: Summary statistics for easy analysis

Directory structure:
```
results/
  my_experiment/
    conversations/
      my_experiment_20240101_120000.json
      my_experiment_20240101_120100.json
    summary.csv
    turns.csv
    metadata.json
```

### Accessing Logged Data

```python
from synthetic_experiments.data import ConversationLogger, ExperimentStorage

# Load a conversation
logger = ConversationLogger.from_json("results/.../conversation.json")

# Access turns
for turn in logger.turns:
    print(f"{turn.agent_name}: {turn.message}")

# Get statistics
print(f"Total tokens: {logger.get_total_tokens()}")
print(f"Duration: {logger.get_duration()} seconds")

# Load all conversations from experiment
storage = ExperimentStorage("results/my_experiment")
conv_ids = storage.list_conversations()

# Export to CSV for R/pandas analysis
storage.export_summary_csv()  # One row per conversation
storage.export_turns_csv()     # One row per turn
```

## Analysis

The framework provides various analysis tools.

### Basic Metrics

```python
from synthetic_experiments.analysis import calculate_basic_metrics

metrics = calculate_basic_metrics(conversation_logger)

print(f"Average message length: {metrics.avg_message_length}")
print(f"User turns: {metrics.user_turns}")
print(f"Assistant turns: {metrics.assistant_turns}")
```

### Political Language Analysis

```python
from synthetic_experiments.analysis.political import (
    detect_political_language,
    analyze_conversation_polarization,
    calculate_opinion_shift
)

# Analyze a single message
analysis = detect_political_language("We need strong environmental regulations")
print(f"Liberal markers: {analysis.liberal_markers}")
print(f"Conservative markers: {analysis.conservative_markers}")

# Analyze full conversation
polarization = analyze_conversation_polarization(conversation_logger)
print(f"Average polarization: {polarization['overall_metrics']['avg_polarization']}")

# Track opinion shift
shift = calculate_opinion_shift(conversation_logger)
print(f"Opinion shift: {shift['polarization_shift']}")
```

### Custom Metrics

```python
from synthetic_experiments.analysis import MetricCalculator

calculator = MetricCalculator()

# Add custom metric
def count_technical_terms(turn):
    technical_words = ["algorithm", "data", "model", "framework"]
    return sum(word in turn.message.lower() for word in technical_words)

calculator.register_metric("technical_terms", count_technical_terms)

# Calculate for conversation
metrics = calculator.calculate_conversation_metrics(conversation_logger)
```

## Best Practices

### For Research Validity

1. **Use replicates**: Run multiple conversations per condition (5-10 recommended)
2. **Control randomness**: Set `seed` in GenerationConfig for reproducibility
3. **Document everything**: Use detailed metadata in configurations
4. **Save raw data**: Always enable `save_conversations: true`

### For Cost Efficiency

1. **Start local**: Use Ollama for development and testing
2. **Use appropriate models**: Don't use GPT-4 when GPT-3.5 suffices
3. **Limit max_tokens**: Set reasonable limits to avoid excessive costs
4. **Test on small samples**: Validate your design before full runs

### For Code Organization

1. **Use YAML configs**: Keep experiment configurations in version control
2. **Separate personas**: Store personas as individual YAML files
3. **Version your data**: Include dates/versions in output directories
4. **Document research questions**: Use metadata fields extensively

### For Analysis

1. **Export to CSV early**: Makes analysis in R/pandas much easier
2. **Calculate metrics during collection**: Don't reprocess everything later
3. **Save analysis scripts**: Keep your analysis code with your data
4. **Visualize incrementally**: Check patterns as data comes in

## Example Workflow

Here's a complete research workflow:

```python
# 1. Design: Create personas
from synthetic_experiments.agents import Persona, PersonaFactory

user_persona = PersonaFactory.create_political_user("conservative", "moderate")
advisor_persona = PersonaFactory.create_neutral_advisor()

# 2. Configure: Set up experiment
from synthetic_experiments import Experiment
from synthetic_experiments.providers import OllamaProvider
from synthetic_experiments.agents import ConversationAgent

provider = OllamaProvider("llama2")

user = ConversationAgent(provider, user_persona, role="user")
advisor = ConversationAgent(provider, advisor_persona, role="assistant")

experiment = Experiment(
    name="conservative_baseline",
    agents=[user, advisor],
    output_dir="results/pilot_study"
)

# 3. Run: Execute conversations
results = experiment.run_multiple(
    num_conversations=5,
    max_turns=20,
    initial_topic="climate policy"
)

# 4. Analyze: Extract insights
from synthetic_experiments.analysis import calculate_basic_metrics
from synthetic_experiments.analysis.political import analyze_conversation_polarization

for result in results:
    metrics = calculate_basic_metrics(result)
    polarization = analyze_conversation_polarization(result)

    print(f"Conversation {result.conversation_id}:")
    print(f"  Turns: {metrics.total_turns}")
    print(f"  Polarization: {polarization['overall_metrics']['avg_polarization']:.2f}")

# 5. Export: Save for further analysis in R/Python
from synthetic_experiments.data import ExperimentStorage

storage = ExperimentStorage("results/pilot_study")
storage.export_summary_csv("analysis/summary.csv")
storage.export_turns_csv("analysis/turns.csv")
```

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed class/method documentation
- Follow the [Political Polarization Tutorial](political_polarization_tutorial.md) for a complete research example
- Check out example scripts in `examples/` directory
