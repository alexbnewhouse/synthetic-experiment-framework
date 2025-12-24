# Advanced Features Guide

This document describes all the advanced features available in the Synthetic Experiments Framework.

## Table of Contents

1. [Conversation Visualization](#conversation-visualization)
2. [Cost Estimation](#cost-estimation)
3. [Parallel Execution](#parallel-execution)
4. [Model Comparison](#model-comparison)
5. [Streaming Support](#streaming-support)
6. [Conversation Continuation](#conversation-continuation)
7. [Smart Stopping Conditions](#smart-stopping-conditions)
8. [Statistical Export](#statistical-export)
9. [Jupyter Widgets](#jupyter-widgets)
10. [Rate Limiting](#rate-limiting)
11. [Conversation Branching](#conversation-branching)
12. [API Documentation](#api-documentation)

---

## Conversation Visualization

Plot conversation dynamics including sentiment trajectories, polarization over time, and message length patterns.

```python
from synthetic_experiments.analysis.visualization import (
    ConversationVisualizer,
    quick_plot_sentiment,
    plot_experiment_summary
)

# Quick sentiment plot
quick_plot_sentiment(conversation_logger, output_path="sentiment.png")

# Full visualizer with multiple plots
viz = ConversationVisualizer(conversation_logger)
viz.plot_sentiment_trajectory()
viz.plot_message_lengths()
viz.plot_comparison(other_logger, metric="sentiment")

# Complete experiment summary
plot_experiment_summary(results_dir="./results", output_dir="./plots")
```

**Requirements:** `pip install matplotlib seaborn`

---

## Cost Estimation

Estimate API costs before running experiments.

```python
from synthetic_experiments.costs import CostEstimator, estimate_cost

# Quick estimate
cost = estimate_cost(
    model="gpt-4",
    num_turns=20,
    avg_message_length=500
)
print(f"Estimated cost: ${cost.total_cost:.2f}")

# Detailed estimation
estimator = CostEstimator()
estimate = estimator.estimate_experiment(
    model="claude-3-opus",
    num_turns=50,
    num_agents=2,
    avg_tokens_per_message=300
)

# Compare costs across models
comparison = estimator.compare_models(
    models=["gpt-4", "claude-3-opus", "llama2"],
    num_turns=20
)
for model, est in comparison.items():
    print(f"{model}: ${est.total_cost:.4f}")
```

---

## Parallel Execution

Run multiple experiments concurrently with progress tracking.

```python
from synthetic_experiments.parallel import ParallelRunner, BatchRunner

# Run batch of experiments
runner = ParallelRunner(max_workers=4, show_progress=True)
experiments = [create_experiment(config) for config in configs]
results = runner.run_batch(experiments, max_turns=20)

# Factorial design
runner = BatchRunner()
results = runner.run_factorial_design(
    base_config={"max_turns": 20},
    factors={
        "model": ["gpt-4", "claude-3-opus"],
        "persona_type": ["conservative", "liberal", "moderate"]
    }
)

# With checkpointing for long runs
results = runner.run_batch(
    experiments,
    checkpoint_dir="./checkpoints",
    checkpoint_every=10
)
```

---

## Model Comparison

Compare conversation outcomes across different LLM models.

```python
from synthetic_experiments.comparison import ModelComparator

comparator = ModelComparator()

# Compare models with same setup
results = comparator.compare_models_simple(
    models=["gpt-4", "claude-3-opus", "llama2:70b"],
    initial_topic="Discuss climate change policy",
    max_turns=15,
    num_runs=3  # Multiple runs per model
)

# Detailed comparison with metrics
summary = comparator.summarize_comparison(results)
print(summary.to_dataframe())

# Export comparison
comparator.export_comparison(results, "model_comparison.csv")
```

---

## Streaming Support

Watch conversations unfold in real-time with callbacks.

```python
from synthetic_experiments.streaming import (
    StreamingExperiment,
    ConsolePrinter,
    EventLogger,
    stream_conversation
)

# Quick streaming to console
stream_conversation(
    experiment,
    max_turns=20,
    callbacks=[ConsolePrinter(use_colors=True)]
)

# Custom event handling
def my_handler(event):
    if event.event_type == "message":
        print(f"[{event.agent}]: {event.content[:100]}...")
        # Save to database, update UI, etc.

streaming = StreamingExperiment(
    experiment,
    callbacks=[my_handler, EventLogger("events.jsonl")]
)
result = streaming.run(max_turns=20)
```

---

## Conversation Continuation

Save and resume interrupted conversations.

```python
from synthetic_experiments.continuation import (
    ContinuableExperiment,
    save_conversation_state,
    load_conversation_state
)

# Run with auto-checkpointing
continuable = ContinuableExperiment(
    experiment,
    checkpoint_dir="./checkpoints",
    checkpoint_every=5
)
result = continuable.run(max_turns=50)

# If interrupted, resume later
resumed = ContinuableExperiment.resume("./checkpoints/my_exp_latest.json")
result = resumed.continue_conversation(additional_turns=20)

# Manual save/load
save_conversation_state(experiment, "state.json")
loaded = load_conversation_state("state.json")
```

---

## Smart Stopping Conditions

Automatically stop conversations based on semantic criteria.

```python
from synthetic_experiments.stopping import (
    StoppingConditionManager,
    TopicDriftCondition,
    SentimentExtremeCondition,
    RepetitionCondition,
    ConsensusCondition,
    create_default_stopping_conditions
)

# Use default conditions
stopper = create_default_stopping_conditions()

# Or customize
stopper = StoppingConditionManager([
    TopicDriftCondition(threshold=0.3, window=5),
    RepetitionCondition(window=6, threshold=0.75),
    SentimentExtremeCondition(threshold=0.85),
    ConsensusCondition(agreement_threshold=3)
], mode='any')  # Stop if ANY condition triggers

# Use in conversation loop
for turn in range(max_turns):
    message = agent.respond(prompt)
    messages.append({"role": "assistant", "content": message})
    
    if stopper.should_stop(messages):
        print(f"Stopping: {stopper.get_reason()}")
        break
```

---

## Statistical Export

Export data for R, SPSS, Stata, and other statistical software.

```python
from synthetic_experiments.export import (
    ConversationDataExporter,
    export_to_rds,
    export_to_spss,
    export_for_analysis
)

# Quick export to multiple formats
export_for_analysis(
    conversation_logger,
    output_dir="./analysis_data",
    formats=['csv', 'r', 'spss', 'stata'],
    include_sentiment=True
)

# Detailed export with custom options
exporter = ConversationDataExporter(conversation_logger)
exporter.to_csv("data.csv", include_metadata=True)
exporter.to_rds("data.rds", extra_vars={'treatment': 1})
exporter.to_spss("data.sav")
exporter.to_stata("data.dta")
```

**Note:** R/SPSS exports create conversion scripts. Run them with the respective software:
- R: `Rscript data_convert.R`
- SPSS: Run syntax file or `python data_convert.py` (requires pyreadstat)

---

## Jupyter Widgets

Interactive experiment configuration in Jupyter notebooks.

```python
from synthetic_experiments.widgets import (
    ExperimentConfigurator,
    ConversationViewer,
    ResultsExplorer,
    interactive_experiment
)

# Launch interactive configurator
config = interactive_experiment()
# User fills in GUI form...
experiment = config.create_experiment()

# View conversation results
viewer = ConversationViewer(conversation_logger)
viewer.display()

# Explore saved results
explorer = ResultsExplorer()
explorer.load_results("./results")
explorer.display()
```

**Requirements:** `pip install ipywidgets`

---

## Rate Limiting

Prevent API throttling and manage costs with built-in rate limiting.

```python
from synthetic_experiments.rate_limiting import (
    RateLimiter,
    RateLimitedProvider,
    create_openai_limiter,
    create_claude_limiter,
    AdaptiveRateLimiter
)

# Pre-configured limiters
limiter = create_openai_limiter(tier="tier1")  # 60 rpm, 60k tpm
limiter = create_claude_limiter(tier="standard")

# Custom limits
limiter = RateLimiter(
    requests_per_minute=30,
    tokens_per_minute=50000,
    requests_per_day=1000,
    concurrent_requests=2
)

# Wrap provider
from synthetic_experiments.providers import OpenAIProvider
provider = OpenAIProvider(model_name="gpt-4")
rate_limited = RateLimitedProvider(provider, limiter)

# Use rate-limited provider
agent = ConversationAgent(provider=rate_limited, persona=persona)

# Adaptive limiting (backs off on errors)
adaptive = AdaptiveRateLimiter(requests_per_minute=60)
# Automatically reduces rate when API returns 429 errors
```

---

## Conversation Branching

Fork conversations to explore alternative paths (counterfactual analysis).

```python
from synthetic_experiments.branching import (
    ConversationTree,
    BranchingExperiment,
    fork_conversation,
    explore_counterfactuals
)

# Run initial conversation
branching = BranchingExperiment(experiment)
result = branching.run(max_turns=10, initial_topic="Climate policy")

# Fork at turn 5 with different response
alt_branch = branching.fork_and_continue(
    turn=5,
    new_message="What if we considered nuclear energy?",
    additional_turns=10
)

# Explore multiple alternatives
branches = branching.explore_alternatives(
    turn=5,
    alternatives=[
        "Let's focus on renewable energy",
        "What about carbon capture technology?",
        "Should we discuss economic impacts?"
    ],
    additional_turns=10
)

# Compare outcomes
comparison = branching.compare_outcomes(
    metric_func=lambda msgs: calculate_polarization(msgs)
)

# Visualize tree structure
print(branching.visualize_tree())
# Output:
# ConversationTree: climate_policy
# ========================================
# [root] ROOT (turns: 10)
# ├── [a1b2c3d4] (turns: 15, fork: 5)
# ├── [e5f6g7h8] (turns: 15, fork: 5)
# └── [i9j0k1l2] (turns: 15, fork: 5)
```

---

## API Documentation

Generate comprehensive API documentation using Sphinx.

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs
python generate_docs.py

# Serve locally
python generate_docs.py --serve

# Build PDF (requires LaTeX)
python generate_docs.py --format pdf
```

Generated docs include:
- All modules with docstrings
- Class and function references
- Type hints and examples
- Cross-references and indices

---

## Combining Features

These features work together. Here's an example combining several:

```python
from synthetic_experiments import (
    Experiment, CostEstimator, ParallelRunner,
    RateLimiter, RateLimitedProvider,
    StoppingConditionManager, TopicDriftCondition,
    BranchingExperiment, export_for_analysis
)

# 1. Estimate costs first
estimator = CostEstimator()
cost = estimator.estimate_experiment("gpt-4", num_turns=30, num_conversations=10)
print(f"Estimated total cost: ${cost.total_cost:.2f}")

# 2. Set up rate limiting
limiter = RateLimiter(requests_per_minute=20, tokens_per_minute=40000)
provider = RateLimitedProvider(OpenAIProvider("gpt-4"), limiter)

# 3. Create experiment with smart stopping
stopper = StoppingConditionManager([
    TopicDriftCondition(threshold=0.3),
    RepetitionCondition(threshold=0.7)
])

# 4. Run in parallel with branching
runner = ParallelRunner(max_workers=2)
experiments = [create_experiment(c, provider) for c in configs]
results = runner.run_batch(experiments, stopping_conditions=stopper)

# 5. Branch interesting conversations
for result in results:
    if is_interesting(result):
        branching = BranchingExperiment(result.experiment)
        branching.explore_alternatives(turn=5, alternatives=alt_messages)

# 6. Export for analysis
for result in results:
    export_for_analysis(result.logger, f"./data/{result.name}")
```

---

## Installation

```bash
# Core framework
pip install -e .

# All optional dependencies
pip install -e ".[all]"

# Or specific extras
pip install -e ".[visualization]"  # matplotlib, seaborn
pip install -e ".[jupyter]"        # ipywidgets
pip install -e ".[export]"         # pyreadstat
pip install -e ".[docs]"           # sphinx
```
