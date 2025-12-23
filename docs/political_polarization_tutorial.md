# Political Polarization Study Tutorial

This tutorial walks through conducting a political polarization study using the Synthetic Experiments Framework.

## Research Question

**How do user political personas and advisor framing affect the dynamics of political discourse in LLM conversations?**

## Experimental Design

### Independent Variables

1. **User Persona** (3 levels)
   - Conservative
   - Liberal
   - Moderate

2. **Advisor Type** (3 levels)
   - Neutral (balanced information)
   - Empathetic (validating and supportive)
   - Challenging (Socratic questioning)

### Dependent Variables

- Political language markers (conservative/liberal)
- Polarization levels over conversation
- Opinion shift (early vs late)
- Agreement/disagreement patterns
- Message length and engagement

### Design

3 × 3 factorial design = 9 conditions
5 replicates per condition = 45 total conversations

## Step 1: Understanding the Personas

The framework provides pre-built political personas. Let's examine them:

```python
from synthetic_experiments.agents import Persona

# Load and inspect personas
conservative = Persona.from_yaml("examples/political_polarization/personas/conservative_user.yaml")
liberal = Persona.from_yaml("examples/political_polarization/personas/liberal_user.yaml")
moderate = Persona.from_yaml("examples/political_polarization/personas/moderate_user.yaml")

print("Conservative beliefs:", conservative.beliefs)
print("Liberal beliefs:", liberal.beliefs)
print("Moderate beliefs:", moderate.beliefs)
```

Each persona includes:
- Political orientation
- Background demographics
- Core beliefs on key issues
- Communication style

## Step 2: Running a Single Conversation

Start with one conversation to understand the output:

```bash
cd examples/political_polarization
python run_experiment.py
```

This runs the configuration in `config.yaml`:
- Conservative user persona
- Neutral advisor
- 20-turn conversation on climate policy

Examine the output:
- `results/political_polarization/conversations/` - Full JSON logs
- `results/political_polarization/summary.csv` - Quick statistics

## Step 3: Analyzing a Conversation

```python
from synthetic_experiments.data import ConversationLogger
from synthetic_experiments.analysis import calculate_basic_metrics
from synthetic_experiments.analysis.political import (
    analyze_conversation_polarization,
    calculate_opinion_shift
)

# Load conversation
conv = ConversationLogger.from_json(
    "results/political_polarization/conversations/political_polarization_baseline_20240101_120000.json"
)

# Basic metrics
metrics = calculate_basic_metrics(conv)
print(f"Total turns: {metrics.total_turns}")
print(f"Avg message length: {metrics.avg_message_length:.1f}")

# Political analysis
polarization = analyze_conversation_polarization(conv)
print(f"Average polarization: {polarization['overall_metrics']['avg_polarization']:.2f}")
print(f"User conservative score: {polarization['overall_metrics']['user_avg_conservative']:.2f}")
print(f"User liberal score: {polarization['overall_metrics']['user_avg_liberal']:.2f}")

# Opinion shift
shift = calculate_opinion_shift(conv)
print(f"Polarization shift: {shift['polarization_shift']:.2f}")
print(f"(Positive = more polarized, Negative = less polarized)")

# Examine specific turns
for turn in conv.turns:
    from synthetic_experiments.analysis.political import detect_political_language

    analysis = detect_political_language(turn.message)

    if analysis.conservative_markers or analysis.liberal_markers:
        print(f"\nTurn {turn.turn_number} ({turn.agent_name}):")
        print(f"  Message: {turn.message[:100]}...")
        print(f"  Conservative markers: {analysis.conservative_markers}")
        print(f"  Liberal markers: {analysis.liberal_markers}")
```

## Step 4: Running Multiple Replicates

Run 5 replicates of the same condition:

```bash
python run_experiment.py --replicates 5
```

This creates 5 conversations with the same configuration. Random variation in LLM responses provides natural variability.

## Step 5: Analyzing Replicates

```python
from synthetic_experiments.data import ExperimentStorage
from synthetic_experiments.analysis.political import analyze_conversation_polarization
import pandas as pd

# Load all conversations
storage = ExperimentStorage("results/political_polarization")
conv_ids = storage.list_conversations()

results = []
for conv_id in conv_ids:
    conv = storage.load_conversation(conv_id)
    polarization = analyze_conversation_polarization(conv)

    results.append({
        "conversation_id": conv_id,
        "avg_polarization": polarization["overall_metrics"]["avg_polarization"],
        "user_conservative": polarization["overall_metrics"]["user_avg_conservative"],
        "user_liberal": polarization["overall_metrics"]["user_avg_liberal"]
    })

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print("\nAcross all replicates:")
print(df.describe())
```

## Step 6: Running the Full Factorial Design

To run all 9 conditions (this takes a while!):

```bash
python run_experiment.py --full-design
```

This runs:
- 3 user personas × 3 advisor types × 3 topics × 5 replicates
- Total: 135 conversations

Results are saved to `results/full_factorial_design/`

## Step 7: Analyzing the Full Design

```python
import pandas as pd
from synthetic_experiments.data import ExperimentStorage

# Load data
storage = ExperimentStorage("results/full_factorial_design")
storage.export_summary_csv("analysis/summary.csv")
storage.export_turns_csv("analysis/turns.csv")

# Load CSV for analysis
summary = pd.read_csv("analysis/summary.csv")
turns = pd.read_csv("analysis/turns.csv")

# Analyze by condition
condition_stats = summary.groupby(["meta_user_persona", "meta_advisor_persona"]).agg({
    "total_turns": "mean",
    "total_tokens": "mean",
    "duration_seconds": "mean"
})

print("\nMean statistics by condition:")
print(condition_stats)

# Add political metrics
from synthetic_experiments.analysis.political import analyze_conversation_polarization

polarization_results = []
for conv_id in storage.list_conversations():
    conv = storage.load_conversation(conv_id)
    pol = analyze_conversation_polarization(conv)

    polarization_results.append({
        "conversation_id": conv_id,
        "avg_polarization": pol["overall_metrics"]["avg_polarization"],
        "user_avg_conservative": pol["overall_metrics"]["user_avg_conservative"],
        "user_avg_liberal": pol["overall_metrics"]["user_avg_liberal"]
    })

pol_df = pd.DataFrame(polarization_results)

# Merge with summary data
analysis_df = summary.merge(pol_df, on="conversation_id")

# Group by conditions
condition_polarization = analysis_df.groupby(
    ["meta_user_persona", "meta_advisor_persona"]
).agg({
    "avg_polarization": ["mean", "std"],
    "user_avg_conservative": ["mean", "std"],
    "user_avg_liberal": ["mean", "std"]
})

print("\nPolarization by condition:")
print(condition_polarization)
```

## Step 8: Statistical Analysis

```python
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
summary = pd.read_csv("analysis/summary.csv")

# Add political metrics (from step 7)
# ... merge polarization data ...

# Two-way ANOVA
from scipy.stats import f_oneway

# Group by user persona
conservative_pol = analysis_df[analysis_df["meta_user_persona"] == "Conservative User"]["avg_polarization"]
liberal_pol = analysis_df[analysis_df["meta_user_persona"] == "Liberal User"]["avg_polarization"]
moderate_pol = analysis_df[analysis_df["meta_user_persona"] == "Moderate User"]["avg_polarization"]

# Test effect of user persona on polarization
f_stat, p_value = f_oneway(conservative_pol, liberal_pol, moderate_pol)
print(f"Effect of user persona on polarization: F={f_stat:.2f}, p={p_value:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=analysis_df,
    x="meta_user_persona",
    y="avg_polarization",
    hue="meta_advisor_persona"
)
plt.title("Polarization by User Persona and Advisor Type")
plt.xlabel("User Persona")
plt.ylabel("Average Polarization")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("polarization_by_condition.png")
```

## Step 9: Qualitative Analysis

Examine specific conversation examples:

```python
from synthetic_experiments.data import ConversationLogger

# Find most polarized conversation
most_polarized_id = analysis_df.nlargest(1, "avg_polarization").iloc[0]["conversation_id"]
conv = ConversationLogger.from_json(
    f"results/full_factorial_design/conversations/{most_polarized_id}.json"
)

print(f"\nMost polarized conversation: {most_polarized_id}")
print(f"Metadata: {conv.metadata}")
print("\nConversation transcript:")

for turn in conv.turns:
    print(f"\n{turn.agent_name} (Turn {turn.turn_number}):")
    print(turn.message)
```

## Key Research Questions to Explore

1. **Do conservative, liberal, and moderate users show different polarization patterns?**
   - Compare `user_avg_conservative` and `user_avg_liberal` across personas

2. **Do different advisor types affect polarization differently?**
   - Compare polarization scores across neutral, empathetic, and challenging advisors

3. **Does polarization increase or decrease over conversations?**
   - Use `calculate_opinion_shift()` to track changes

4. **Do certain topics elicit more polarized discourse?**
   - Compare polarization across climate, healthcare, and immigration topics

5. **How do agreement/disagreement patterns vary?**
   - Use `count_agreement_disagreement()` on messages

## Extending the Study

### Add More Topics

Modify `run_experiment.py` to include additional topics:
```python
topics = [
    "climate policy",
    "healthcare reform",
    "immigration policy",
    "gun control",
    "education funding"
]
```

### Try Different Models

Compare Llama 2 vs Mistral vs Claude:

```yaml
# In config.yaml
agents:
  - name: "user"
    provider:
      type: "ollama"
      model: "mistral"  # or claude/gpt-4
```

### Add Custom Personas

Create your own persona variations:
```yaml
name: "Independent Voter"
background: "Fiscally conservative, socially liberal"
political_orientation: "independent"
beliefs:
  fiscal: "Balanced budgets are important"
  social: "Individual freedom in personal matters"
```

### Implement New Metrics

```python
from synthetic_experiments.analysis import MetricCalculator

calculator = MetricCalculator()

def count_facts_vs_opinions(turn):
    # Implement logic to distinguish factual statements from opinions
    fact_markers = ["according to", "data shows", "research indicates"]
    opinion_markers = ["I think", "I believe", "in my opinion"]

    facts = sum(marker in turn.message.lower() for marker in fact_markers)
    opinions = sum(marker in turn.message.lower() for marker in opinion_markers)

    return {"facts": facts, "opinions": opinions}

calculator.register_metric("fact_opinion", count_facts_vs_opinions)
```

## Step 9: Measuring Polarization with Pre/Post Surveys

For rigorous measurement of how conversations affect LLM "advisor" attitudes, the framework provides a survey system that measures **affective polarization** (feelings toward political groups) and **ideological polarization** (strength of policy positions).

### Key Design Principle: Context Isolation

A critical methodological challenge is ensuring that pre-survey responses don't "leak" into the conversation or post-survey. The `SurveyAdministrator` solves this by:

1. Creating a **fresh LLM instance** for each survey administration
2. Using the **same random seed** to ensure identical initialization
3. Each survey question gets its own isolated context window

```python
from synthetic_experiments.analysis.survey import (
    PolarizationSurvey,
    SurveyAdministrator,
    calculate_polarization_delta
)
from synthetic_experiments.providers.ollama import OllamaProvider
from synthetic_experiments.agents import Persona

# Load advisor persona
advisor = Persona.from_yaml("examples/political_polarization/personas/neutral_advisor.yaml")

# Create survey administrator with a fixed seed
admin = SurveyAdministrator(
    provider_class=OllamaProvider,
    provider_kwargs={"model_name": "llama3.2"},
    persona=advisor,
    seed=42  # Same seed ensures reproducible LLM initialization
)

# Administer PRE-survey (fresh LLM instance)
pre_results = admin.administer_survey(survey_type="pre")
print(f"Pre-survey affective score: {pre_results.affective_score:.3f}")
print(f"Pre-survey ideological score: {pre_results.ideological_score:.3f}")

# ========================================
# CONDUCT CONVERSATION EXPERIMENT HERE
# (Use a SEPARATE agent with the same seed)
# ========================================

# Administer POST-survey (fresh LLM instance, same seed)
post_results = admin.administer_survey(survey_type="post")
print(f"Post-survey affective score: {post_results.affective_score:.3f}")

# Calculate treatment effect
delta = calculate_polarization_delta(pre_results, post_results)
print(f"\n=== Treatment Effects ===")
print(f"Affective polarization change: {delta.affective_delta:+.3f}")
print(f"Ideological polarization change: {delta.ideological_delta:+.3f}")
print(f"Overall change: {delta.overall_delta:+.3f}")
```

### Survey Questions

The `PolarizationSurvey` includes validated question instruments:

**Affective Polarization Questions (6 total):**
- Feeling thermometers toward liberals/conservatives
- Trust attributions toward political groups
- Social distance measures (comfort with outgroup friends)

**Ideological Polarization Questions (6 total):**
- Policy positions on climate, healthcare, immigration, taxes, guns
- Certainty about political views

### Experimental Protocol

For a rigorous study with treatment (user political slant):

```python
from synthetic_experiments.analysis.survey import create_survey_experiment_protocol

# Create protocol
protocol = create_survey_experiment_protocol(
    provider_class=OllamaProvider,
    provider_kwargs={"model_name": "llama3.2"},
    advisor_persona=advisor,
    seed=42
)

# The protocol documents the steps:
for step, desc in protocol["protocol"].items():
    print(f"{step}: {desc}")
```

### Running a Full Pre/Post Study

```python
from synthetic_experiments.experiments import Experiment
import json

results = []

# For each condition (e.g., liberal user, conservative user, moderate user)
for user_persona_path in ["liberal_user.yaml", "conservative_user.yaml", "moderate_user.yaml"]:
    # Load personas
    user = Persona.from_yaml(f"personas/{user_persona_path}")
    advisor = Persona.from_yaml("personas/neutral_advisor.yaml")
    
    # Create fresh administrator for this condition
    admin = SurveyAdministrator(
        provider_class=OllamaProvider,
        provider_kwargs={"model_name": "llama3.2"},
        persona=advisor,
        seed=42
    )
    
    # 1. Pre-survey
    pre_results = admin.administer_survey("pre")
    
    # 2. Run conversation (use a separate agent with same seed)
    # ... experiment code here ...
    
    # 3. Post-survey  
    post_results = admin.administer_survey("post")
    
    # 4. Calculate delta
    delta = calculate_polarization_delta(pre_results, post_results)
    
    results.append({
        "condition": user_persona_path,
        "pre_affective": pre_results.affective_score,
        "post_affective": post_results.affective_score,
        "delta_affective": delta.affective_delta,
        "delta_ideological": delta.ideological_delta,
        "delta_overall": delta.overall_delta
    })

# Export results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("survey_results.csv", index=False)
print(df)
```

### Customizing Survey Questions

You can add custom questions or use only a subset:

```python
from synthetic_experiments.analysis.survey import SurveyQuestion, PolarizationType

# Custom question
custom_q = SurveyQuestion(
    id="custom_media_trust",
    text="How much do you trust mainstream media to report accurately? (1=Not at all, 7=Completely)",
    scale_min=1,
    scale_max=7,
    polarization_type=PolarizationType.AFFECTIVE
)

# Create survey with custom questions
survey = PolarizationSurvey(
    include_affective=True,
    include_ideological=False,  # Skip ideological questions
    custom_questions=[custom_q]
)

# Use in administrator
admin = SurveyAdministrator(
    provider_class=OllamaProvider,
    provider_kwargs={"model_name": "llama3.2"},
    persona=advisor,
    seed=42,
    survey=survey
)
```

## Publishing Your Results

When publishing research using this framework:

1. **Save all configurations**: Include YAML files in supplementary materials
2. **Report model details**: Document exact model versions used
3. **Share aggregated data**: Export CSV files for transparency
4. **Document random seeds**: For reproducibility (critical for surveys!)
5. **Report survey valid response rates**: Survey results include `valid_response_rate`
6. **Cite the framework**: Reference this package in methods section

## Conclusion

This framework enables rigorous social science research on LLM conversations. The political polarization study demonstrates how to:

- Design factorial experiments
- Collect comprehensive data
- **Measure treatment effects with pre/post surveys**
- Analyze both quantitative metrics and qualitative patterns
- Export results for statistical analysis
- Scale from pilot studies to full experiments

The same principles apply to other research domains - simply adapt the personas and metrics to your research questions.
