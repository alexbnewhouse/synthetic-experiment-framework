"""
Model comparison utilities for running experiments across multiple models.

This module provides tools for comparing how different LLM models behave
in identical experimental conditions.

Example:
    >>> from synthetic_experiments.comparison import ModelComparator
    >>> 
    >>> comparator = ModelComparator()
    >>> results = comparator.compare(
    ...     config_path="config.yaml",
    ...     models=["llama2", "mistral", "gpt-3.5-turbo"],
    ...     replicates=5
    ... )
    >>> comparator.print_summary(results)
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import copy

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """
    Results from a single model in comparison study.
    
    Attributes:
        model: Model name
        provider_type: Provider type (ollama, claude, openai)
        conversations: List of ConversationLogger results
        metrics: Aggregated metrics across conversations
        avg_tokens_per_turn: Average tokens per turn
        avg_time_per_conversation: Average conversation time (if tracked)
        errors: List of any errors encountered
    """
    model: str
    provider_type: str
    conversations: List[Any] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    avg_tokens_per_turn: float = 0.0
    avg_time_per_conversation: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def num_successful(self) -> int:
        return len(self.conversations)
    
    @property
    def num_failed(self) -> int:
        return len(self.errors)


@dataclass
class ComparisonSummary:
    """
    Summary of multi-model comparison.
    
    Attributes:
        models: List of models compared
        results: Dictionary of model -> ModelComparisonResult
        best_by_metric: Dictionary of metric -> best model
        config_used: Configuration used for comparison
    """
    models: List[str]
    results: Dict[str, ModelComparisonResult]
    best_by_metric: Dict[str, str] = field(default_factory=dict)
    config_used: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric_comparison(self, metric: str) -> Dict[str, float]:
        """Get a specific metric across all models."""
        return {
            model: result.metrics.get(metric, 0.0)
            for model, result in self.results.items()
        }


class ModelComparator:
    """
    Compares identical experiments across different LLM models.
    
    This is useful for:
    - Understanding model-specific biases
    - Comparing response quality across models
    - Benchmarking cost vs performance tradeoffs
    
    Example:
        >>> comparator = ModelComparator()
        >>> 
        >>> # Compare from config file
        >>> results = comparator.compare_from_config(
        ...     "config.yaml",
        ...     models=["llama2", "mistral"],
        ...     replicates=5
        ... )
        >>> 
        >>> # Or compare programmatically
        >>> results = comparator.compare_experiments(
        ...     base_experiment=experiment,
        ...     model_configs=[
        ...         {"type": "ollama", "model": "llama2"},
        ...         {"type": "ollama", "model": "mistral"},
        ...         {"type": "openai", "model": "gpt-3.5-turbo"}
        ...     ],
        ...     replicates=3
        ... )
    """
    
    def __init__(self, parallel_workers: int = 4):
        """
        Initialize model comparator.
        
        Args:
            parallel_workers: Number of parallel workers for running experiments
        """
        self.parallel_workers = parallel_workers
    
    def compare_from_config(
        self,
        config_path: str,
        models: List[str],
        provider_type: str = "ollama",
        replicates: int = 5,
        **run_kwargs
    ) -> ComparisonSummary:
        """
        Compare models using a config file as template.
        
        The config file is loaded once, then the model is swapped out for each
        comparison. All agents will use the same model.
        
        Args:
            config_path: Path to experiment YAML config
            models: List of model names to compare
            provider_type: Provider type for all models (ollama, claude, openai)
            replicates: Number of replicates per model
            **run_kwargs: Additional arguments passed to experiment.run()
            
        Returns:
            ComparisonSummary with results for all models
        """
        import yaml
        from synthetic_experiments.experiments.config import load_experiment_config
        
        # Load base config
        with open(config_path) as f:
            base_config = yaml.safe_load(f)
        
        results = {}
        
        for model in models:
            logger.info(f"Running comparison for model: {model}")
            
            # Create modified config with new model
            modified_config = copy.deepcopy(base_config)
            for agent in modified_config.get('agents', []):
                agent['provider'] = {
                    'type': provider_type,
                    'model': model
                }
            
            # Save temporary config and load
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(modified_config, f)
                temp_path = f.name
            
            try:
                experiment = load_experiment_config(temp_path)
                
                # Run replicates
                model_result = self._run_model_experiment(
                    experiment, model, provider_type, replicates, **run_kwargs
                )
                results[model] = model_result
                
            except Exception as e:
                logger.error(f"Failed to run model {model}: {e}")
                results[model] = ModelComparisonResult(
                    model=model,
                    provider_type=provider_type,
                    errors=[str(e)]
                )
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
        
        # Create summary with best-by-metric
        summary = self._create_summary(models, results, base_config)
        return summary
    
    def compare_models_simple(
        self,
        models: List[Dict[str, str]],
        persona_configs: Dict[str, Any],
        max_turns: int = 10,
        initial_topic: str = "current events",
        replicates: int = 3
    ) -> ComparisonSummary:
        """
        Simple model comparison without config file.
        
        Args:
            models: List of dicts with 'type' and 'model' keys
            persona_configs: Dict with 'user' and 'assistant' persona configs
            max_turns: Max turns per conversation
            initial_topic: Conversation topic
            replicates: Replicates per model
            
        Returns:
            ComparisonSummary
            
        Example:
            >>> results = comparator.compare_models_simple(
            ...     models=[
            ...         {"type": "ollama", "model": "llama2"},
            ...         {"type": "ollama", "model": "mistral"}
            ...     ],
            ...     persona_configs={
            ...         "user": {"name": "Curious User", "background": "Student"},
            ...         "assistant": {"name": "Advisor", "background": "Expert"}
            ...     },
            ...     max_turns=10,
            ...     replicates=3
            ... )
        """
        from synthetic_experiments import Experiment
        from synthetic_experiments.agents import ConversationAgent, Persona
        from synthetic_experiments.providers import OllamaProvider, ClaudeProvider, OpenAIProvider
        
        results = {}
        
        provider_classes = {
            'ollama': OllamaProvider,
            'claude': ClaudeProvider,
            'openai': OpenAIProvider
        }
        
        for model_config in models:
            provider_type = model_config.get('type', 'ollama')
            model_name = model_config['model']
            
            logger.info(f"Running comparison for {provider_type}:{model_name}")
            
            try:
                # Create provider
                ProviderClass = provider_classes.get(provider_type, OllamaProvider)
                provider = ProviderClass(model_name=model_name)
                
                # Create agents
                user_persona = Persona(**persona_configs.get('user', {'name': 'User'}))
                asst_persona = Persona(**persona_configs.get('assistant', {'name': 'Assistant'}))
                
                user = ConversationAgent(provider=provider, persona=user_persona, role='user')
                assistant = ConversationAgent(provider=provider, persona=asst_persona, role='assistant')
                
                # Create experiment
                experiment = Experiment(
                    name=f"comparison_{model_name}",
                    agents=[user, assistant]
                )
                
                # Run replicates
                model_result = self._run_model_experiment(
                    experiment, model_name, provider_type, replicates,
                    max_turns=max_turns, initial_topic=initial_topic
                )
                results[model_name] = model_result
                
            except Exception as e:
                logger.error(f"Failed to run model {model_name}: {e}")
                results[model_name] = ModelComparisonResult(
                    model=model_name,
                    provider_type=provider_type,
                    errors=[str(e)]
                )
        
        model_names = [m['model'] for m in models]
        return self._create_summary(model_names, results, {})
    
    def _run_model_experiment(
        self,
        experiment,
        model: str,
        provider_type: str,
        replicates: int,
        **run_kwargs
    ) -> ModelComparisonResult:
        """Run experiment for a single model."""
        from synthetic_experiments.parallel import ParallelRunner
        from synthetic_experiments.analysis import calculate_basic_metrics
        
        runner = ParallelRunner(max_workers=self.parallel_workers, show_progress=True)
        parallel_result = runner.run_multiple(experiment, replicates, **run_kwargs)
        
        # Calculate aggregated metrics
        all_metrics = []
        total_tokens = 0
        total_turns = 0
        
        for conv in parallel_result.results:
            metrics = calculate_basic_metrics(conv)
            all_metrics.append(metrics)
            total_tokens += conv.get_total_tokens()
            total_turns += len(conv.turns)
        
        # Aggregate
        aggregated = {}
        if all_metrics:
            aggregated = {
                'avg_message_length': sum(m.avg_message_length for m in all_metrics) / len(all_metrics),
                'avg_turns': sum(m.total_turns for m in all_metrics) / len(all_metrics),
                'total_conversations': len(all_metrics),
            }
        
        avg_tokens = total_tokens / total_turns if total_turns > 0 else 0
        
        return ModelComparisonResult(
            model=model,
            provider_type=provider_type,
            conversations=parallel_result.results,
            metrics=aggregated,
            avg_tokens_per_turn=avg_tokens,
            avg_time_per_conversation=parallel_result.avg_time_per_conversation,
            errors=[e[1] for e in parallel_result.errors]
        )
    
    def _create_summary(
        self,
        models: List[str],
        results: Dict[str, ModelComparisonResult],
        config: Dict[str, Any]
    ) -> ComparisonSummary:
        """Create comparison summary with best-by-metric analysis."""
        best_by_metric = {}
        
        # Find best model for each metric
        metric_names = set()
        for result in results.values():
            metric_names.update(result.metrics.keys())
        
        for metric in metric_names:
            best_model = None
            best_value = None
            for model, result in results.items():
                value = result.metrics.get(metric)
                if value is not None:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_model = model
            if best_model:
                best_by_metric[metric] = best_model
        
        return ComparisonSummary(
            models=models,
            results=results,
            best_by_metric=best_by_metric,
            config_used=config
        )
    
    def print_summary(self, summary: ComparisonSummary):
        """Print formatted comparison summary."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"\nModels compared: {', '.join(summary.models)}")
        
        print("\n" + "-" * 60)
        print("Results by Model:")
        print("-" * 60)
        
        for model in summary.models:
            result = summary.results.get(model)
            if result:
                print(f"\n{model} ({result.provider_type}):")
                print(f"  Successful: {result.num_successful}")
                print(f"  Failed: {result.num_failed}")
                print(f"  Avg tokens/turn: {result.avg_tokens_per_turn:.1f}")
                print(f"  Avg time/conv: {result.avg_time_per_conversation:.2f}s")
                
                if result.metrics:
                    print("  Metrics:")
                    for metric, value in result.metrics.items():
                        print(f"    {metric}: {value:.2f}")
        
        if summary.best_by_metric:
            print("\n" + "-" * 60)
            print("Best by Metric:")
            print("-" * 60)
            for metric, model in summary.best_by_metric.items():
                print(f"  {metric}: {model}")
        
        print("\n" + "=" * 60)
    
    def export_comparison_csv(
        self,
        summary: ComparisonSummary,
        output_path: str
    ):
        """Export comparison results to CSV."""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['model', 'provider', 'successful', 'failed', 
                     'avg_tokens_per_turn', 'avg_time']
            
            # Add metric columns
            metric_names = set()
            for result in summary.results.values():
                metric_names.update(result.metrics.keys())
            metric_names = sorted(metric_names)
            header.extend(metric_names)
            
            writer.writerow(header)
            
            # Data rows
            for model in summary.models:
                result = summary.results.get(model)
                if result:
                    row = [
                        model,
                        result.provider_type,
                        result.num_successful,
                        result.num_failed,
                        result.avg_tokens_per_turn,
                        result.avg_time_per_conversation
                    ]
                    for metric in metric_names:
                        row.append(result.metrics.get(metric, ''))
                    writer.writerow(row)
        
        logger.info(f"Exported comparison to {output_path}")


def compare_models(
    models: List[str],
    config_path: str,
    provider_type: str = "ollama",
    replicates: int = 5
) -> ComparisonSummary:
    """
    Convenience function for model comparison.
    
    Args:
        models: List of model names
        config_path: Path to experiment config
        provider_type: Provider type for all models
        replicates: Number of replicates per model
        
    Returns:
        ComparisonSummary
    """
    comparator = ModelComparator()
    return comparator.compare_from_config(
        config_path, models, provider_type, replicates
    )
