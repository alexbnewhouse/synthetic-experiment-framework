"""
Parallel execution utilities for running multiple conversations concurrently.

This module provides tools for running experiments in parallel with progress
tracking, making large factorial designs much faster to complete.

Example:
    >>> from synthetic_experiments.parallel import ParallelRunner
    >>> from synthetic_experiments import load_experiment_config
    >>>
    >>> experiment = load_experiment_config("config.yaml")
    >>> runner = ParallelRunner(max_workers=4)
    >>> results = runner.run_multiple(experiment, num_conversations=20)
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import threading
import time
import logging
import copy

logger = logging.getLogger(__name__)

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.info("tqdm not installed. Install with: pip install tqdm for progress bars")


@dataclass
class ParallelResult:
    """
    Results from parallel execution.
    
    Attributes:
        results: List of successful ConversationLogger results
        errors: List of (index, error_message) tuples for failures
        total_time: Total execution time in seconds
        avg_time_per_conversation: Average time per conversation
        num_successful: Number of successful conversations
        num_failed: Number of failed conversations
    """
    results: List[Any] = field(default_factory=list)
    errors: List[Tuple[int, str]] = field(default_factory=list)
    total_time: float = 0.0
    avg_time_per_conversation: float = 0.0
    num_successful: int = 0
    num_failed: int = 0
    
    def __str__(self) -> str:
        return (
            f"ParallelResult: {self.num_successful}/{self.num_successful + self.num_failed} "
            f"successful in {self.total_time:.1f}s "
            f"(avg {self.avg_time_per_conversation:.1f}s/conv)"
        )


class SimpleProgress:
    """Simple progress indicator when tqdm is not available."""
    
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.current = 0
        self.desc = desc
        self.lock = threading.Lock()
        self._start_time = time.time()
        
    def update(self, n: int = 1):
        with self.lock:
            self.current += n
            elapsed = time.time() - self._start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            print(f"\r{self.desc}: {self.current}/{self.total} "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", end="", flush=True)
    
    def close(self):
        print()  # New line after progress
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class ParallelRunner:
    """
    Runs multiple conversations in parallel with progress tracking.
    
    Supports both thread-based and process-based parallelism.
    Thread-based is default and works best with I/O-bound LLM API calls.
    
    Example:
        >>> runner = ParallelRunner(max_workers=4)
        >>> results = runner.run_multiple(
        ...     experiment,
        ...     num_conversations=20,
        ...     initial_topic="climate change",
        ...     show_progress=True
        ... )
        >>> print(f"Completed {results.num_successful} conversations")
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        show_progress: bool = True
    ):
        """
        Initialize parallel runner.
        
        Args:
            max_workers: Maximum number of concurrent workers
            use_processes: Use ProcessPoolExecutor instead of threads
                          (not recommended for most LLM providers)
            show_progress: Show progress bar/indicator
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.show_progress = show_progress
    
    def run_multiple(
        self,
        experiment,
        num_conversations: int,
        initial_topic: Optional[str] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        **run_kwargs
    ) -> ParallelResult:
        """
        Run multiple conversations in parallel.
        
        Args:
            experiment: Experiment instance
            num_conversations: Number of conversations to run
            initial_topic: Topic for all conversations (or use metadata_list)
            metadata_list: Optional list of metadata dicts, one per conversation
            **run_kwargs: Additional arguments passed to experiment.run()
            
        Returns:
            ParallelResult with all results and statistics
        """
        start_time = time.time()
        
        # Prepare tasks
        tasks = []
        for i in range(num_conversations):
            task_kwargs = {**run_kwargs}
            if initial_topic:
                task_kwargs['initial_topic'] = initial_topic
            if metadata_list and i < len(metadata_list):
                task_kwargs['metadata'] = metadata_list[i]
            else:
                task_kwargs.setdefault('metadata', {})
                task_kwargs['metadata']['parallel_index'] = i
            
            task_kwargs['conversation_id'] = f"{experiment.name}_{i:04d}"
            tasks.append(task_kwargs)
        
        # Execute in parallel
        results = []
        errors = []
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        # Create progress tracker
        if self.show_progress:
            if TQDM_AVAILABLE:
                progress = tqdm(total=num_conversations, desc="Running conversations")
            else:
                progress = SimpleProgress(num_conversations, "Running conversations")
        else:
            progress = None
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {}
                for i, task_kwargs in enumerate(tasks):
                    # Create a fresh copy of experiment for thread safety
                    future = executor.submit(
                        self._run_single,
                        experiment,
                        task_kwargs
                    )
                    future_to_index[future] = i
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        errors.append((index, str(e)))
                        logger.error(f"Conversation {index} failed: {e}")
                    
                    if progress:
                        progress.update(1)
        finally:
            if progress:
                progress.close()
        
        total_time = time.time() - start_time
        num_successful = len(results)
        
        return ParallelResult(
            results=results,
            errors=errors,
            total_time=total_time,
            avg_time_per_conversation=total_time / num_conversations if num_conversations > 0 else 0,
            num_successful=num_successful,
            num_failed=len(errors)
        )
    
    def _run_single(self, experiment, kwargs: Dict[str, Any]) -> Any:
        """Run a single conversation (called by worker threads)."""
        # Reset agents before each run to ensure clean state
        for agent in experiment.agents:
            agent.reset_conversation()
        return experiment.run(**kwargs)
    
    def run_conditions(
        self,
        experiment_factory: Callable[..., Any],
        conditions: List[Dict[str, Any]],
        replicates_per_condition: int = 1,
        **common_kwargs
    ) -> Dict[str, ParallelResult]:
        """
        Run multiple experimental conditions in parallel.
        
        Args:
            experiment_factory: Function that creates an experiment given condition params
            conditions: List of condition parameter dicts
            replicates_per_condition: Number of replicates per condition
            **common_kwargs: Arguments passed to all runs
            
        Returns:
            Dictionary mapping condition labels to ParallelResults
        """
        all_results = {}
        
        total_runs = len(conditions) * replicates_per_condition
        logger.info(f"Running {len(conditions)} conditions x {replicates_per_condition} "
                   f"replicates = {total_runs} total conversations")
        
        for i, condition in enumerate(conditions):
            label = condition.get('label', f'condition_{i}')
            logger.info(f"Running condition: {label}")
            
            # Create experiment for this condition
            experiment = experiment_factory(**condition)
            
            # Run replicates
            result = self.run_multiple(
                experiment,
                num_conversations=replicates_per_condition,
                **common_kwargs
            )
            
            all_results[label] = result
            logger.info(f"  {label}: {result.num_successful}/{replicates_per_condition} successful")
        
        return all_results


def run_parallel(
    experiment,
    num_conversations: int,
    max_workers: int = 4,
    show_progress: bool = True,
    **kwargs
) -> ParallelResult:
    """
    Convenience function for running parallel experiments.
    
    Args:
        experiment: Experiment instance
        num_conversations: Number of conversations
        max_workers: Number of parallel workers
        show_progress: Show progress bar
        **kwargs: Passed to experiment.run()
        
    Returns:
        ParallelResult
    """
    runner = ParallelRunner(max_workers=max_workers, show_progress=show_progress)
    return runner.run_multiple(experiment, num_conversations, **kwargs)


def run_factorial_design(
    base_experiment_factory: Callable,
    factors: Dict[str, List[Any]],
    replicates: int = 5,
    max_workers: int = 4,
    show_progress: bool = True
) -> Dict[str, ParallelResult]:
    """
    Run a full factorial experimental design.
    
    Args:
        base_experiment_factory: Function(factor_values) -> Experiment
        factors: Dictionary of factor_name -> list of levels
        replicates: Replicates per condition
        max_workers: Parallel workers
        show_progress: Show progress
        
    Returns:
        Dictionary of condition_label -> ParallelResult
        
    Example:
        >>> factors = {
        ...     "user_type": ["conservative", "liberal", "moderate"],
        ...     "advisor_type": ["neutral", "empathetic", "challenging"]
        ... }
        >>> results = run_factorial_design(
        ...     create_experiment_for_condition,
        ...     factors,
        ...     replicates=5
        ... )
    """
    import itertools
    
    # Generate all combinations
    factor_names = list(factors.keys())
    factor_levels = [factors[name] for name in factor_names]
    combinations = list(itertools.product(*factor_levels))
    
    logger.info(f"Factorial design: {len(combinations)} conditions")
    for name, levels in factors.items():
        logger.info(f"  {name}: {levels}")
    
    # Create conditions
    conditions = []
    for combo in combinations:
        condition = {name: value for name, value in zip(factor_names, combo)}
        condition['label'] = "_".join(str(v) for v in combo)
        conditions.append(condition)
    
    # Run all conditions
    runner = ParallelRunner(max_workers=max_workers, show_progress=show_progress)
    return runner.run_conditions(
        base_experiment_factory,
        conditions,
        replicates_per_condition=replicates
    )


class BatchRunner:
    """
    Batch runner with checkpointing for very large experiments.
    
    Saves progress periodically so experiments can be resumed if interrupted.
    
    Example:
        >>> runner = BatchRunner(checkpoint_dir="checkpoints")
        >>> results = runner.run_with_checkpoints(
        ...     experiment,
        ...     num_conversations=100,
        ...     checkpoint_every=10
        ... )
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_workers: int = 4
    ):
        """
        Initialize batch runner.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_workers: Number of parallel workers
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
    
    def run_with_checkpoints(
        self,
        experiment,
        num_conversations: int,
        checkpoint_every: int = 10,
        **kwargs
    ) -> ParallelResult:
        """
        Run conversations with periodic checkpointing.
        
        Args:
            experiment: Experiment instance
            num_conversations: Total conversations to run
            checkpoint_every: Save checkpoint every N conversations
            **kwargs: Passed to experiment.run()
            
        Returns:
            ParallelResult with all results
        """
        import json
        
        checkpoint_file = self.checkpoint_dir / f"{experiment.name}_checkpoint.json"
        
        # Load existing checkpoint if any
        completed = []
        start_index = 0
        
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint_data = json.load(f)
                start_index = checkpoint_data.get('completed_count', 0)
                logger.info(f"Resuming from checkpoint: {start_index} already completed")
        
        # Run remaining conversations in batches
        runner = ParallelRunner(max_workers=self.max_workers, show_progress=True)
        all_results = []
        all_errors = []
        
        for batch_start in range(start_index, num_conversations, checkpoint_every):
            batch_size = min(checkpoint_every, num_conversations - batch_start)
            
            logger.info(f"Running batch {batch_start}-{batch_start + batch_size}")
            
            result = runner.run_multiple(
                experiment,
                num_conversations=batch_size,
                **kwargs
            )
            
            all_results.extend(result.results)
            all_errors.extend(result.errors)
            
            # Save checkpoint
            checkpoint_data = {
                'completed_count': batch_start + batch_size,
                'total': num_conversations,
                'successful': len(all_results),
                'failed': len(all_errors)
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            
            logger.info(f"Checkpoint saved: {batch_start + batch_size}/{num_conversations}")
        
        # Clean up checkpoint on completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        return ParallelResult(
            results=all_results,
            errors=all_errors,
            total_time=0,  # Not tracked across batches
            avg_time_per_conversation=0,
            num_successful=len(all_results),
            num_failed=len(all_errors)
        )
