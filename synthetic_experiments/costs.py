"""
Cost estimation utilities for LLM API usage.

This module provides tools to estimate the cost of running experiments
before executing them, helping researchers budget their API usage.

Example:
    >>> from synthetic_experiments.costs import CostEstimator, estimate_experiment_cost
    >>> 
    >>> # Quick estimate
    >>> cost = estimate_experiment_cost(
    ...     model="claude-3-sonnet-20240229",
    ...     max_turns=20,
    ...     num_conversations=10,
    ...     avg_tokens_per_turn=500
    ... )
    >>> print(f"Estimated cost: ${cost:.2f}")
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Pricing as of late 2024 (per 1M tokens)
# These should be updated periodically
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models (per 1M tokens)
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-16k": {"input": 0.50, "output": 1.50},
    
    # Claude models (per 1M tokens)
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    
    # Ollama models (free - local)
    "llama2": {"input": 0.0, "output": 0.0},
    "llama3": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "mixtral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "phi": {"input": 0.0, "output": 0.0},
    "qwen": {"input": 0.0, "output": 0.0},
}


@dataclass
class CostEstimate:
    """
    Detailed cost estimate for an experiment.
    
    Attributes:
        model: Model name used for estimation
        total_cost: Total estimated cost in USD
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        total_input_tokens: Estimated input tokens
        total_output_tokens: Estimated output tokens
        num_conversations: Number of conversations
        turns_per_conversation: Turns per conversation
        breakdown: Detailed breakdown by component
        warnings: Any warnings about the estimate
    """
    model: str
    total_cost: float
    input_cost: float
    output_cost: float
    total_input_tokens: int
    total_output_tokens: int
    num_conversations: int
    turns_per_conversation: int
    breakdown: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = [
            f"Cost Estimate for {self.model}",
            "=" * 40,
            f"Conversations: {self.num_conversations}",
            f"Turns per conversation: {self.turns_per_conversation}",
            f"",
            f"Token Estimates:",
            f"  Input tokens:  {self.total_input_tokens:,}",
            f"  Output tokens: {self.total_output_tokens:,}",
            f"",
            f"Cost Breakdown:",
            f"  Input cost:  ${self.input_cost:.4f}",
            f"  Output cost: ${self.output_cost:.4f}",
            f"  -----------",
            f"  Total:       ${self.total_cost:.4f}",
        ]
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "total_cost": self.total_cost,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_conversations": self.num_conversations,
            "turns_per_conversation": self.turns_per_conversation,
            "breakdown": self.breakdown,
            "warnings": self.warnings
        }


class CostEstimator:
    """
    Estimates costs for LLM API experiments.
    
    Takes into account:
    - Model pricing (input vs output tokens)
    - System prompts (repeated each turn)
    - Conversation history growth
    - Number of agents (each needs their own API call)
    
    Example:
        >>> estimator = CostEstimator()
        >>> estimate = estimator.estimate_experiment(
        ...     model="claude-3-sonnet-20240229",
        ...     max_turns=20,
        ...     num_conversations=10,
        ...     num_agents=2
        ... )
        >>> print(estimate)
    """
    
    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize cost estimator.
        
        Args:
            custom_pricing: Override default pricing for models
        """
        self.pricing = {**MODEL_PRICING}
        if custom_pricing:
            self.pricing.update(custom_pricing)
    
    def get_model_pricing(self, model: str) -> Tuple[float, float]:
        """
        Get pricing for a model.
        
        Args:
            model: Model name
            
        Returns:
            Tuple of (input_price, output_price) per 1M tokens
        """
        # Try exact match first
        if model in self.pricing:
            p = self.pricing[model]
            return p["input"], p["output"]
        
        # Try partial match
        model_lower = model.lower()
        for name, p in self.pricing.items():
            if name.lower() in model_lower or model_lower in name.lower():
                return p["input"], p["output"]
        
        # Unknown model - return None to trigger warning
        return None, None
    
    def estimate_experiment(
        self,
        model: str,
        max_turns: int = 20,
        num_conversations: int = 1,
        num_agents: int = 2,
        avg_system_prompt_tokens: int = 200,
        avg_message_tokens: int = 150,
        include_history: bool = True
    ) -> CostEstimate:
        """
        Estimate cost for an experiment.
        
        Args:
            model: Model name
            max_turns: Maximum turns per conversation
            num_conversations: Number of conversations to run
            num_agents: Number of agents in each conversation
            avg_system_prompt_tokens: Average system prompt size
            avg_message_tokens: Average tokens per message
            include_history: Whether conversation history is included in each call
            
        Returns:
            CostEstimate with detailed breakdown
        """
        warnings = []
        
        # Get pricing
        input_price, output_price = self.get_model_pricing(model)
        
        if input_price is None:
            warnings.append(f"Unknown model '{model}'. Assuming free (local model).")
            input_price, output_price = 0.0, 0.0
        
        if input_price == 0 and output_price == 0:
            warnings.append("This appears to be a local model. No API costs expected.")
        
        # Calculate tokens per conversation
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Each turn involves calls from multiple agents
        turns_per_agent = max_turns // num_agents
        
        for turn in range(max_turns):
            # Input tokens: system prompt + history + current message
            history_tokens = turn * avg_message_tokens if include_history else 0
            input_tokens = avg_system_prompt_tokens + history_tokens + avg_message_tokens
            total_input_tokens += input_tokens
            
            # Output tokens: one message
            total_output_tokens += avg_message_tokens
        
        # Scale by number of conversations
        total_input_tokens *= num_conversations
        total_output_tokens *= num_conversations
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (total_input_tokens / 1_000_000) * input_price
        output_cost = (total_output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost
        
        # Create breakdown
        breakdown = {
            "per_conversation": total_cost / num_conversations if num_conversations > 0 else 0,
            "per_turn": total_cost / (max_turns * num_conversations) if max_turns > 0 else 0,
            "system_prompts": (avg_system_prompt_tokens * max_turns * num_conversations / 1_000_000) * input_price,
            "history_growth": ((total_input_tokens - avg_system_prompt_tokens * max_turns * num_conversations) / 1_000_000) * input_price
        }
        
        return CostEstimate(
            model=model,
            total_cost=total_cost,
            input_cost=input_cost,
            output_cost=output_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            num_conversations=num_conversations,
            turns_per_conversation=max_turns,
            breakdown=breakdown,
            warnings=warnings
        )
    
    def estimate_from_config(
        self,
        config_path: str,
        num_conversations: int = 1
    ) -> CostEstimate:
        """
        Estimate cost from a YAML configuration file.
        
        Args:
            config_path: Path to experiment config
            num_conversations: Number of conversations to run
            
        Returns:
            CostEstimate
        """
        import yaml
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        exp_config = config.get("experiment", {})
        agents = config.get("agents", [])
        
        max_turns = exp_config.get("max_turns", 20)
        num_agents = len(agents)
        
        # Get model from first agent
        model = "unknown"
        if agents:
            provider = agents[0].get("provider", {})
            model = provider.get("model", "unknown")
        
        return self.estimate_experiment(
            model=model,
            max_turns=max_turns,
            num_conversations=num_conversations,
            num_agents=num_agents
        )
    
    def compare_models(
        self,
        models: List[str],
        max_turns: int = 20,
        num_conversations: int = 10
    ) -> Dict[str, CostEstimate]:
        """
        Compare costs across multiple models.
        
        Args:
            models: List of model names to compare
            max_turns: Turns per conversation
            num_conversations: Number of conversations
            
        Returns:
            Dictionary mapping model names to estimates
        """
        estimates = {}
        for model in models:
            estimates[model] = self.estimate_experiment(
                model=model,
                max_turns=max_turns,
                num_conversations=num_conversations
            )
        return estimates
    
    def print_comparison(
        self,
        models: List[str],
        max_turns: int = 20,
        num_conversations: int = 10
    ):
        """Print a formatted comparison of model costs."""
        estimates = self.compare_models(models, max_turns, num_conversations)
        
        print(f"\nCost Comparison ({num_conversations} conversations, {max_turns} turns each)")
        print("=" * 60)
        print(f"{'Model':<35} {'Total Cost':>12} {'Per Conv':>12}")
        print("-" * 60)
        
        # Sort by cost
        sorted_estimates = sorted(estimates.items(), key=lambda x: x[1].total_cost)
        
        for model, est in sorted_estimates:
            per_conv = est.total_cost / num_conversations if num_conversations > 0 else 0
            print(f"{model:<35} ${est.total_cost:>10.4f} ${per_conv:>10.4f}")
        
        print("-" * 60)
        
        # Show cheapest
        if sorted_estimates:
            cheapest = sorted_estimates[0]
            print(f"\n✓ Cheapest: {cheapest[0]} (${cheapest[1].total_cost:.4f})")


def estimate_experiment_cost(
    model: str,
    max_turns: int = 20,
    num_conversations: int = 1,
    avg_tokens_per_turn: int = 300,
    num_agents: int = 2
) -> float:
    """
    Quick function to estimate experiment cost.
    
    Args:
        model: Model name
        max_turns: Turns per conversation
        num_conversations: Number of conversations
        avg_tokens_per_turn: Average tokens per turn (input + output)
        num_agents: Number of agents
        
    Returns:
        Estimated total cost in USD
    """
    estimator = CostEstimator()
    estimate = estimator.estimate_experiment(
        model=model,
        max_turns=max_turns,
        num_conversations=num_conversations,
        num_agents=num_agents,
        avg_message_tokens=avg_tokens_per_turn // 2
    )
    return estimate.total_cost


def list_supported_models() -> Dict[str, Dict[str, float]]:
    """Return dictionary of supported models and their pricing."""
    return MODEL_PRICING.copy()


def get_model_cost_per_1k_tokens(model: str) -> Tuple[float, float]:
    """
    Get cost per 1000 tokens for a model.
    
    Returns:
        Tuple of (input_cost, output_cost) per 1K tokens
    """
    estimator = CostEstimator()
    input_price, output_price = estimator.get_model_pricing(model)
    if input_price is None:
        return 0.0, 0.0
    return input_price / 1000, output_price / 1000
