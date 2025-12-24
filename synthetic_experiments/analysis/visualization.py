"""
Visualization tools for conversation analysis.

This module provides plotting utilities for visualizing conversation dynamics,
polarization trends, sentiment trajectories, and experimental comparisons.

Requires matplotlib (optional dependency).

Example:
    >>> from synthetic_experiments.analysis.visualization import ConversationVisualizer
    >>> from synthetic_experiments.data import ConversationLogger
    >>>
    >>> conversation = ConversationLogger.from_json("conversation.json")
    >>> viz = ConversationVisualizer()
    >>> viz.plot_sentiment_trajectory(conversation)
    >>> viz.save("sentiment_plot.png")
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try importing matplotlib - it's optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


class ConversationVisualizer:
    """
    Visualization tools for conversation analysis.
    
    Provides methods for plotting various aspects of conversations:
    - Sentiment trajectories over turns
    - Polarization dynamics
    - Message length patterns
    - Comparison across conditions
    
    Example:
        >>> viz = ConversationVisualizer(figsize=(12, 6))
        >>> viz.plot_sentiment_trajectory(conversation)
        >>> viz.show()
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        style: str = "seaborn-v0_8-whitegrid",
        dpi: int = 100
    ):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size (width, height) in inches
            style: matplotlib style to use
            dpi: Resolution for saved figures
        """
        _check_matplotlib()
        
        self.figsize = figsize
        self.dpi = dpi
        self._current_fig = None
        self._current_axes = None
        
        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                pass  # Use default style
    
    def _create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[Any, Any]:
        """Create a new figure."""
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize or self.figsize
        )
        self._current_fig = fig
        self._current_axes = axes
        return fig, axes
    
    def plot_sentiment_trajectory(
        self,
        conversation,
        title: str = "Sentiment Trajectory",
        show_agents: bool = True,
        smoothing: int = 0
    ) -> Any:
        """
        Plot sentiment over conversation turns.
        
        Args:
            conversation: ConversationLogger instance
            title: Plot title
            show_agents: Color-code by agent
            smoothing: Window size for moving average (0 = no smoothing)
            
        Returns:
            matplotlib figure
        """
        from synthetic_experiments.analysis.metrics import calculate_sentiment_simple
        
        fig, ax = self._create_figure()
        
        turns = conversation.turns
        sentiments = []
        agent_names = []
        
        for turn in turns:
            sentiment = calculate_sentiment_simple(turn.message)
            sentiments.append(sentiment)
            agent_names.append(turn.agent_name)
        
        x = list(range(1, len(sentiments) + 1))
        
        if show_agents:
            # Get unique agents and assign colors
            unique_agents = list(set(agent_names))
            colors = plt.cm.tab10(range(len(unique_agents)))
            agent_colors = {agent: colors[i] for i, agent in enumerate(unique_agents)}
            
            # Plot each point with agent color
            for i, (xi, yi) in enumerate(zip(x, sentiments)):
                ax.scatter(xi, yi, c=[agent_colors[agent_names[i]]], s=50, zorder=5)
            
            # Add line connecting points
            ax.plot(x, sentiments, 'k-', alpha=0.3, linewidth=1)
            
            # Add legend
            patches = [mpatches.Patch(color=c, label=a) for a, c in agent_colors.items()]
            ax.legend(handles=patches, loc='best')
        else:
            ax.plot(x, sentiments, 'b-o', markersize=6)
        
        # Apply smoothing if requested
        if smoothing > 1 and len(sentiments) > smoothing:
            smoothed = self._moving_average(sentiments, smoothing)
            smooth_x = list(range(smoothing, len(sentiments) + 1))
            ax.plot(smooth_x, smoothed, 'r-', linewidth=2, label=f'{smoothing}-turn avg')
            ax.legend(loc='best')
        
        ax.set_xlabel("Turn")
        ax.set_ylabel("Sentiment Score")
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_polarization_trajectory(
        self,
        conversation,
        title: str = "Polarization Over Time",
        metric: str = "political_ratio"
    ) -> Any:
        """
        Plot polarization metrics over conversation turns.
        
        Args:
            conversation: ConversationLogger instance
            title: Plot title
            metric: Which metric to plot ('political_ratio', 'liberal', 'conservative')
            
        Returns:
            matplotlib figure
        """
        from synthetic_experiments.analysis.political import detect_political_language
        
        fig, ax = self._create_figure()
        
        turns = conversation.turns
        values = []
        
        for turn in turns:
            analysis = detect_political_language(turn.message)
            
            if metric == "political_ratio":
                total = analysis.liberal_count + analysis.conservative_count
                if total > 0:
                    # Positive = more liberal, negative = more conservative
                    values.append((analysis.liberal_count - analysis.conservative_count) / total)
                else:
                    values.append(0)
            elif metric == "liberal":
                values.append(analysis.liberal_count)
            elif metric == "conservative":
                values.append(analysis.conservative_count)
            else:
                values.append(analysis.political_intensity)
        
        x = list(range(1, len(values) + 1))
        
        # Color by value
        colors = ['blue' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
        ax.bar(x, values, color=colors, alpha=0.7)
        ax.plot(x, values, 'k-', alpha=0.5)
        
        ax.set_xlabel("Turn")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_message_lengths(
        self,
        conversation,
        title: str = "Message Lengths",
        by_agent: bool = True
    ) -> Any:
        """
        Plot message lengths over turns.
        
        Args:
            conversation: ConversationLogger instance
            title: Plot title
            by_agent: Group bars by agent
            
        Returns:
            matplotlib figure
        """
        fig, ax = self._create_figure()
        
        turns = conversation.turns
        lengths = [len(turn.message) for turn in turns]
        x = list(range(1, len(lengths) + 1))
        
        if by_agent:
            agent_names = [turn.agent_name for turn in turns]
            unique_agents = list(set(agent_names))
            colors = plt.cm.tab10(range(len(unique_agents)))
            agent_colors = {agent: colors[i] for i, agent in enumerate(unique_agents)}
            
            bar_colors = [agent_colors[name] for name in agent_names]
            ax.bar(x, lengths, color=bar_colors, alpha=0.8)
            
            patches = [mpatches.Patch(color=c, label=a) for a, c in agent_colors.items()]
            ax.legend(handles=patches, loc='best')
        else:
            ax.bar(x, lengths, alpha=0.8)
        
        ax.set_xlabel("Turn")
        ax.set_ylabel("Message Length (characters)")
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(
        self,
        conversations: List,
        labels: List[str],
        metric: str = "avg_sentiment",
        title: str = "Condition Comparison"
    ) -> Any:
        """
        Compare metrics across multiple conversations/conditions.
        
        Args:
            conversations: List of ConversationLogger instances
            labels: Labels for each conversation
            metric: Metric to compare ('avg_sentiment', 'total_turns', 
                   'avg_length', 'polarization')
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        from synthetic_experiments.analysis import calculate_basic_metrics
        from synthetic_experiments.analysis.political import analyze_conversation_polarization
        
        fig, ax = self._create_figure()
        
        values = []
        for conv in conversations:
            if metric == "avg_sentiment":
                from synthetic_experiments.analysis.metrics import calculate_sentiment_simple
                sentiments = [calculate_sentiment_simple(t.message) for t in conv.turns]
                values.append(sum(sentiments) / len(sentiments) if sentiments else 0)
            elif metric == "total_turns":
                values.append(len(conv.turns))
            elif metric == "avg_length":
                lengths = [len(t.message) for t in conv.turns]
                values.append(sum(lengths) / len(lengths) if lengths else 0)
            elif metric == "polarization":
                pol = analyze_conversation_polarization(conv)
                values.append(pol.get('overall_metrics', {}).get('avg_polarization', 0))
            else:
                metrics = calculate_basic_metrics(conv)
                values.append(getattr(metrics, metric, 0))
        
        x = range(len(labels))
        colors = plt.cm.tab10(range(len(labels)))
        
        ax.bar(x, values, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_survey_results(
        self,
        pre_results,
        post_results,
        title: str = "Pre/Post Survey Comparison"
    ) -> Any:
        """
        Plot pre/post survey comparison.
        
        Args:
            pre_results: SurveyResults from pre-survey
            post_results: SurveyResults from post-survey
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        fig, axes = self._create_figure(nrows=1, ncols=2, figsize=(12, 5))
        
        # Left plot: Overall scores
        ax1 = axes[0]
        categories = ['Ideological', 'Affective', 'Overall']
        pre_scores = [
            pre_results.ideological_score or 0,
            pre_results.affective_score or 0,
            pre_results.overall_score
        ]
        post_scores = [
            post_results.ideological_score or 0,
            post_results.affective_score or 0,
            post_results.overall_score
        ]
        
        x = range(len(categories))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], pre_scores, width, label='Pre', alpha=0.8)
        ax1.bar([i + width/2 for i in x], post_scores, width, label='Post', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.set_ylabel("Score")
        ax1.set_title("Score Comparison")
        ax1.legend()
        ax1.set_ylim(0, max(max(pre_scores), max(post_scores)) * 1.2)
        
        # Right plot: Delta
        ax2 = axes[1]
        deltas = [post - pre for pre, post in zip(pre_scores, post_scores)]
        colors = ['green' if d > 0 else 'red' for d in deltas]
        
        ax2.bar(categories, deltas, color=colors, alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_ylabel("Change (Post - Pre)")
        ax2.set_title("Treatment Effect")
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_multi_conversation_trajectories(
        self,
        conversations: List,
        labels: List[str],
        metric: str = "sentiment",
        title: str = "Multi-Conversation Trajectories"
    ) -> Any:
        """
        Plot trajectories of multiple conversations on same axes.
        
        Args:
            conversations: List of ConversationLogger instances
            labels: Labels for each conversation
            metric: 'sentiment' or 'length'
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        fig, ax = self._create_figure()
        
        colors = plt.cm.tab10(range(len(conversations)))
        
        for i, (conv, label) in enumerate(zip(conversations, labels)):
            if metric == "sentiment":
                from synthetic_experiments.analysis.metrics import calculate_sentiment_simple
                values = [calculate_sentiment_simple(t.message) for t in conv.turns]
            else:  # length
                values = [len(t.message) for t in conv.turns]
            
            x = list(range(1, len(values) + 1))
            ax.plot(x, values, '-o', color=colors[i], label=label, 
                   markersize=4, alpha=0.8)
        
        ax.set_xlabel("Turn")
        ax.set_ylabel(metric.title())
        ax.set_title(title)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average."""
        result = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            result.append(avg)
        return result
    
    def show(self):
        """Display the current figure."""
        _check_matplotlib()
        plt.show()
    
    def save(
        self,
        filepath: Union[str, Path],
        dpi: Optional[int] = None
    ):
        """
        Save the current figure to file.
        
        Args:
            filepath: Output path (supports .png, .pdf, .svg)
            dpi: Resolution (uses instance default if not specified)
        """
        if self._current_fig:
            self._current_fig.savefig(
                filepath,
                dpi=dpi or self.dpi,
                bbox_inches='tight'
            )
            logger.info(f"Saved figure to {filepath}")
        else:
            raise ValueError("No figure to save. Create a plot first.")
    
    def close(self):
        """Close the current figure."""
        if self._current_fig:
            plt.close(self._current_fig)
            self._current_fig = None
            self._current_axes = None


def plot_experiment_summary(
    storage,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive visualization summary for an experiment.
    
    Args:
        storage: ExperimentStorage instance
        output_dir: Directory to save plots (optional)
        
    Returns:
        Dictionary of figure objects
    """
    _check_matplotlib()
    
    from synthetic_experiments.data import ExperimentStorage, ConversationLogger
    
    viz = ConversationVisualizer()
    figures = {}
    
    # Load all conversations
    conv_ids = storage.list_conversations()
    conversations = []
    for conv_id in conv_ids[:10]:  # Limit to first 10
        try:
            conv = storage.load_conversation(conv_id)
            conversations.append(conv)
        except Exception as e:
            logger.warning(f"Failed to load {conv_id}: {e}")
    
    if not conversations:
        logger.warning("No conversations found")
        return figures
    
    # Generate summary plots
    try:
        # Sentiment comparison
        labels = [f"Conv {i+1}" for i in range(len(conversations))]
        fig = viz.plot_comparison(
            conversations, labels,
            metric="avg_sentiment",
            title="Average Sentiment by Conversation"
        )
        figures['sentiment_comparison'] = fig
        if output_dir:
            viz.save(Path(output_dir) / "sentiment_comparison.png")
        viz.close()
        
        # Turn counts
        fig = viz.plot_comparison(
            conversations, labels,
            metric="total_turns",
            title="Total Turns by Conversation"
        )
        figures['turns_comparison'] = fig
        if output_dir:
            viz.save(Path(output_dir) / "turns_comparison.png")
        viz.close()
        
        # First conversation trajectory
        if conversations:
            fig = viz.plot_sentiment_trajectory(
                conversations[0],
                title="Sentiment Trajectory (First Conversation)"
            )
            figures['first_sentiment_trajectory'] = fig
            if output_dir:
                viz.save(Path(output_dir) / "sentiment_trajectory.png")
            viz.close()
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return figures


# Convenience functions
def quick_plot_sentiment(conversation, save_path: Optional[str] = None):
    """Quick sentiment trajectory plot."""
    _check_matplotlib()
    viz = ConversationVisualizer()
    viz.plot_sentiment_trajectory(conversation)
    if save_path:
        viz.save(save_path)
    else:
        viz.show()
    viz.close()


def quick_plot_polarization(conversation, save_path: Optional[str] = None):
    """Quick polarization trajectory plot."""
    _check_matplotlib()
    viz = ConversationVisualizer()
    viz.plot_polarization_trajectory(conversation)
    if save_path:
        viz.save(save_path)
    else:
        viz.show()
    viz.close()
