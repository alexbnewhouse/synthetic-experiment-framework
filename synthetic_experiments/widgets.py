"""
Jupyter notebook widgets for interactive experiment configuration.

This module provides ipywidgets-based interfaces for configuring
and running experiments interactively in Jupyter notebooks.

Example:
    >>> from synthetic_experiments.widgets import ExperimentConfigurator
    >>> 
    >>> # Display interactive configurator
    >>> config = ExperimentConfigurator()
    >>> config.display()
    >>> 
    >>> # After configuration, run experiment
    >>> experiment = config.create_experiment()
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

# Check for ipywidgets availability
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None


def require_widgets(func):
    """Decorator to check ipywidgets availability."""
    def wrapper(*args, **kwargs):
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for interactive widgets. "
                "Install with: pip install ipywidgets"
            )
        return func(*args, **kwargs)
    return wrapper


@require_widgets
class ExperimentConfigurator:
    """
    Interactive widget for configuring experiments.
    
    Provides a GUI interface for:
    - Selecting LLM providers and models
    - Defining agent personas
    - Setting experiment parameters
    - Running and monitoring experiments
    
    Example:
        >>> configurator = ExperimentConfigurator()
        >>> configurator.display()  # Shows interactive form
        >>> 
        >>> # User fills in configuration...
        >>> 
        >>> # Get the configured experiment
        >>> experiment = configurator.create_experiment()
    """
    
    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize configurator.
        
        Args:
            defaults: Default configuration values
        """
        self.defaults = defaults or {}
        self._config = {}
        self._output = widgets.Output()
        self._build_widgets()
    
    def _build_widgets(self):
        """Build all widget components."""
        # Experiment Settings
        self.name_input = widgets.Text(
            value=self.defaults.get('name', 'my_experiment'),
            description='Name:',
            style={'description_width': '120px'}
        )
        
        self.max_turns = widgets.IntSlider(
            value=self.defaults.get('max_turns', 20),
            min=1, max=100, step=1,
            description='Max Turns:',
            style={'description_width': '120px'}
        )
        
        self.topic_input = widgets.Textarea(
            value=self.defaults.get('initial_topic', 'Discuss the impact of technology on society.'),
            description='Topic:',
            layout=widgets.Layout(width='100%', height='80px'),
            style={'description_width': '120px'}
        )
        
        # Provider Selection
        self.provider_dropdown = widgets.Dropdown(
            options=['ollama', 'openai', 'claude'],
            value=self.defaults.get('provider', 'ollama'),
            description='Provider:',
            style={'description_width': '120px'}
        )
        
        self.model_input = widgets.Text(
            value=self.defaults.get('model', 'llama2'),
            description='Model:',
            style={'description_width': '120px'}
        )
        
        # Agent 1 Configuration
        self.agent1_name = widgets.Text(
            value='User',
            description='Agent 1 Name:',
            style={'description_width': '120px'}
        )
        
        self.agent1_persona = widgets.Textarea(
            value='A curious person interested in discussing various topics.',
            description='Persona:',
            layout=widgets.Layout(width='100%', height='100px'),
            style={'description_width': '120px'}
        )
        
        self.agent1_orientation = widgets.Dropdown(
            options=['liberal', 'moderate', 'conservative', 'neutral'],
            value='moderate',
            description='Orientation:',
            style={'description_width': '120px'}
        )
        
        # Agent 2 Configuration
        self.agent2_name = widgets.Text(
            value='Advisor',
            description='Agent 2 Name:',
            style={'description_width': '120px'}
        )
        
        self.agent2_persona = widgets.Textarea(
            value='A knowledgeable advisor who provides balanced perspectives.',
            description='Persona:',
            layout=widgets.Layout(width='100%', height='100px'),
            style={'description_width': '120px'}
        )
        
        self.agent2_orientation = widgets.Dropdown(
            options=['liberal', 'moderate', 'conservative', 'neutral'],
            value='neutral',
            description='Orientation:',
            style={'description_width': '120px'}
        )
        
        # Options
        self.save_results = widgets.Checkbox(
            value=True,
            description='Save Results',
            style={'description_width': '120px'}
        )
        
        self.output_dir = widgets.Text(
            value='./results',
            description='Output Dir:',
            style={'description_width': '120px'}
        )
        
        # Buttons
        self.run_button = widgets.Button(
            description='Run Experiment',
            button_style='primary',
            icon='play'
        )
        self.run_button.on_click(self._on_run_click)
        
        self.export_button = widgets.Button(
            description='Export Config',
            button_style='info',
            icon='download'
        )
        self.export_button.on_click(self._on_export_click)
        
        self.clear_button = widgets.Button(
            description='Clear Output',
            button_style='warning',
            icon='trash'
        )
        self.clear_button.on_click(self._on_clear_click)
        
        # Progress
        self.progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Progress:',
            style={'description_width': '120px'}
        )
        
        self.status_label = widgets.Label(value='Ready')
    
    def display(self):
        """Display the configurator widget."""
        # Experiment section
        experiment_section = widgets.VBox([
            widgets.HTML('<h3>📋 Experiment Settings</h3>'),
            self.name_input,
            self.max_turns,
            self.topic_input
        ])
        
        # Provider section
        provider_section = widgets.VBox([
            widgets.HTML('<h3>🤖 LLM Provider</h3>'),
            self.provider_dropdown,
            self.model_input
        ])
        
        # Agent 1 section
        agent1_section = widgets.VBox([
            widgets.HTML('<h3>👤 Agent 1</h3>'),
            self.agent1_name,
            self.agent1_persona,
            self.agent1_orientation
        ])
        
        # Agent 2 section
        agent2_section = widgets.VBox([
            widgets.HTML('<h3>👥 Agent 2</h3>'),
            self.agent2_name,
            self.agent2_persona,
            self.agent2_orientation
        ])
        
        # Options section
        options_section = widgets.VBox([
            widgets.HTML('<h3>⚙️ Options</h3>'),
            self.save_results,
            self.output_dir
        ])
        
        # Buttons
        button_row = widgets.HBox([
            self.run_button,
            self.export_button,
            self.clear_button
        ])
        
        # Progress section
        progress_section = widgets.VBox([
            widgets.HTML('<h3>📊 Status</h3>'),
            self.progress,
            self.status_label
        ])
        
        # Main layout
        left_col = widgets.VBox([experiment_section, provider_section])
        right_col = widgets.VBox([agent1_section, agent2_section])
        
        main_grid = widgets.HBox([left_col, right_col])
        
        full_layout = widgets.VBox([
            widgets.HTML('<h2>🔬 Synthetic Experiment Configurator</h2>'),
            main_grid,
            options_section,
            button_row,
            progress_section,
            widgets.HTML('<h3>📝 Output</h3>'),
            self._output
        ])
        
        display(full_layout)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            'name': self.name_input.value,
            'max_turns': self.max_turns.value,
            'initial_topic': self.topic_input.value,
            'provider': self.provider_dropdown.value,
            'model': self.model_input.value,
            'agents': [
                {
                    'name': self.agent1_name.value,
                    'persona': self.agent1_persona.value,
                    'orientation': self.agent1_orientation.value,
                    'role': 'user'
                },
                {
                    'name': self.agent2_name.value,
                    'persona': self.agent2_persona.value,
                    'orientation': self.agent2_orientation.value,
                    'role': 'assistant'
                }
            ],
            'save_results': self.save_results.value,
            'output_dir': self.output_dir.value
        }
    
    def create_experiment(self):
        """
        Create experiment from current configuration.
        
        Returns:
            Experiment instance
        """
        from synthetic_experiments import Experiment
        from synthetic_experiments.agents import ConversationAgent, Persona
        from synthetic_experiments.providers import OllamaProvider, ClaudeProvider, OpenAIProvider
        
        config = self.get_config()
        
        # Create provider
        provider_map = {
            'ollama': OllamaProvider,
            'openai': OpenAIProvider,
            'claude': ClaudeProvider
        }
        ProviderClass = provider_map.get(config['provider'], OllamaProvider)
        
        # Create agents
        agents = []
        for agent_config in config['agents']:
            provider = ProviderClass(model_name=config['model'])
            persona = Persona(
                name=agent_config['name'],
                background=agent_config['persona'],
                political_orientation=agent_config['orientation']
            )
            agent = ConversationAgent(
                provider=provider,
                persona=persona,
                role=agent_config['role']
            )
            agents.append(agent)
        
        # Create experiment
        experiment = Experiment(
            name=config['name'],
            agents=agents
        )
        experiment.config.max_turns = config['max_turns']
        experiment.config.initial_topic = config['initial_topic']
        
        return experiment
    
    def _on_run_click(self, button):
        """Handle run button click."""
        with self._output:
            clear_output(wait=True)
            print("🚀 Starting experiment...")
            
            self.status_label.value = 'Running...'
            self.progress.value = 0
            
            try:
                experiment = self.create_experiment()
                config = self.get_config()
                
                # Run with progress updates
                result = experiment.run(
                    max_turns=config['max_turns'],
                    initial_topic=config['initial_topic']
                )
                
                self.progress.value = 100
                self.status_label.value = 'Complete!'
                
                print(f"✅ Experiment complete!")
                print(f"   Turns: {len(result.turns)}")
                print(f"   Duration: {result.metadata.get('duration', 'N/A')}")
                
                if config['save_results']:
                    # Save results
                    import os
                    os.makedirs(config['output_dir'], exist_ok=True)
                    output_path = os.path.join(
                        config['output_dir'],
                        f"{config['name']}_results.json"
                    )
                    result.save(output_path)
                    print(f"   Saved to: {output_path}")
                
                self._config['last_result'] = result
                
            except Exception as e:
                self.status_label.value = f'Error: {str(e)}'
                print(f"❌ Error: {e}")
    
    def _on_export_click(self, button):
        """Handle export button click."""
        with self._output:
            clear_output(wait=True)
            config = self.get_config()
            
            print("📄 Configuration:")
            print(json.dumps(config, indent=2))
    
    def _on_clear_click(self, button):
        """Handle clear button click."""
        with self._output:
            clear_output()


@require_widgets
class ConversationViewer:
    """
    Interactive widget for viewing conversation results.
    
    Example:
        >>> viewer = ConversationViewer(conversation_logger)
        >>> viewer.display()
    """
    
    def __init__(self, logger=None):
        """
        Initialize viewer.
        
        Args:
            logger: ConversationLogger instance
        """
        self.logger = logger
        self._output = widgets.Output()
    
    def display(self):
        """Display conversation viewer."""
        if not self.logger:
            display(widgets.HTML('<p>No conversation data loaded.</p>'))
            return
        
        # Turn selector
        turn_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=len(self.logger.turns),
            description='Turn:'
        )
        
        # Message display
        message_html = widgets.HTML()
        
        def update_message(change):
            turn_idx = change['new'] - 1
            if 0 <= turn_idx < len(self.logger.turns):
                turn = self.logger.turns[turn_idx]
                message_html.value = f'''
                <div style="padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                    <strong>{turn.get('agent', 'Unknown')}:</strong>
                    <p>{turn.get('content', '')}</p>
                    <small>Turn {turn_idx + 1}</small>
                </div>
                '''
        
        turn_slider.observe(update_message, names='value')
        update_message({'new': 1})  # Initialize
        
        # Stats
        stats_html = widgets.HTML(value=f'''
        <div style="padding: 10px; background: #f5f5f5; border-radius: 5px;">
            <h4>📊 Conversation Stats</h4>
            <ul>
                <li>Total turns: {len(self.logger.turns)}</li>
                <li>Experiment: {self.logger.experiment_name}</li>
            </ul>
        </div>
        ''')
        
        # Layout
        display(widgets.VBox([
            widgets.HTML('<h2>💬 Conversation Viewer</h2>'),
            stats_html,
            turn_slider,
            message_html
        ]))


@require_widgets
class ResultsExplorer:
    """
    Interactive widget for exploring experiment results.
    
    Example:
        >>> explorer = ResultsExplorer()
        >>> explorer.load_results('./results')
        >>> explorer.display()
    """
    
    def __init__(self):
        self._results = []
        self._output = widgets.Output()
    
    def load_results(self, directory: str):
        """Load results from directory."""
        import os
        import glob
        
        pattern = os.path.join(directory, '*.json')
        files = glob.glob(pattern)
        
        self._results = []
        for f in files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    data['_file'] = f
                    self._results.append(data)
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")
    
    def display(self):
        """Display results explorer."""
        if not self._results:
            display(widgets.HTML('<p>No results loaded. Call load_results() first.</p>'))
            return
        
        # File selector
        file_options = [(r.get('experiment_name', r['_file']), i) for i, r in enumerate(self._results)]
        file_selector = widgets.Dropdown(
            options=file_options,
            description='Experiment:'
        )
        
        # Details output
        details_html = widgets.HTML()
        
        def update_details(change):
            idx = change['new']
            result = self._results[idx]
            
            turns = result.get('turns', [])
            details_html.value = f'''
            <div style="padding: 10px;">
                <h4>📄 {result.get('experiment_name', 'Unknown')}</h4>
                <p><strong>Turns:</strong> {len(turns)}</p>
                <p><strong>File:</strong> {result['_file']}</p>
                <hr>
                <h5>Preview (first 3 turns):</h5>
            </div>
            '''
            
            for turn in turns[:3]:
                details_html.value += f'''
                <div style="margin: 5px 0; padding: 5px; background: #f9f9f9; border-radius: 3px;">
                    <strong>{turn.get('agent', 'Unknown')}:</strong>
                    {turn.get('content', '')[:200]}...
                </div>
                '''
        
        file_selector.observe(update_details, names='value')
        if self._results:
            update_details({'new': 0})
        
        display(widgets.VBox([
            widgets.HTML('<h2>🔍 Results Explorer</h2>'),
            file_selector,
            details_html
        ]))


# Convenience function
@require_widgets
def interactive_experiment():
    """
    Launch an interactive experiment configurator.
    
    Returns:
        ExperimentConfigurator instance
    """
    config = ExperimentConfigurator()
    config.display()
    return config


@require_widgets
def view_conversation(logger):
    """
    Launch conversation viewer.
    
    Args:
        logger: ConversationLogger instance
    """
    viewer = ConversationViewer(logger)
    viewer.display()
    return viewer
