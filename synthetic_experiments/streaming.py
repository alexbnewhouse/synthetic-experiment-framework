"""
Streaming support for real-time conversation observation.

This module provides utilities for watching conversations as they happen,
with support for callbacks and real-time display.

Example:
    >>> from synthetic_experiments.streaming import StreamingExperiment
    >>> 
    >>> def on_message(agent_name, message):
    ...     print(f"{agent_name}: {message[:100]}...")
    >>> 
    >>> experiment = StreamingExperiment(
    ...     name="streaming_demo",
    ...     agents=[user, advisor],
    ...     on_message=on_message
    ... )
    >>> results = experiment.run(max_turns=10)
"""

from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass
import sys
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """
    Event emitted during streaming conversation.
    
    Attributes:
        event_type: Type of event (message, turn_start, turn_end, conversation_start, etc.)
        agent_name: Name of agent (for message events)
        content: Content of the message
        turn_number: Current turn number
        metadata: Additional event metadata
        timestamp: Event timestamp
    """
    event_type: str
    agent_name: Optional[str] = None
    content: Optional[str] = None
    turn_number: int = 0
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class StreamingMixin:
    """
    Mixin class that adds streaming capabilities to experiments.
    
    Add this to your Experiment class to enable real-time callbacks
    during conversation execution.
    """
    
    def __init__(self):
        self._stream_callbacks: List[Callable[[StreamEvent], None]] = []
        self._streaming_enabled = False
    
    def add_stream_callback(self, callback: Callable[[StreamEvent], None]):
        """Add a callback to receive stream events."""
        self._stream_callbacks.append(callback)
        self._streaming_enabled = True
    
    def remove_stream_callback(self, callback: Callable[[StreamEvent], None]):
        """Remove a stream callback."""
        if callback in self._stream_callbacks:
            self._stream_callbacks.remove(callback)
        self._streaming_enabled = len(self._stream_callbacks) > 0
    
    def _emit_event(self, event: StreamEvent):
        """Emit event to all callbacks."""
        for callback in self._stream_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Stream callback error: {e}")


class StreamingExperiment:
    """
    Experiment wrapper that enables real-time streaming of conversations.
    
    This wraps a standard Experiment and adds streaming callbacks for
    real-time observation of conversation progress.
    
    Example:
        >>> # Simple print callback
        >>> experiment = StreamingExperiment(
        ...     base_experiment=my_experiment,
        ...     on_message=lambda name, msg: print(f"{name}: {msg}")
        ... )
        >>> 
        >>> # Full event callback
        >>> def handle_event(event):
        ...     if event.event_type == "message":
        ...         print(f"[Turn {event.turn_number}] {event.agent_name}: {event.content}")
        ...     elif event.event_type == "turn_end":
        ...         print(f"--- Turn {event.turn_number} complete ---")
        >>> 
        >>> experiment = StreamingExperiment(
        ...     base_experiment=my_experiment,
        ...     on_event=handle_event
        ... )
    """
    
    def __init__(
        self,
        base_experiment=None,
        name: str = None,
        agents: List = None,
        on_message: Optional[Callable[[str, str], None]] = None,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
        delay_between_messages: float = 0.0,
        **experiment_kwargs
    ):
        """
        Initialize streaming experiment.
        
        Args:
            base_experiment: Existing Experiment to wrap (optional)
            name: Experiment name (if not using base_experiment)
            agents: Agents list (if not using base_experiment)
            on_message: Simple callback(agent_name, message_content)
            on_event: Full event callback(StreamEvent)
            delay_between_messages: Artificial delay between messages for readability
            **experiment_kwargs: Additional args passed to Experiment
        """
        if base_experiment:
            self._experiment = base_experiment
        else:
            from synthetic_experiments import Experiment
            self._experiment = Experiment(name=name, agents=agents, **experiment_kwargs)
        
        self._callbacks: List[Callable[[StreamEvent], None]] = []
        self._simple_callback = on_message
        self._delay = delay_between_messages
        
        if on_event:
            self._callbacks.append(on_event)
    
    @property
    def name(self):
        return self._experiment.name
    
    @property
    def agents(self):
        return self._experiment.agents
    
    @property
    def config(self):
        return self._experiment.config
    
    def add_callback(self, callback: Callable[[StreamEvent], None]):
        """Add an event callback."""
        self._callbacks.append(callback)
    
    def _emit(self, event: StreamEvent):
        """Emit event to callbacks."""
        # Simple message callback
        if self._simple_callback and event.event_type == "message":
            self._simple_callback(event.agent_name, event.content)
        
        # Full event callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def run(
        self,
        max_turns: Optional[int] = None,
        initial_topic: Optional[str] = None,
        **kwargs
    ):
        """
        Run conversation with streaming callbacks.
        
        Args:
            max_turns: Maximum turns
            initial_topic: Starting topic
            **kwargs: Additional run arguments
            
        Returns:
            ConversationLogger with results
        """
        from synthetic_experiments.data.logger import ConversationLogger
        
        # Emit conversation start
        self._emit(StreamEvent(
            event_type="conversation_start",
            metadata={"max_turns": max_turns, "topic": initial_topic}
        ))
        
        max_turns = max_turns or self._experiment.config.max_turns
        initial_topic = initial_topic or self._experiment.config.initial_topic
        
        # Reset agents
        for agent in self._experiment.agents:
            agent.reset_conversation()
        
        # Create logger
        self._experiment.current_logger = ConversationLogger(
            experiment_name=self._experiment.name,
            metadata=kwargs.get('metadata', {})
        )
        
        # Run conversation with streaming
        self._run_streaming_conversation(max_turns, initial_topic)
        
        # Finalize
        self._experiment.current_logger.finalize()
        
        if self._experiment.storage:
            self._experiment.storage.save_conversation(self._experiment.current_logger)
        
        # Emit conversation end
        self._emit(StreamEvent(
            event_type="conversation_end",
            metadata={"total_turns": len(self._experiment.current_logger.turns)}
        ))
        
        return self._experiment.current_logger
    
    def _run_streaming_conversation(self, max_turns: int, initial_topic: str):
        """Run conversation with streaming events."""
        # Get agent order
        agent_order = self._experiment._get_agent_order()
        
        # Initial prompt
        current_prompt = self._experiment._create_initial_prompt(initial_topic)
        last_speaker = None
        
        for turn in range(max_turns):
            # Emit turn start
            self._emit(StreamEvent(
                event_type="turn_start",
                turn_number=turn + 1
            ))
            
            # Get next agent
            if self._experiment.config.turn_order == "random":
                import random
                available = [a for a in self._experiment.agents if a != last_speaker]
                current_agent = random.choice(available) if available else self._experiment.agents[0]
            else:
                current_agent = agent_order[turn % len(agent_order)]
            
            try:
                # Generate response
                message = current_agent.respond(current_prompt)
                
                # Log message
                self._experiment.current_logger.log_message(current_agent.name, message)
                
                # Emit message event
                self._emit(StreamEvent(
                    event_type="message",
                    agent_name=current_agent.name,
                    content=message.content,
                    turn_number=turn + 1,
                    metadata={
                        "tokens": message.tokens_used if hasattr(message, 'tokens_used') else None,
                        "role": current_agent.role
                    }
                ))
                
                # Delay if configured
                if self._delay > 0:
                    time.sleep(self._delay)
                
                # Check stopping condition
                if self._experiment._should_stop(message.content):
                    self._emit(StreamEvent(
                        event_type="stop_condition",
                        turn_number=turn + 1,
                        metadata={"reason": "stopping_condition_met"}
                    ))
                    break
                
                # Prepare next prompt
                if len(self._experiment.agents) > 2:
                    current_prompt = f"[{current_agent.name}]: {message.content}"
                else:
                    current_prompt = message.content
                
                last_speaker = current_agent
                
            except Exception as e:
                self._emit(StreamEvent(
                    event_type="error",
                    turn_number=turn + 1,
                    metadata={"error": str(e)}
                ))
                raise
            
            # Emit turn end
            self._emit(StreamEvent(
                event_type="turn_end",
                turn_number=turn + 1
            ))


class ConsolePrinter:
    """
    Callback class that prints conversations to console in real-time.
    
    Example:
        >>> printer = ConsolePrinter(colors=True)
        >>> experiment = StreamingExperiment(
        ...     base_experiment=my_experiment,
        ...     on_event=printer
        ... )
    """
    
    # ANSI color codes
    COLORS = {
        'user': '\033[94m',      # Blue
        'assistant': '\033[92m', # Green
        'system': '\033[93m',    # Yellow
        'error': '\033[91m',     # Red
        'reset': '\033[0m',      # Reset
        'bold': '\033[1m',
        'dim': '\033[2m',
    }
    
    def __init__(
        self,
        colors: bool = True,
        show_turn_markers: bool = True,
        max_message_preview: int = 0,
        stream: Any = None
    ):
        """
        Initialize console printer.
        
        Args:
            colors: Use ANSI colors
            show_turn_markers: Show turn start/end markers
            max_message_preview: Truncate messages (0 = no truncation)
            stream: Output stream (default: sys.stdout)
        """
        self.colors = colors
        self.show_turn_markers = show_turn_markers
        self.max_preview = max_message_preview
        self.stream = stream or sys.stdout
    
    def _color(self, name: str) -> str:
        """Get color code."""
        if not self.colors:
            return ""
        return self.COLORS.get(name, "")
    
    def __call__(self, event: StreamEvent):
        """Handle stream event."""
        if event.event_type == "conversation_start":
            self._print(f"\n{self._color('bold')}═══ Conversation Started ═══{self._color('reset')}\n")
            if event.metadata.get('topic'):
                self._print(f"{self._color('dim')}Topic: {event.metadata['topic']}{self._color('reset')}\n")
        
        elif event.event_type == "turn_start":
            if self.show_turn_markers:
                self._print(f"\n{self._color('dim')}── Turn {event.turn_number} ──{self._color('reset')}\n")
        
        elif event.event_type == "message":
            role = event.metadata.get('role', 'user')
            color = self._color(role)
            
            content = event.content
            if self.max_preview > 0 and len(content) > self.max_preview:
                content = content[:self.max_preview] + "..."
            
            self._print(f"{color}{self._color('bold')}{event.agent_name}:{self._color('reset')} {content}\n")
        
        elif event.event_type == "error":
            self._print(f"{self._color('error')}ERROR: {event.metadata.get('error')}{self._color('reset')}\n")
        
        elif event.event_type == "conversation_end":
            total = event.metadata.get('total_turns', 0)
            self._print(f"\n{self._color('bold')}═══ Conversation Ended ({total} turns) ═══{self._color('reset')}\n")
    
    def _print(self, text: str):
        """Print to stream."""
        self.stream.write(text)
        self.stream.flush()


class EventLogger:
    """
    Callback class that logs events to a list for later analysis.
    
    Example:
        >>> event_log = EventLogger()
        >>> experiment = StreamingExperiment(
        ...     base_experiment=my_experiment,
        ...     on_event=event_log
        ... )
        >>> experiment.run()
        >>> print(f"Recorded {len(event_log.events)} events")
    """
    
    def __init__(self):
        self.events: List[StreamEvent] = []
    
    def __call__(self, event: StreamEvent):
        """Record event."""
        self.events.append(event)
    
    def clear(self):
        """Clear recorded events."""
        self.events = []
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get just the message events."""
        return [
            {
                'agent': e.agent_name,
                'content': e.content,
                'turn': e.turn_number,
                'timestamp': e.timestamp
            }
            for e in self.events
            if e.event_type == "message"
        ]


def stream_conversation(
    experiment,
    max_turns: int = 20,
    initial_topic: str = "",
    print_to_console: bool = True,
    **kwargs
):
    """
    Convenience function to run a streaming conversation.
    
    Args:
        experiment: Experiment instance
        max_turns: Maximum turns
        initial_topic: Starting topic
        print_to_console: Print messages to console
        **kwargs: Additional run arguments
        
    Returns:
        ConversationLogger with results
    """
    callbacks = []
    
    if print_to_console:
        printer = ConsolePrinter()
        callbacks.append(printer)
    
    streaming = StreamingExperiment(
        base_experiment=experiment,
        on_event=callbacks[0] if callbacks else None
    )
    
    return streaming.run(max_turns=max_turns, initial_topic=initial_topic, **kwargs)
