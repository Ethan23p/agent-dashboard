# model.py
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Dict

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.core.prompt import Prompt
from rich.text import Text


async def save_history(history: list[PromptMessageMultipart], filepath: str) -> bool:
    """Save conversation history to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable_history = [message.model_dump(mode='json') for message in history]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


async def load_history(filepath: str) -> list[PromptMessageMultipart] | None:
    """Load conversation history from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_history = json.load(f)
        return [PromptMessageMultipart(**data) for data in raw_history]
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


@dataclass
class Interaction:
    content: Text
    tag: str = "message"

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    status: str = "pending"  # pending, running, completed, failed
    agent_name: str = "minimal"
    conversation_history: List[PromptMessageMultipart] = field(default_factory=list)
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


# State classes
from abc import ABC
class IAppState(ABC): pass
class IdleState(IAppState): pass
class AgentIsThinkingState(IAppState): pass
class ErrorState(IAppState): pass


class Model:
    """
    The Model represents the single source of truth for the application's state.
    It holds all data and notifies listeners when its state changes.
    """
    def __init__(self):
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tasks: List[Task] = []
        self.interactions: List[Interaction] = []  # For the UI log
        self.is_thinking: bool = False
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        
        # Initialize user preferences
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"
        self.default_agent_name: str = "minimal"

        self._listeners: List[Callable] = []

    def _get_context_dir(self) -> str:
        """Get the context directory from preferences."""
        return self.user_preferences.get("context_dir", "_context")

    async def _notify_listeners(self):
        """Notify all registered listeners of a state change."""
        for listener in self._listeners:
            await listener()

    def register_listener(self, listener: Callable):
        """Register a callback to be notified of state changes."""
        self._listeners.append(listener)

    async def add_interaction(self, interaction: Interaction):
        """Add an interaction to the conversation log."""
        self.interactions.append(interaction)
        await self._notify_listeners()
    
    async def create_task(self, prompt: str, agent_name: str) -> Task:
        """Creates a new task, adds it to the model, and returns it."""
        task = Task(prompt=prompt, agent_name=agent_name)
        task.conversation_history.append(Prompt.user(prompt))
        self.tasks.append(task)
        interaction = Interaction(Text.from_markup(f"[bold yellow]New Task '{task.id[:8]}':[/] {prompt}"), tag="task_created")
        await self.add_interaction(interaction)
        return task

    async def update_task(self, task_id: str, **updates):
        """Updates attributes of a specific task."""
        task = self.get_task(task_id)
        if task:
            for key, value in updates.items():
                setattr(task, key, value)
            if "status" in updates:
                interaction = Interaction(Text.from_markup(f"[dim]Task '{task.id[:8]}' status changed to {task.status}[/]"), tag="task_status")
                await self.add_interaction(interaction)
        await self._notify_listeners()

    async def add_assistant_turn_to_task(self, task_id: str, response_message: PromptMessageMultipart):
        """Adds an assistant response to a specific task's history."""
        task = self.get_task(task_id)
        if task:
            task.conversation_history.append(response_message)
            agent_interaction = Interaction(
                content=Text.from_markup(f"[bold magenta]Task '{task_id[:8]}':[/] {response_message.last_text()}"),
                tag="agent_response"
            )
            await self.add_interaction(agent_interaction)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Find a task by its ID."""
        return next((task for task in self.tasks if task.id == task_id), None)

    def get_last_task(self) -> Optional[Task]:
        """Get the most recently created task."""
        return self.tasks[-1] if self.tasks else None

    async def clear_tasks(self):
        """Clear all tasks and interactions."""
        self.tasks = []
        self.interactions = []
        await self._notify_listeners()

    async def add_user_turn(self, user_input: str):
        """Adds a user turn to both the agent history and the UI log."""
        user_message = Prompt.user(user_input)
        self.conversation_history.append(user_message)
        user_interaction = Interaction(Text.from_markup(f"[bold blue]You:[/bold blue] {user_input}"), tag="user_prompt")
        self.interactions.append(user_interaction)
        await self._notify_listeners()

    async def add_assistant_turn(self, response_message: PromptMessageMultipart):
        """Adds an assistant turn to both the agent history and the UI log."""
        self.conversation_history.append(response_message)
        agent_interaction = Interaction(
            content=Text.from_markup(f"[bold magenta]Agent:[/bold magenta] {response_message.last_text()}"),
            tag="agent_response"
        )
        self.interactions.append(agent_interaction)
        await self._notify_listeners()

    async def clear_log(self):
        """Clear the conversation log."""
        self.interactions = []
        self.conversation_history = []
        await self._notify_listeners()

    async def set_thinking_status(self, is_thinking: bool):
        """Set the agent's thinking status."""
        self.is_thinking = is_thinking
        await self._notify_listeners()