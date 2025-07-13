# model.py
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

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
        self.interactions: list[Interaction] = []  # Renamed from conversation_log
        self.conversation_history: list[PromptMessageMultipart] = []  # Agent conversation history
        self.is_thinking: bool = False
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        
        # Initialize user preferences
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"

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