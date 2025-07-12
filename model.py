# model.py
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

# Core types from the fast-agent framework.
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import ElicitRequestParams
from rich.text import Text

async def save_history(history: list[PromptMessageMultipart], filepath: str) -> bool:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable_history = [message.model_dump(mode='json') for message in history]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

async def load_history(filepath: str) -> list[PromptMessageMultipart] | None:
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
class ElicitationContext:
    params: ElicitRequestParams
    future: asyncio.Future

from abc import ABC
class IAppState(ABC): pass
class IdleState(IAppState): pass
class AgentIsThinkingState(IAppState): pass
class ErrorState(IAppState): pass

class Model:
    """
    The Model represents the single source of truth for the application's state.
    It holds all data and notifies listeners when its state changes. It contains
    no business logic and is entirely passive.
    """
    def __init__(self):
        # --- State Data ---
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_log: list[Interaction] = []
        self.active_elicitation_context: ElicitationContext | None = None
        self.is_thinking: bool = False
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        
        # Corrected initialization sequence:
        # 1. Initialize the dictionary with static keys first.
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        # 2. Now that self.user_preferences exists, we can safely use its
        #    values to construct and add the dynamic key.
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"

        # --- Notification System ---
        self._listeners: List[Callable] = []

    def _get_context_dir(self) -> str:
        """Helper to access the context directory from preferences."""
        return self.user_preferences.get("context_dir", "_context")

    async def _notify_listeners(self):
        """Asynchronously notify all registered listeners of a state change."""
        for listener in self._listeners:
            await listener()

    def register_listener(self, listener: Callable):
        """
        Allows other components (like the View) to register a callback
        to be notified of state changes.
        """
        self._listeners.append(listener)

    # --- Methods to Mutate State (Instructed by the Controller) ---

    async def add_interaction(self, interaction: Interaction):
        self.conversation_log.append(interaction)
        await self._notify_listeners()

    async def clear_log(self):
        self.conversation_log = []
        await self._notify_listeners()

    async def start_elicitation(self, params: ElicitRequestParams, future: asyncio.Future):
        self.active_elicitation_context = ElicitationContext(params=params, future=future)
        await self._notify_listeners()

    async def end_elicitation(self):
        self.active_elicitation_context = None
        await self._notify_listeners()

    async def set_thinking_status(self, is_thinking: bool):
        self.is_thinking = is_thinking
        await self._notify_listeners()