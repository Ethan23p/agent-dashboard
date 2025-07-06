# model.py
import asyncio
import json
import os
from datetime import datetime
from enum import Enum, auto
from typing import Callable, List, Optional

# Core types from the fast-agent framework.
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

class AppState(Enum):
    """Defines the possible states of the client application."""
    IDLE = auto()
    AGENT_IS_THINKING = auto()
    WAITING_FOR_USER_INPUT = auto()
    ERROR = auto()

class Model:
    """
    The Model represents the single source of truth for the application's state.
    It holds all data and notifies listeners when its state changes. It contains
    no business logic and is entirely passive.
    """
    def __init__(self):
        # --- State Data ---
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history: List[PromptMessageMultipart] = []
        self.application_state: AppState = AppState.IDLE
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

    async def add_message(self, message: PromptMessageMultipart):
        """Appends a new message to the conversation history."""
        self.conversation_history.append(message)
        await self._notify_listeners()

    async def pop_last_message(self) -> Optional[PromptMessageMultipart]:
        """
        Removes and returns the last message from the history.
        Crucial for rolling back state on agent failure.
        """
        if not self.conversation_history:
            return None
        last_message = self.conversation_history.pop()
        await self._notify_listeners()
        return last_message

    async def clear_history(self):
        """Clears the entire conversation history."""
        self.conversation_history = []
        await self._notify_listeners()

    async def set_state(self, new_state: AppState, error_message: Optional[str] = None, success_message: Optional[str] = None):
        """Updates the application's current state and notifies listeners."""
        self.application_state = new_state
        if new_state == AppState.ERROR:
            self.last_error_message = error_message
            self.last_success_message = None
        else:
            self.last_error_message = None # Clear error on non-error states.
            self.last_success_message = success_message
        await self._notify_listeners()

    async def load_history_from_file(self, filepath: str) -> bool:
        """
        Loads conversation history from a JSON file, replacing the current history.
        Returns True on success, False on failure.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)
            # Re-create the rich PromptMessageMultipart objects from the raw dicts.
            self.conversation_history = [
                PromptMessageMultipart(**data) for data in raw_history
            ]
            await self._notify_listeners()
            return True
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            # We don't change state on failure, just report it.
            await self.set_state(AppState.ERROR, f"Failed to load history: {e}")
            return False

    # --- Methods for Actions (Instructed by the Controller) ---

    async def save_history_to_file(self, filepath: Optional[str] = None) -> bool:
        """
        Saves the current conversation history to a specified JSON file.
        This method does not mutate the model's state.
        Returns True on success, False on failure.
        """
        target_filepath = filepath or self.user_preferences["auto_save_filename"]
        context_dir = self._get_context_dir()
        os.makedirs(context_dir, exist_ok=True)

        try:
            serializable_history = [
                message.model_dump(mode='json') for message in self.conversation_history
            ]
            with open(target_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            # If saving fails, we set an error state to inform the user.
            await self.set_state(AppState.ERROR, f"Could not write to file {target_filepath}")
            return False