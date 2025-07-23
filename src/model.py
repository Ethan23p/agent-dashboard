import asyncio
import json
import logging
import os
from typing import Callable, List, Optional

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from primitives import Interaction, Session

logger = logging.getLogger(__name__)

async def save_session(session: Session, filepath: str) -> bool:
    """Saves a session to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Session record saved successfully to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session record to {filepath}: {e}", exc_info=True)
        return False

async def load_session(filepath: str) -> Optional[Session]:
    """Loads a session from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        session = Session.from_dict(session_data)
        logger.info(f"Session record loaded successfully from {filepath}")
        return session
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to load session record from {filepath}: {e}", exc_info=True)
        return None

class Model:
    """
    Manages the application's state, including all sessions, and notifies
    listeners of changes.
    """
    def __init__(self, default_agent_name: str):
        self.sessions: List[Session] = []
        self.active_session_id: Optional[str] = None
        self.is_thinking: bool = False
        self.default_agent_name: str = default_agent_name
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        self._listeners: List[Callable] = []

    async def _notify_listeners(self):
        """Notifies listeners of a state change."""
        # Create tasks to avoid blocking the model while listeners run.
        for listener in self._listeners:
            asyncio.create_task(listener())

    def register_listener(self, listener: Callable):
        """Register a callback for state changes."""
        self._listeners.append(listener)

    async def create_session(self, agent_name: Optional[str] = None) -> Session:
        """Creates a new session and sets it as active."""
        agent = agent_name or self.default_agent_name
        new_session = Session(agent_name=agent)
        self.sessions.append(new_session)
        self.active_session_id = new_session.id
        logger.info(f"Created and activated new session {new_session.id} for agent '{agent}'.")
        await self._notify_listeners()
        return new_session

    def get_session(self, session_id: str) -> Optional[Session]:
        return next((s for s in self.sessions if s.id == session_id), None)

    def get_active_session(self) -> Optional[Session]:
        if self.active_session_id:
            return self.get_session(self.active_session_id)
        return None

    async def set_active_session(self, session_id: str):
        if self.get_session(session_id):
            self.active_session_id = session_id
            logger.info(f"Switched active session to {session_id}.")
            await self._notify_listeners()
        else:
            logger.warning(f"Attempted to switch to non-existent session {session_id}.")

    async def clear_sessions(self):
        """Clears all sessions and creates a new default one."""
        self.sessions = []
        self.active_session_id = None
        await self.create_session(self.default_agent_name)
        logger.info("All sessions cleared; new default session created.")


    async def add_interaction_to_active_session(self, interaction: Interaction):
        """Adds an interaction to the active session."""
        active_session = self.get_active_session()
        if active_session:
            active_session.interactions.append(interaction)
            await self._notify_listeners()
        else:
            logger.warning("Attempted to add interaction, but no active session.")

    def get_agent_history_for_active_session(self) -> List[PromptMessageMultipart]:
        """Constructs the conversation history for the agent from the active session."""
        active_session = self.get_active_session()
        if not active_session:
            return []
        
        history: List[PromptMessageMultipart] = []
        for interaction in active_session.interactions:
            if isinstance(interaction.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in interaction.contents):
                history.extend(interaction.contents)
        return history

    @property
    def display_history(self) -> List[Interaction]:
        """
        Returns a filtered list of interactions suitable for UI display.
        This simplifies the view's rendering logic.
        """
        active_session = self.get_active_session()
        if not active_session:
            return []
        
        return [
            interaction for interaction in active_session.interactions
            if interaction.metadata.get("user-facing", False)
        ]


    async def set_thinking_status(self, is_thinking: bool):
        """Sets the agent's thinking status and notifies listeners."""
        if self.is_thinking != is_thinking:
            self.is_thinking = is_thinking
            await self._notify_listeners()


    async def save_active_session(self):
        """Saves the active session to a file."""
        active_session = self.get_active_session()
        if not active_session:
            logger.warning("Attempted to save, but no active session.")
            return

        context_dir = self.user_preferences.get("context_dir", "_context")
        filename = f"{context_dir}/{active_session.id}.json"
        await save_session(active_session, filename)