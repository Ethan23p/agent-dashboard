import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text

logger = logging.getLogger(__name__)

# Defines the flexible content for an Interaction.
ContentsType = Union[List[PromptMessageMultipart], Text, str, Dict[str, Any]]

@dataclass
class Interaction:
    """Represents a single event or exchange; the atomic unit of a session."""
    contents: ContentsType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Interaction to a dictionary."""
        serialized_contents: Any
        if isinstance(self.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in self.contents):
            serialized_contents = [msg.model_dump(mode='json') for msg in self.contents]
            # Add a type hint for robust deserialization.
            self.metadata['_content_type'] = 'prompt_message_multipart_list'
        elif isinstance(self.contents, Text):
            serialized_contents = self.contents.markup
            self.metadata['_content_type'] = 'rich_text'
        else:
            serialized_contents = self.contents
            self.metadata['_content_type'] = 'primitive'

        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "contents": serialized_contents,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Deserializes a dictionary back into an Interaction."""
        metadata = data.get("metadata", {})
        content_type = metadata.get('_content_type')
        
        deserialized_contents: ContentsType
        if content_type == 'prompt_message_multipart_list':
            deserialized_contents = [PromptMessageMultipart(**msg_data) for msg_data in data["contents"]]
        elif content_type == 'rich_text':
            deserialized_contents = Text.from_markup(data["contents"])
        else: # 'primitive' or fallback
            deserialized_contents = data["contents"]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            contents=deserialized_contents,
            metadata=metadata,
        )

@dataclass
class Session:
    """Represents a complete, continuous context of interactions."""
    id: str = field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: datetime = field(default_factory=datetime.now)
    agent_name: str = "minimal"
    status: str = "active"
    interactions: List[Interaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Session to a dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "agent_name": self.agent_name,
            "status": self.status,
            "interactions": [interaction.to_dict() for interaction in self.interactions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserializes a dictionary back into a Session."""
        return cls(
            id=data.get("id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_name=data.get("agent_name", "minimal"),
            status=data.get("status", "completed"),
            interactions=[Interaction.from_dict(interaction_data) for interaction_data in data["interactions"]],
        )


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
    """Manages the application's state and notifies listeners of changes."""
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
        # Create tasks to avoid blocking while listeners run.
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
        """Returns a filtered list of interactions for UI display."""
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
