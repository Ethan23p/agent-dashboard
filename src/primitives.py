import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Union

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text

# Defines the flexible content for an Interaction, allowing it to hold agent
# messages, system notifications, or other data structures.
ContentsType = Union[List[PromptMessageMultipart], Text, str, Dict[str, Any]]

@dataclass
class Interaction:
    """Represents a single event or exchange, the atomic unit of a session."""
    contents: ContentsType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Interaction to a dictionary."""
        serialized_contents: Any
        if isinstance(self.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in self.contents):
            serialized_contents = [msg.model_dump(mode='json') for msg in self.contents]
            # Add a type hint to the metadata for robust deserialization.
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
    """
    Represents a complete, continuous context of interactions, replacing the
    previous concepts of 'Task' and 'Conversation'.
    """
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
