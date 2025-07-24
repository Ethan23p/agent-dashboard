# tests/test_primitives.py
import pytest
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text
from primitives import Interaction, Session

@pytest.fixture
def sample_pmp_list() -> list[PromptMessageMultipart]:
    """Fixture for a sample list of PromptMessageMultipart objects."""
    # LINTER FIX: Used Prompt.user() and Prompt.assistant() which are the correct
    # factory methods, instead of the non-existent from_text().
    return [
        Prompt.user("Hello"),
        Prompt.assistant("Hi there!")
    ]

@pytest.fixture
def sample_rich_text() -> Text:
    """Fixture for a sample Rich Text object."""
    return Text.from_markup("[bold red]System Alert![/]")

def test_interaction_serialization_pmp(sample_pmp_list):
    """Test round-trip serialization for an Interaction with PromptMessageMultipart list."""
    # An interaction's content can be a list of messages representing a full turn.
    interaction = Interaction(contents=sample_pmp_list, metadata={"source": "agent"})
    
    interaction_dict = interaction.to_dict()
    assert interaction_dict["metadata"]["_content_type"] == "prompt_message_multipart_list"
    
    reconstructed_interaction = Interaction.from_dict(interaction_dict)
    
    assert isinstance(reconstructed_interaction.contents, list)
    assert len(reconstructed_interaction.contents) == 2
    assert all(isinstance(item, PromptMessageMultipart) for item in reconstructed_interaction.contents)
    assert reconstructed_interaction.contents[0].last_text() == "Hello"
    assert reconstructed_interaction.metadata["source"] == "agent"

def test_interaction_serialization_rich_text(sample_rich_text):
    """Test round-trip serialization for an Interaction with Rich Text."""
    interaction = Interaction(contents=sample_rich_text, metadata={"type": "system"})
    
    interaction_dict = interaction.to_dict()
    assert interaction_dict["metadata"]["_content_type"] == "rich_text"
    assert interaction_dict["contents"] == "[bold red]System Alert![/bold red]"
    
    reconstructed_interaction = Interaction.from_dict(interaction_dict)
    
    assert isinstance(reconstructed_interaction.contents, Text)
    assert reconstructed_interaction.contents.markup == "[bold red]System Alert![/bold red]"
    assert reconstructed_interaction.metadata["type"] == "system"

def test_session_serialization(sample_pmp_list, sample_rich_text):
    """Test round-trip serialization for a full Session object."""
    session = Session(agent_name="coding", status="completed")
    # A single interaction can contain a multi-message turn
    session.interactions.append(Interaction(contents=sample_pmp_list))
    session.interactions.append(Interaction(contents=sample_rich_text))
    
    session_dict = session.to_dict()
    reconstructed_session = Session.from_dict(session_dict)
    
    assert reconstructed_session.id == session.id
    assert reconstructed_session.agent_name == "coding"
    assert reconstructed_session.status == "completed"
    assert len(reconstructed_session.interactions) == 2
    
    assert isinstance(reconstructed_session.interactions[0].contents, list)
    assert isinstance(reconstructed_session.interactions[1].contents, Text)