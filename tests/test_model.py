# tests/test_model.py
import pytest
import os
import tempfile
from model import Model, load_session, save_session
from primitives import Interaction, Session
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text

@pytest.fixture
def model() -> Model:
    """Fixture to provide a clean Model instance for each test."""
    return Model(default_agent_name="minimal")

@pytest.mark.asyncio
async def test_model_initial_state(model: Model):
    """Test the initial state of the Model."""
    assert model.sessions == []
    assert model.active_session_id is None
    assert model.is_thinking is False

@pytest.mark.asyncio
async def test_session_creation_and_management(model: Model):
    """Test creating, getting, and switching sessions."""
    session1 = await model.create_session(agent_name="coding")
    assert len(model.sessions) == 1
    assert model.active_session_id == session1.id
    assert model.get_active_session() is session1
    assert session1.agent_name == "coding"

    session2 = await model.create_session(agent_name="interpreter")
    assert len(model.sessions) == 2
    assert model.active_session_id == session2.id

    await model.set_active_session(session1.id)
    assert model.active_session_id == session1.id

@pytest.mark.asyncio
async def test_interaction_and_history(model: Model):
    """Test adding interactions and the different history views."""
    await model.create_session()
    
    user_prompt = Interaction([Prompt.user("Hello")], metadata={"user-facing": True})
    agent_response = Interaction([Prompt.assistant("Hi")], metadata={"user-facing": True})
    internal_thought = Interaction(Text("Thinking..."), metadata={"user-facing": False})

    await model.add_interaction_to_active_session(user_prompt)
    await model.add_interaction_to_active_session(internal_thought)
    await model.add_interaction_to_active_session(agent_response)

    active_session = model.get_active_session()
    assert active_session is not None
    assert len(active_session.interactions) == 3

    display_history = model.display_history
    assert len(display_history) == 2
    assert display_history[0] is user_prompt
    assert display_history[1] is agent_response

    agent_history = model.get_agent_history_for_active_session()
    assert len(agent_history) == 2
    assert agent_history[0].role == "user"
    assert agent_history[1].role == "assistant"

@pytest.mark.asyncio
async def test_save_and_load_session(model: Model):
    """Test saving a session to a file and loading it back."""
    session = await model.create_session()
    await model.add_interaction_to_active_session(
        Interaction([Prompt.user("Test message")])
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
    
    try:
        success = await save_session(session, temp_filename)
        assert success is True
        
        loaded_session = await load_session(temp_filename)
        assert loaded_session is not None
        assert loaded_session.id == session.id
        assert len(loaded_session.interactions) == 1
        
        interaction_contents = loaded_session.interactions[0].contents
        assert isinstance(interaction_contents, list)
        assert isinstance(interaction_contents[0], PromptMessageMultipart)
        assert interaction_contents[0].last_text() == "Test message"
        
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)