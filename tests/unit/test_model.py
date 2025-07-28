import pytest
from src.model import Model
from src.primitives import Interaction
from src.agent_registry import DEFAULT_AGENT

@pytest.fixture
def model():
    """Provides a fresh Model instance for each test."""
    return Model(default_agent_name=DEFAULT_AGENT)

@pytest.mark.asyncio
async def test_model_initialization(model: Model):
    """Test that the model initializes correctly without a session."""
    assert not model.sessions
    assert model.active_session_id is None

@pytest.mark.asyncio
async def test_create_session(model: Model):
    """Test session creation and activation."""
    assert len(model.sessions) == 0
    
    session = await model.create_session()
    
    assert len(model.sessions) == 1
    assert model.active_session_id == session.id
    assert model.get_active_session() is session
    assert session.agent_name == DEFAULT_AGENT

@pytest.mark.asyncio
async def test_add_interaction(model: Model):
    """Test adding an interaction to the active session."""
    session = await model.create_session()
    
    interaction = Interaction("Test content")
    await model.add_interaction_to_active_session(interaction)
    
    active_session = model.get_active_session()
    assert active_session
    assert len(active_session.interactions) == 1
    assert active_session.interactions[0] == interaction

@pytest.mark.asyncio
async def test_display_history_filtering(model: Model):
    """Test that display_history only returns user-facing interactions."""
    await model.create_session()
    
    await model.add_interaction_to_active_session(Interaction("User message", metadata={"user-facing": True}))
    await model.add_interaction_to_active_session(Interaction("System log", metadata={"user-facing": False}))
    await model.add_interaction_to_active_session(Interaction("Agent response", metadata={"user-facing": True}))
    
    display_history = model.display_history
    assert len(display_history) == 2
    assert display_history[0].contents == "User message"
    assert display_history[1].contents == "Agent response"