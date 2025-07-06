import pytest
import tempfile
import os
from model import Model, AppState
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_model_initial_state():
    """Test that the model starts in the correct initial state."""
    model = Model()
    assert model.application_state == AppState.IDLE
    assert len(model.conversation_history) == 0
    assert model.last_error_message is None
    assert model.last_success_message is None


@pytest.mark.asyncio
async def test_model_state_change():
    """Test that state changes work correctly."""
    model = Model()
    assert model.application_state == AppState.IDLE
    
    await model.set_state(AppState.ERROR, "Test Error")
    assert model.application_state == AppState.ERROR
    assert model.last_error_message == "Test Error"
    
    await model.set_state(AppState.IDLE, "Test Success")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message == "Test Success"


@pytest.mark.asyncio
async def test_add_message():
    """Test adding messages to conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    assert len(model.conversation_history) == 1
    assert model.conversation_history[0].role == 'user'
    
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    assert model.conversation_history[1].role == 'assistant'


@pytest.mark.asyncio
async def test_pop_last_message():
    """Test removing the last message from conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    
    await model.pop_last_message()
    assert len(model.conversation_history) == 1
    assert model.conversation_history[0].role == 'user'


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing the conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    
    await model.clear_history()
    assert len(model.conversation_history) == 0


@pytest.mark.asyncio
async def test_save_and_load_history():
    """Test saving and loading conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    
    # Test saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
    
    try:
        success = await model.save_history_to_file(temp_filename)
        assert success is True
        
        # Test loading
        new_model = Model()
        success = await new_model.load_history_from_file(temp_filename)
        assert success is True
        assert len(new_model.conversation_history) == 2
        assert new_model.conversation_history[0].role == 'user'
        assert new_model.conversation_history[1].role == 'assistant'
        
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@pytest.mark.asyncio
async def test_user_preferences():
    """Test user preferences functionality."""
    model = Model()
    
    # Test default preferences
    assert model.user_preferences.get("auto_save_enabled") is True
    
    # Test setting preferences
    model.user_preferences["auto_save_enabled"] = False
    assert model.user_preferences.get("auto_save_enabled") is False
    
    model.user_preferences["test_setting"] = "test_value"
    assert model.user_preferences.get("test_setting") == "test_value" 