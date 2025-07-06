import pytest
from unittest.mock import AsyncMock, MagicMock
from model import Model, AppState
from controller import Controller
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_prompt_handling_integration():
    """Test the full integration between Model and Controller for prompt handling."""
    model = Model()
    
    # Mock the agent_app and the agent's generate method
    mock_agent = AsyncMock()
    mock_response = MagicMock()
    mock_response.role = 'assistant'
    mock_response.content = [{'type': 'text', 'text': 'Mocked response'}]
    mock_agent.generate.return_value = mock_response
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    assert len(model.conversation_history) == 2
    assert model.conversation_history[0].role == 'user'
    assert model.conversation_history[1].role == 'assistant'
    assert model.conversation_history[1].last_text() == 'Mocked response'


@pytest.mark.asyncio
async def test_command_integration():
    """Test the integration of command handling with the Model."""
    model = Model()
    mock_agent_app = MagicMock()
    controller = Controller(model, mock_agent_app)

    # Test save command integration
    await controller.process_user_input("/save test_integration.json")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message == "History saved successfully."

    # Test clear command integration
    await controller.process_user_input("/clear")
    assert len(model.conversation_history) == 0
    assert model.last_success_message == "Conversation history cleared."


@pytest.mark.asyncio
async def test_error_handling_integration():
    """Test error handling integration between Model and Controller."""
    model = Model()
    
    # Mock agent that always fails
    mock_agent = AsyncMock()
    mock_agent.generate.side_effect = Exception("Test error")
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    # Add a message first to test rollback
    await model.add_message(Prompt.user("Previous message"))
    initial_history_length = len(model.conversation_history)

    await controller.process_user_input("This will fail")

    # Should have rolled back the user message
    assert len(model.conversation_history) == initial_history_length
    assert model.application_state == AppState.ERROR
    assert "Test error" in model.last_error_message


@pytest.mark.asyncio
async def test_state_management_integration():
    """Test that state management works correctly across the integration."""
    model = Model()
    mock_agent_app = MagicMock()
    controller = Controller(model, mock_agent_app)

    # Test that state changes are properly managed
    assert model.application_state == AppState.IDLE
    
    # Simulate a command that changes state
    await controller.process_user_input("/clear")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message is not None


@pytest.mark.asyncio
async def test_conversation_flow_integration():
    """Test a complete conversation flow with multiple turns."""
    model = Model()
    
    # Mock agent that returns different responses
    mock_agent = AsyncMock()
    responses = [
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'First response'}]),
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Second response'}]),
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Third response'}])
    ]
    mock_agent.generate.side_effect = responses
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    # Simulate a conversation
    await controller.process_user_input("First message")
    await controller.process_user_input("Second message")
    await controller.process_user_input("Third message")

    assert len(model.conversation_history) == 6  # 3 user + 3 assistant messages
    assert model.conversation_history[0].role == 'user'
    assert model.conversation_history[1].role == 'assistant'
    assert model.conversation_history[2].role == 'user'
    assert model.conversation_history[3].role == 'assistant'
    assert model.conversation_history[4].role == 'user'
    assert model.conversation_history[5].role == 'assistant' 