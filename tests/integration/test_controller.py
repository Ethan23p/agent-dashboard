import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from src.controller import Controller
from src.primitives import Interaction
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent

@pytest.fixture
def controller(mock_model, mock_app):
    """Provides a Controller instance with mocked dependencies."""
    return Controller(mock_model, mock_app)

@pytest.mark.asyncio
async def test_handle_prompt_and_agent_turn(controller: Controller, mocker):
    """
    Tests the main interaction loop: user prompt -> agent call -> agent response.
    This test validates the core logic without calling a real LLM.
    """
    # 1. Setup: Mock the agent's response
    mock_agent_response = PromptMessageMultipart(
        role="assistant",
        content=[TextContent(type="text", text="This is a mock agent response.")]
    )
    
    # Mock the agent instance and its generate method
    mock_agent_instance = MagicMock()
    mock_agent_instance.generate = AsyncMock(return_value=mock_agent_response)
    
    # Mock the agent application context manager
    mock_agent_app = MagicMock()
    mock_agent_app.__getitem__.return_value = mock_agent_instance
    
    mock_agent_context = AsyncMock()
    mock_agent_context.__aenter__.return_value = mock_agent_app
    
    # Mock the get_agent function to return our controlled agent
    mocker.patch(
        'src.controller.get_agent', 
        return_value=MagicMock(run=MagicMock(return_value=mock_agent_context))
    )

    # Mock the active session to control its state
    mock_session = MagicMock()
    mock_session.agent_name = "minimal"
    controller.model.get_active_session.return_value = mock_session

    # 2. Action: Process a user prompt
    user_prompt = "Hello, agent!"
    await controller.process_user_input(user_prompt)
    
    await asyncio.sleep(0.01) # Wait for the background worker task to run

    # 3. Assertions: Verify the controller's behavior
    controller.model.set_thinking_status.assert_any_call(True)
    controller.model.set_thinking_status.assert_any_call(False)

    assert controller.model.add_interaction_to_active_session.call_count == 2

    # Verify the user prompt interaction was added correctly
    user_interaction_call = controller.model.add_interaction_to_active_session.call_args_list[0]
    user_interaction_arg: Interaction = user_interaction_call.args[0]
    assert isinstance(user_interaction_arg, Interaction)
    assert user_interaction_arg.metadata["type"] == "user_prompt"
    
    assert isinstance(user_interaction_arg.contents, list)
    assert isinstance(user_interaction_arg.contents[0], PromptMessageMultipart)
    assert user_interaction_arg.contents[0].last_text() == user_prompt

    # Verify the agent response interaction was added correctly
    agent_interaction_call = controller.model.add_interaction_to_active_session.call_args_list[1]
    agent_interaction_arg: Interaction = agent_interaction_call.args[0]
    assert isinstance(agent_interaction_arg, Interaction)
    assert agent_interaction_arg.metadata["type"] == "agent_response"
    
    assert isinstance(agent_interaction_arg.contents, list)
    assert agent_interaction_arg.contents[0] == mock_agent_response