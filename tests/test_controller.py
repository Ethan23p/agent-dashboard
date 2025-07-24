# tests/test_controller.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from controller import Controller, ExitCommand, SwitchAgentCommand
from model import Model
from primitives import Interaction
# LINTER FIX: Added the missing import for the Prompt helper.
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

@pytest.fixture
def mock_model() -> AsyncMock:
    """Fixture for a mocked Model."""
    model = AsyncMock(spec=Model)
    model.get_active_session.return_value = MagicMock()
    model.user_preferences = {"auto_save_enabled": True}
    return model

@pytest.fixture
def mock_app() -> MagicMock:
    """Fixture for a mocked Textual App."""
    return MagicMock()

@pytest.fixture
def controller(mock_model: AsyncMock, mock_app: MagicMock) -> Controller:
    """Fixture for a Controller instance with mocks."""
    return Controller(mock_model, mock_app)

@pytest.mark.asyncio
async def test_handle_exit_command(controller: Controller):
    """Test that the /exit command raises the ExitCommand exception."""
    with pytest.raises(ExitCommand):
        await controller.process_user_input("/exit")

@pytest.mark.asyncio
async def test_handle_switch_command(controller: Controller):
    """Test that the /switch command raises the SwitchAgentCommand exception."""
    with patch('commands.list_available_agents', return_value=['minimal', 'coding']):
        with pytest.raises(SwitchAgentCommand) as exc_info:
            await controller.process_user_input("/switch coding")
        assert exc_info.value.agent_name == "coding"

@pytest.mark.asyncio
async def test_handle_save_command(controller: Controller):
    """Test that the /save command calls the model's save method."""
    with patch('commands.save_session', new_callable=AsyncMock) as mock_save:
        await controller.process_user_input("/save")
        mock_save.assert_called_once()
        controller.model.add_interaction_to_active_session.assert_called_once() # type: ignore
        interaction_arg = controller.model.add_interaction_to_active_session.call_args[0][0] # type: ignore
        assert "Success" in str(interaction_arg.contents)

@pytest.mark.asyncio
async def test_handle_prompt_initiates_agent_turn(controller: Controller, mock_app: MagicMock):
    """Test that a user prompt correctly triggers a background worker."""
    await controller.process_user_input("Hello agent")
    
    controller.model.add_interaction_to_active_session.assert_called_once() # type: ignore
    interaction_arg = controller.model.add_interaction_to_active_session.call_args[0][0] # type: ignore
    assert interaction_arg.metadata["user-facing"] is True
    assert interaction_arg.contents[0].last_text() == "Hello agent"
    
    controller.model.save_active_session.assert_called_once() # type: ignore
    mock_app.run_worker.assert_called_once()

@pytest.mark.asyncio
@patch('controller.get_agent')
async def test_execute_agent_turn_success(mock_get_agent, controller: Controller):
    """Test a successful agent turn execution."""
    mock_agent_instance = MagicMock()
    mock_agent_app = AsyncMock()
    mock_agent = AsyncMock()
    mock_response = Prompt.assistant("Agent response")
    mock_agent.generate.return_value = mock_response
    mock_agent_app.__getitem__.return_value = mock_agent
    mock_agent_instance.run.return_value.__aenter__.return_value = mock_agent_app
    mock_get_agent.return_value = mock_agent_instance

    await controller._execute_agent_turn()

    controller.model.set_thinking_status.assert_any_call(True) # type: ignore
    controller.model.set_thinking_status.assert_any_call(False) # type: ignore

    assert controller.model.add_interaction_to_active_session.call_count == 1 # type: ignore
    interaction_arg = controller.model.add_interaction_to_active_session.call_args[0][0] # type: ignore
    assert interaction_arg.metadata["type"] == "agent_response"
    assert interaction_arg.contents[0].last_text() == "Agent response"

    controller.model.save_active_session.assert_called_once() # type: ignore