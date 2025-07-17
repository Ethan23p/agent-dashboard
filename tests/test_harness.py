# tests/test_harness.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from main import Application
from model import Task
from mcp_agent.core.prompt import Prompt
from textual_view import AgentDashboardApp
from controller import Controller, ExitCommand, SwitchAgentCommand

# This is our mock agent that will be returned by the patched get_agent
mock_agent_instance = MagicMock()
mock_agent_instance.run = MagicMock()

# We need an async context manager for `async with agent.run()...`
mock_agent_context = AsyncMock()
mock_agent_instance.run.return_value = mock_agent_context

# The agent object itself inside the context
mock_agent_object = AsyncMock()
# The generate method is what we really care about
mock_agent_object.generate = AsyncMock(
    return_value=Prompt.assistant("This is a mocked response.")
)
# Make the context manager return the agent object
mock_agent_context.__aenter__.return_value = MagicMock(minimal=mock_agent_object)


@pytest.mark.asyncio
@patch('controller.get_agent', return_value=mock_agent_instance)
async def test_end_to_end_task_execution(mock_get_agent):
    """
    Tests the full application lifecycle from user input to task completion
    using the Textual Pilot.
    """
    # 1. Run the app headlessly with the Pilot
    tui_app = AgentDashboardApp(agent_name="minimal")
    async with tui_app.run_test() as pilot:
        # 2. Simulate user typing a prompt and pressing enter
        prompt = "Analyze this data"
        await pilot.press(*prompt)
        await pilot.press("enter")

        # 3. Wait for the UI and any immediate workers to settle
        await pilot.pause()

        # 4. Assert that a task was created in the model
        assert len(tui_app.model.tasks) == 1
        task = tui_app.model.tasks[0]
        assert task.prompt == prompt
        assert task.status in ("running", "completed")

        # 5. Wait for all background workers to complete
        await pilot.wait_for_scheduled_animations()
        await tui_app.workers.wait_for_complete()

        # 6. Assert that the task is now completed
        completed_task = tui_app.model.get_task(task.id)
        assert completed_task is not None
        assert completed_task.status == "completed"
        assert completed_task.result == "This is a mocked response."

        # 7. Verify that the agent was called correctly
        mock_get_agent.assert_called_once_with("minimal")
        mock_agent_object.generate.assert_called_once()