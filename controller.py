# controller.py
from typing import TYPE_CHECKING

from mcp_agent.core.prompt import Prompt
from model import AppState, Model

if TYPE_CHECKING:
    from mcp_agent.core.agent_app import AgentApp

class ExitCommand(Exception):
    """Custom exception to signal a graceful exit from the main loop."""
    pass

class Controller:
    """
    The Controller contains the application's business logic. It responds
    to user input from the View and orchestrates interactions between the
    Model and the Agent (fast-agent).
    """
    def __init__(self, model: Model, agent_app: "AgentApp"):
        self.model = model
        self.agent_app = agent_app
        self.agent = agent_app.base_agent

    async def process_user_input(self, user_input: str):
        """
        The main entry point for handling actions initiated by the user.
        It parses the input to determine if it's a command or a prompt
        for the agent.
        """
        stripped_input = user_input.strip()

        if not stripped_input:
            return

        if stripped_input.lower().startswith('/'):
            await self._handle_command(stripped_input)
        else:
            await self._handle_agent_prompt(stripped_input)

    async def _handle_command(self, command: str):
        """Parses and executes client-side commands like /save or /exit."""
        parts = command.lower().split()
        command_name = parts[0]

        if command_name in ('/exit', '/quit'):
            # Raise our custom exception to signal the main loop to exit.
            raise ExitCommand()

        elif command_name == '/save':
            filename = parts[1] if len(parts) > 1 else None
            success = await self.model.save_history_to_file(filename)
            # The view will be notified of the state change by the model.
        else:
            await self.model.set_state(AppState.ERROR, f"Unknown command: {command_name}")


    async def _handle_agent_prompt(self, user_prompt: str):
        """
        Manages the full lifecycle of a conversational turn with the agent.
        """
        await self.model.set_state(AppState.AGENT_IS_THINKING)
        user_message = Prompt.user(user_prompt)
        await self.model.add_message(user_message)

        try:
            response_message = await self.agent.generate(
                self.model.conversation_history
            )
            await self.model.add_message(response_message)
        except Exception as e:
            await self.model.set_state(AppState.ERROR, f"Agent Error: {e}")
            await self.model.pop_last_message()
        finally:
            if self.model.application_state != AppState.ERROR:
                await self.model.set_state(AppState.IDLE)
            if self.model.user_preferences.get("auto_save_enabled"):
                await self.model.save_history_to_file()