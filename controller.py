# controller.py
import asyncio
import random
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
        # Get the first (default) agent without knowing its name
        # For now, we'll use the direct agent access since we only have one agent
        # This can be enhanced later when we support multiple agents
        self.agent = agent_app.agent

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

    async def _handle_command(self, command_str: str):
        """Parses and executes client-side commands like /save or /exit."""
        parts = command_str.lower().split()
        command_name = parts[0][1:]  # remove the '/'
        args = parts[1:]

        command_map = {
            'exit': self._cmd_exit,
            'quit': self._cmd_exit,
            'save': self._cmd_save,
            'load': self._cmd_load,
            'clear': self._cmd_clear,
        }

        handler = command_map.get(command_name)
        if handler:
            await handler(args)
        else:
            await self.model.set_state(AppState.ERROR, error_message=f"Unknown command: /{command_name}")

    async def _cmd_exit(self, args):
        raise ExitCommand()

    async def _cmd_save(self, args):
        filename = args[0] if args else None
        # If filename provided, ensure it's in the context directory
        if filename and not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = self.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        success = await self.model.save_history_to_file(filename)
        if success:
            await self.model.set_state(AppState.IDLE, success_message="History saved successfully.")
        else:
            await self.model.set_state(AppState.ERROR, error_message="Failed to save history.")

    async def _cmd_load(self, args):
        if not args:
            await self.model.set_state(AppState.ERROR, error_message="Please provide a filename: /load <filename>")
            return
        filename = args[0]
        # If filename doesn't start with path separator, assume it's in context directory
        if not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = self.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        success = await self.model.load_history_from_file(filename)
        if success:
            await self.model.set_state(AppState.IDLE, success_message="History loaded successfully.")
        else:
            await self.model.set_state(AppState.ERROR, error_message="Failed to load history.")

    async def _cmd_clear(self, args):
        await self.model.clear_history()
        await self.model.set_state(AppState.IDLE, success_message="Conversation history cleared.")


    async def _handle_agent_prompt(self, user_prompt: str):
        """
        Manages the full lifecycle of a conversational turn with the agent,
        now with a retry mechanism.
        """
        await self.model.set_state(AppState.AGENT_IS_THINKING)
        user_message = Prompt.user(user_prompt)
        await self.model.add_message(user_message)

        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                # The core agent call
                response_message = await self.agent.generate(
                    self.model.conversation_history
                )
                await self.model.add_message(response_message)
                
                # If successful, break the loop
                await self.model.set_state(AppState.IDLE)
                if self.model.user_preferences.get("auto_save_enabled"):
                    await self.model.save_history_to_file()
                return # Exit the method on success

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    await self.model.set_state(AppState.ERROR, error_message=f"Agent Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    await self.model.set_state(AppState.ERROR, error_message=f"Agent Error after {max_retries} attempts: {e}")
                    await self.model.pop_last_message() # Roll back the user message
                    return # Exit after final failure