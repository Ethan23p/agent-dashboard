# controller.py
import asyncio
import random
from typing import TYPE_CHECKING

from model import Model, Interaction
from commands import ExitCommand, SwitchAgentCommand, ExitCommandImpl, SwitchCommand, ListAgentsCommand, SaveCommand, LoadCommand, ClearCommand
from rich.text import Text

if TYPE_CHECKING:
    from mcp_agent.core.agent_app import AgentApp


class Controller:
    """
    The Controller contains the application's business logic. It responds
    to user input from the View and orchestrates interactions between the
    Model and the Agent.
    """
    def __init__(self, model: Model):
        self.model = model
        self.agent_app: "AgentApp | None" = None
        self.agent = None
        self.command_map = {
            'exit': ExitCommandImpl(),
            'quit': ExitCommandImpl(),
            'save': SaveCommand(),
            'load': LoadCommand(),
            'clear': ClearCommand(),
            'switch': SwitchCommand(),
            'agents': ListAgentsCommand(),
        }

        from agent_registry import get_agent
        self.interpreter_agent_app = get_agent("interpreter")

    def link_agent_app(self, agent_app: "AgentApp"):
        """Link an agent app to the controller."""
        self.agent_app = agent_app
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
        """Parse and execute client-side commands."""
        parts = command_str.lower().split()
        command_name = parts[0][1:]
        args = parts[1:]

        command = self.command_map.get(command_name)
        if command:
            await command.execute(self, args)
        else:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Unknown command: /{command_name}"), tag="error")
            await self.model.add_interaction(error_interaction)

    async def _handle_agent_prompt(self, user_prompt: str):
        """
        Manage the full lifecycle of a conversational turn with the agent,
        with retry mechanism.
        """
        if self.agent is None:
            error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Agent has not been linked to the controller."), tag="error")
            await self.model.add_interaction(error_interaction)
            return

        # Tell the Model to update its state
        await self.model.add_user_turn(user_prompt)
        await self.model.set_thinking_status(True)

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Get the history FROM the model to send to the agent
                response_message = await self.agent.generate(
                    self.model.conversation_history
                )
                # Tell the Model to update its state with the response
                await self.model.add_assistant_turn(response_message)
                await self.model.set_thinking_status(False)

                if self.model.user_preferences.get("auto_save_enabled"):
                    from model import save_history
                    await save_history(self.model.conversation_history, self.model.user_preferences["auto_save_filename"])
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Agent Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s..."), tag="error")
                    await self.model.add_interaction(error_interaction)
                    await asyncio.sleep(delay)
                else:
                    await self.model.set_thinking_status(False)
                    if self.model.conversation_history: 
                        self.model.conversation_history.pop()
                    return

