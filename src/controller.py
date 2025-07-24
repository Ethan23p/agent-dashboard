import asyncio
import logging
import random
from typing import TYPE_CHECKING

from agent_registry import get_agent
from commands import (ClearCommand, ExitCommand, ExitCommandImpl,
                        ListAgentsCommand, LoadCommand, SaveCommand,
                        SwitchAgentCommand, SwitchCommand)
from mcp_agent.core.prompt import Prompt
from primitives import Interaction, Session
from rich.text import Text

if TYPE_CHECKING:
    from textual_view import AgentDashboardApp
    from model import Model

logger = logging.getLogger(__name__)

class Controller:
    """
    Contains the application's business logic, responding to user input from
    the View and orchestrating interactions between the Model and the Agent.
    """
    def __init__(self, model: "Model", app: "AgentDashboardApp"):
        self.model = model
        self.app = app
        self.command_map = {
            'exit': ExitCommandImpl(),
            'quit': ExitCommandImpl(),
            'save': SaveCommand(),
            'load': LoadCommand(),
            'clear': ClearCommand(),
            'switch': SwitchCommand(),
            'agents': ListAgentsCommand(),
        }

    async def process_user_input(self, user_input: str):
        """
        Handles user input, routing to either a command handler or the agent.
        """
        stripped_input = user_input.strip()
        if not stripped_input:
            return

        if stripped_input.lower().startswith('/'):
            await self._handle_command(stripped_input)
        else:
            await self._handle_prompt(stripped_input)

    async def _handle_command(self, command_str: str):
        """Parses and executes client-side commands."""
        parts = command_str.split()
        command_name = parts[0][1:].lower()
        args = parts[1:]

        command = self.command_map.get(command_name)
        if command:
            await command.execute(self, args)
        else:
            error_interaction = Interaction(
                Text.from_markup(f"[bold red]Error:[/bold red] Unknown command: /{command_name}"),
                metadata={"user-facing": True, "type": "error"}
            )
            await self.model.add_interaction_to_active_session(error_interaction)

    async def _handle_prompt(self, user_prompt: str):
        """Handles a user prompt by creating an interaction and calling the agent."""
        user_interaction = Interaction(
            contents=[Prompt.user(user_prompt)],
            metadata={"user-facing": True, "type": "user_prompt"}
        )
        await self.model.add_interaction_to_active_session(user_interaction)

        if self.model.user_preferences.get("auto_save_enabled"):
            await self.model.save_active_session()

        # Run the agent turn in the background to keep the UI responsive.
        self.app.run_worker(self._execute_agent_turn, exclusive=False, group="agent_turns")

    async def _execute_agent_turn(self):
        """
        Executes a single agent turn with retry logic for resilience.
        """
        active_session = self.model.get_active_session()
        if not active_session:
            logger.error("Attempted to execute agent turn with no active session.")
            return

        await self.model.set_thinking_status(True)

        max_retries = 3
        base_delay = 1.0

        try:
            agent_instance = get_agent(active_session.agent_name)
        except KeyError as e:
            error_interaction = Interaction(
                Text.from_markup(f"[bold red]Configuration Error:[/bold red] {e}"),
                metadata={"user-facing": True, "type": "error"}
            )
            await self.model.add_interaction_to_active_session(error_interaction)
            await self.model.set_thinking_status(False)
            return

        for attempt in range(max_retries):
            try:
                agent_history = self.model.get_agent_history_for_active_session()

                async with agent_instance.run() as agent_app:
                    agent = agent_app[active_session.agent_name]
                    response_message = await agent.generate(agent_history)

                agent_interaction = Interaction(
                    contents=[response_message],
                    metadata={"user-facing": True, "type": "agent_response", "agent_name": active_session.agent_name}
                )
                await self.model.add_interaction_to_active_session(agent_interaction)

                if self.model.user_preferences.get("auto_save_enabled"):
                    await self.model.save_active_session()

                await self.model.set_thinking_status(False)
                return

            except Exception as e:
                logger.error(f"Agent turn failed on attempt {attempt + 1}/{max_retries}", exc_info=True)
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    retry_interaction = Interaction(
                        Text.from_markup(f"[bold yellow]Agent Error (attempt {attempt + 1}):[/] Retrying in {delay:.2f}s..."),
                        metadata={"user-facing": True, "type": "warning"}
                    )
                    await self.model.add_interaction_to_active_session(retry_interaction)
                    await asyncio.sleep(delay)
                else:
                    final_error_interaction = Interaction(
                        Text.from_markup(f"[bold red]Agent Error:[/bold red] Failed after {max_retries} attempts. Please try again.\nDetails: {e}"),
                        metadata={"user-facing": True, "type": "error"}
                    )
                    await self.model.add_interaction_to_active_session(final_error_interaction)
                    await self.model.set_thinking_status(False)
                    return
