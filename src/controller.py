# controller.py
import asyncio
import random
from typing import TYPE_CHECKING, Dict

from commands import ExitCommand, SwitchAgentCommand, ExitCommandImpl, SwitchCommand, ListAgentsCommand, SaveCommand, LoadCommand, ClearCommand
from model import Model, Interaction, Task
from rich.text import Text
from textual import work
from agent_registry import get_agent

if TYPE_CHECKING:
    from textual_view import AgentDashboardApp


class Controller:
    """
    The Controller contains the application's business logic. It responds
    to user input from the View and orchestrates interactions between the
    Model and the Agent.
    """
    def __init__(self, model: Model, app: "AgentDashboardApp"):
        self.model = model
        self.app = app  # Store a reference to the app instance
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
            await self._create_and_run_task(stripped_input)

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

    async def _create_and_run_task(self, user_prompt: str):
        """
        Creates a new task and starts a background worker to execute it.
        """
        new_task = await self.model.create_task(user_prompt, self.model.default_agent_name)
        self.app.run_worker(self._execute_task(new_task), exclusive=False, group="agent_tasks")

    async def _execute_task(self, task: Task):
        """
        The background worker that executes a single agent task.
        This includes the full retry logic and state management for the task.
        """
        await self.model.set_thinking_status(True)
        await self.model.update_task(task.id, status="running")

        max_retries = 3
        base_delay = 1.0

        agent_instance = get_agent(task.agent_name)

        for attempt in range(max_retries):
            try:
                async with agent_instance.run() as agent_app:
                    agent = getattr(agent_app, task.agent_name)
                    response_message = await agent.generate(task.conversation_history)
                    await self.model.add_assistant_turn_to_task(task.id, response_message)
                    await self.model.update_task(task.id, status="completed", result=response_message.last_text())

                if self.model.user_preferences.get("auto_save_enabled"):
                    updated_task = self.model.get_task(task.id)
                    if updated_task:
                        from model import save_history
                        await save_history(updated_task.conversation_history, self.model.user_preferences["auto_save_filename"])
                await self.model.set_thinking_status(False)
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    error_interaction = Interaction(Text.from_markup(f"[bold red]Task '{task.id[:8]}' failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...[/]"), tag="error")
                    await self.model.add_interaction(error_interaction)
                    await asyncio.sleep(delay)
                else:
                    await self.model.update_task(task.id, status="failed", result=str(e))
                    await self.model.set_thinking_status(False)
                    return

