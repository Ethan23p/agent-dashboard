# controller.py
import asyncio
import random
from typing import TYPE_CHECKING, Dict
from datetime import datetime

from commands import ExitCommand, SwitchAgentCommand, ExitCommandImpl, SwitchCommand, ListAgentsCommand, SaveCommand, LoadCommand, ClearCommand, TaskCommand
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
            'task': TaskCommand(),
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
            await self._continue_or_create_task(stripped_input)

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

    async def _continue_or_create_task(self, user_prompt: str):
        """
        Continues the active task or creates a new one, then starts a worker.
        """
        task = self.model.get_active_task()
        if task and task.status in ("completed", "pending", "failed"):
            # Continue the existing active task
            await self.model.add_user_turn_to_task(task.id, user_prompt)
        else:
            # No suitable task to continue, create a new one
            task = await self.model.create_task(user_prompt, self.model.default_agent_name)
        
        if task:
            self.app.run_worker(self._execute_task(task), exclusive=False, group="agent_tasks")

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
                # Add the clean user prompt to the UI log
                user_interaction = Interaction(
                    content=Text.from_markup(f"[bold blue]You:[/bold blue] {task.prompt}"),
                    tag="user_prompt",
                    meta={"timestamp": datetime.now().isoformat(), "task_id": task.id}
                )
                await self.model.add_interaction(user_interaction)

                async with agent_instance.run() as agent_app:
                    agent = agent_app[task.agent_name]
                    
                    history_before = len(task.conversation_history)
                    final_response_message = await agent.generate(task.conversation_history)
                    full_turn_history = agent.message_history[history_before:]
                    
                    await self.model.update_task_history(task.id, full_turn_history)
                    await self.model.update_task(task.id, status="completed", result=final_response_message.last_text())

                    agent_interaction = Interaction(
                        content=Text.from_markup(f"[bold magenta]Agent:[/bold magenta] {final_response_message.last_text()}"),
                        tag="agent_response",
                        meta={"timestamp": datetime.now().isoformat(), "task_id": task.id}
                    )
                    await self.model.add_interaction(agent_interaction)

                if self.model.user_preferences.get("auto_save_enabled"):
                    updated_task = self.model.get_task(task.id)
                    if updated_task is not None:
                        await self.model.save_task_history(updated_task)
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

