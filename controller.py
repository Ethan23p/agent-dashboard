# controller.py
import asyncio
import random
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from model import Model, save_history, load_history, Interaction
from rich.text import Text

if TYPE_CHECKING:
    from mcp_agent.core.agent_app import AgentApp


class ExitCommand(Exception):
    """Custom exception to signal a graceful exit from the main loop."""
    pass


class SwitchAgentCommand(Exception):
    """Custom exception to signal switching to a different agent."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        super().__init__(f"Switch to agent: {agent_name}")


class Command(ABC):
    """Abstract base class for all commands."""
    @abstractmethod
    async def execute(self, controller: "Controller", args: list[str]):
        pass


class ExitCommandImpl(Command):
    """Command to exit the application."""
    async def execute(self, controller: "Controller", args: list[str]):
        from controller import ExitCommand as ExitException
        raise ExitException()


class SwitchCommand(Command):
    """Command to switch to a different agent."""
    async def execute(self, controller: "Controller", args: list[str]):
        if not args:
            error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Usage: /switch <agent_name>"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        
        agent_name = args[0]
        from agent_definitions import list_available_agents
        available_agents = list_available_agents()
        
        if agent_name not in available_agents:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        
        raise SwitchAgentCommand(agent_name)


class ListAgentsCommand(Command):
    """Command to list available agents."""
    async def execute(self, controller: "Controller", args: list[str]):
        from agent_definitions import list_available_agents
        available_agents = list_available_agents()
        success_interaction = Interaction(Text.from_markup(f"[bold green]Info:[/bold green] Available: {', '.join(available_agents)}"), tag="success")
        await controller.model.add_interaction(success_interaction)


class SaveCommand(Command):
    """Command to save conversation history to a file."""
    async def execute(self, controller: "Controller", args: list[str]):
        target_path = args[0] if args else controller.model.user_preferences["auto_save_filename"]
        
        success = await save_history(controller.conversation_history, target_path)
        if success:
            success_interaction = Interaction(Text.from_markup(f"[bold green]Success:[/bold green] History saved to {os.path.basename(target_path)}"), tag="success")
            await controller.model.add_interaction(success_interaction)
        else:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Failed to save history to {os.path.basename(target_path)}"), tag="error")
            await controller.model.add_interaction(error_interaction)


class LoadCommand(Command):
    """Command to load conversation history from a file."""
    async def execute(self, controller: "Controller", args: list[str]):
        if not args:
            error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Usage: /load <filename>"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        filename = args[0]
        if not os.path.isabs(filename):
            context_dir = controller.model._get_context_dir()
            filename = os.path.join(context_dir, filename)
        
        loaded_history = await load_history(filename)
        if loaded_history is not None:
            controller.conversation_history = loaded_history
            await controller.model.clear_log()
            for message in loaded_history:
                interaction = Interaction(Text.from_markup(f"[bold {'blue' if message.role == 'user' else 'magenta'}]{message.role.capitalize()}:[/] {message.last_text()}"))
                await controller.model.add_interaction(interaction)
            success_interaction = Interaction(Text.from_markup(f"[bold green]Success:[/bold green] History loaded from {os.path.basename(filename)}"), tag="success")
            await controller.model.add_interaction(success_interaction)
        else:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Failed to load history from {os.path.basename(filename)}"), tag="error")
            await controller.model.add_interaction(error_interaction)


class ClearCommand(Command):
    """Command to clear conversation history."""
    async def execute(self, controller: "Controller", args: list[str]):
        controller.conversation_history = []
        await controller.model.clear_log()
        success_interaction = Interaction(Text.from_markup("[bold green]Success:[/bold green] Conversation history cleared."), tag="success")
        await controller.model.add_interaction(success_interaction)


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

        self.conversation_history: list[PromptMessageMultipart] = []
        from agent_definitions import get_agent
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
        user_interaction = Interaction(Text.from_markup(f"[bold blue]You:[/bold blue] {user_input}"), tag="user_prompt")
        await self.model.add_interaction(user_interaction)
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

        user_message = Prompt.user(user_prompt)
        self.conversation_history.append(user_message)
        await self.model.set_thinking_status(True)

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response_message = await self.agent.generate(
                    self.conversation_history
                )
                self.conversation_history.append(response_message)
                
                agent_interaction = Interaction(
                    content=Text.from_markup(f"[bold magenta]Agent:[/bold magenta] {response_message.last_text()}"),
                    tag="agent_response"
                )
                await self.model.add_interaction(agent_interaction)
                await self.model.set_thinking_status(False)

                if self.model.user_preferences.get("auto_save_enabled"):
                    await save_history(self.conversation_history, self.model.user_preferences["auto_save_filename"])
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Agent Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s..."), tag="error")
                    await self.model.add_interaction(error_interaction)
                    await asyncio.sleep(delay)
                else:
                    await self.model.set_thinking_status(False)
                    if self.conversation_history: 
                        self.conversation_history.pop()
                    return

