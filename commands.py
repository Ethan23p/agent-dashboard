# commands.py
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from model import Model, save_history, load_history, Interaction
from rich.text import Text

if TYPE_CHECKING:
    from controller import Controller


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
        raise ExitCommand()


class SwitchCommand(Command):
    """Command to switch to a different agent."""
    async def execute(self, controller: "Controller", args: list[str]):
        if not args:
            error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Usage: /switch <agent_name>"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        
        agent_name = args[0]
        from agent_registry import list_available_agents
        available_agents = list_available_agents()
        
        if agent_name not in available_agents:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        
        raise SwitchAgentCommand(agent_name)


class ListAgentsCommand(Command):
    """Command to list available agents."""
    async def execute(self, controller: "Controller", args: list[str]):
        from agent_registry import list_available_agents
        available_agents = list_available_agents()
        success_interaction = Interaction(Text.from_markup(f"[bold green]Info:[/bold green] Available: {', '.join(available_agents)}"), tag="success")
        await controller.model.add_interaction(success_interaction)


class SaveCommand(Command):
    """Command to save conversation history to a file."""
    async def execute(self, controller: "Controller", args: list[str]):
        target_path = args[0] if args else controller.model.user_preferences["auto_save_filename"]
        
        success = await save_history(controller.model.conversation_history, target_path)
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
            controller.model.conversation_history = loaded_history
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
        await controller.model.clear_log()
        success_interaction = Interaction(Text.from_markup("[bold green]Success:[/bold green] Conversation history cleared."), tag="success")
        await controller.model.add_interaction(success_interaction) 