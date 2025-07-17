# commands.py
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from model import Model, save_history, load_history, Interaction, Task
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
    async def execute(self, controller: "Controller", args: List[str]):
        pass


class ExitCommandImpl(Command):
    """Command to exit the application."""
    async def execute(self, controller: "Controller", args: List[str]):
        raise ExitCommand()


class SwitchCommand(Command):
    """Command to switch to a different agent."""
    async def execute(self, controller: "Controller", args: List[str]):
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
    async def execute(self, controller: "Controller", args: List[str]):
        from agent_registry import list_available_agents
        available_agents = list_available_agents()
        success_interaction = Interaction(Text.from_markup(f"[bold green]Info:[/bold green] Available: {', '.join(available_agents)}"), tag="success")
        await controller.model.add_interaction(success_interaction)


class SaveCommand(Command):
    """Command to save conversation history to a file."""
    async def execute(self, controller: "Controller", args: List[str]):
        target_path = args[0] if args else controller.model.user_preferences["auto_save_filename"]

        # For now, save the history of the most recent task
        last_task = controller.model.get_last_task()
        if not last_task:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] No tasks to save."), tag="error")
            await controller.model.add_interaction(error_interaction)
            return

        success = await save_history(last_task.conversation_history, target_path)
        if success:
            success_interaction = Interaction(Text.from_markup(f"[bold green]Success:[/bold green] History saved to {os.path.basename(target_path)}"), tag="success")
            await controller.model.add_interaction(success_interaction)
        else:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Failed to save history to {os.path.basename(target_path)}"), tag="error")
            await controller.model.add_interaction(error_interaction)


class LoadCommand(Command):
    """Command to load conversation history from a file."""
    async def execute(self, controller: "Controller", args: List[str]):
        if not args:
            error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Usage: /load <filename>"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        filename = args[0]

        loaded_history = await load_history(filename)
        if loaded_history is not None:
            # Create a new task from the loaded history
            prompt = loaded_history[0].last_text() if loaded_history else "Loaded from file"
            loaded_task = await controller.model.create_task(prompt, controller.model.default_agent_name)
            loaded_task.conversation_history = loaded_history
            loaded_task.status = "completed"
            await controller.model.update_task(loaded_task.id, conversation_history=loaded_history, status="completed")
            success_interaction = Interaction(Text.from_markup(f"[bold green]Success:[/bold green] History from {os.path.basename(filename)} loaded as new task."), tag="success")
            await controller.model.add_interaction(success_interaction)
        else:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Failed to load history from {os.path.basename(filename)}"), tag="error")
            await controller.model.add_interaction(error_interaction)


class ClearCommand(Command):
    """Command to clear conversation history."""
    async def execute(self, controller: "Controller", args: List[str]):
        await controller.model.clear_tasks()
        success_interaction = Interaction(Text.from_markup("[bold green]Success:[/bold green] All tasks cleared."), tag="success")
        await controller.model.add_interaction(success_interaction) 