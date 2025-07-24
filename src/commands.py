import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from model import Interaction, save_session, load_session
from rich.text import Text
from agent_registry import list_available_agents

if TYPE_CHECKING:
    from controller import Controller

logger = logging.getLogger(__name__)

class ExitCommand(Exception):
    """Custom exception to signal a graceful exit."""
    pass

class SwitchAgentCommand(Exception):
    """Custom exception to signal an agent switch."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        super().__init__(f"Switch to agent: {agent_name}")

class Command(ABC):
    """Abstract base class for all commands."""
    @abstractmethod
    async def execute(self, controller: "Controller", args: List[str]):
        pass

class ExitCommandImpl(Command):
    """Exits the application."""
    async def execute(self, controller: "Controller", args: List[str]):
        logger.info("Exit command received.")
        raise ExitCommand()

class SwitchCommand(Command):
    """Switches to a different agent."""
    async def execute(self, controller: "Controller", args: List[str]):
        if not args:
            error_interaction = Interaction(
                Text.from_markup("[bold red]Error:[/bold red] Usage: /switch <agent_name>"),
                metadata={"user-facing": True, "type": "error"}
            )
            await controller.model.add_interaction_to_active_session(error_interaction)
            return
        
        agent_name = args[0]
        available = list_available_agents()
        
        if agent_name not in available:
            error_interaction = Interaction(
                Text.from_markup(f"[bold red]Error:[/bold red] Agent '{agent_name}' not found. Available: {', '.join(available)}"),
                metadata={"user-facing": True, "type": "error"}
            )
            await controller.model.add_interaction_to_active_session(error_interaction)
            return
        
        # The exception is caught by the view to trigger the agent switch.
        raise SwitchAgentCommand(agent_name)

class ListAgentsCommand(Command):
    """Lists available agents."""
    async def execute(self, controller: "Controller", args: List[str]):
        available = list_available_agents()
        info_interaction = Interaction(
            Text.from_markup(f"[bold green]Info:[/bold green] Available agents: {', '.join(available)}"),
            metadata={"user-facing": True, "type": "info"}
        )
        await controller.model.add_interaction_to_active_session(info_interaction)

class SaveCommand(Command):
    """Saves the active session to a file."""
    async def execute(self, controller: "Controller", args: List[str]):
        active_session = controller.model.get_active_session()
        if not active_session:
            error_interaction = Interaction(
                Text.from_markup("[bold red]Error:[/bold red] No active session to save."),
                metadata={"user-facing": True, "type": "error"}
            )
            await controller.model.add_interaction_to_active_session(error_interaction)
            return

        context_dir = controller.model.user_preferences.get("context_dir", "_context")
        target_path = args[0] if args else f"{context_dir}/{active_session.id}.json"

        success = await save_session(active_session, target_path)
        if success:
            success_interaction = Interaction(
                Text.from_markup(f"[bold green]Success:[/bold green] Session saved to {os.path.basename(target_path)}"),
                metadata={"user-facing": True, "type": "success"}
            )
            await controller.model.add_interaction_to_active_session(success_interaction)
        else:
            error_interaction = Interaction(
                Text.from_markup(f"[bold red]Error:[/bold red] Failed to save session to {os.path.basename(target_path)}"),
                metadata={"user-facing": True, "type": "error"}
            )
            await controller.model.add_interaction_to_active_session(error_interaction)

class LoadCommand(Command):
    """Loads a session from a file."""
    async def execute(self, controller: "Controller", args: List[str]):
        if not args:
            error_interaction = Interaction(
                Text.from_markup("[bold red]Error:[/bold red] Usage: /load <filename>"),
                metadata={"user-facing": True, "type": "error"}
            )
            await controller.model.add_interaction_to_active_session(error_interaction)
            return
        
        filename = args[0]
        loaded_session = await load_session(filename)
        if loaded_session:
            controller.model.sessions.append(loaded_session)
            await controller.model.set_active_session(loaded_session.id)
            success_interaction = Interaction(
                Text.from_markup(f"[bold green]Success:[/bold green] Session from {os.path.basename(filename)} loaded and activated."),
                metadata={"user-facing": True, "type": "success"}
            )
            await controller.model.add_interaction_to_active_session(success_interaction)
        else:
            error_interaction = Interaction(
                Text.from_markup(f"[bold red]Error:[/bold red] Failed to load session from {os.path.basename(filename)}"),
                metadata={"user-facing": True, "type": "error"}
            )
            await controller.model.add_interaction_to_active_session(error_interaction)

class ClearCommand(Command):
    """Clears all sessions and starts fresh."""
    async def execute(self, controller: "Controller", args: List[str]):
        await controller.model.clear_sessions()
        success_interaction = Interaction(
            Text.from_markup("[bold green]Success:[/bold green] All sessions cleared. New session started."),
            metadata={"user-facing": True, "type": "success"}
        )
        await controller.model.add_interaction_to_active_session(success_interaction)