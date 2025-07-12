# controller.py
import asyncio
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mcp_agent.core.prompt import Prompt
from model import Model, save_history, load_history, IdleState, AgentIsThinkingState, ErrorState

if TYPE_CHECKING:
    from mcp_agent.core.agent_app import AgentApp

# EXCEPTIONS

class ExitCommand(Exception):
    """Custom exception to signal a graceful exit from the main loop."""
    pass

class SwitchAgentCommand(Exception):
    """Custom exception to signal switching to a different agent."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        super().__init__(f"Switch to agent: {agent_name}")

# COMMAND PATTERN IMPLEMENTATION

class Command(ABC):
    """Abstract base class for all commands."""
    @abstractmethod
    async def execute(self, controller: "Controller", args: list[str]):
        pass

# CONCRETE COMMAND IMPLEMENTATIONS

class ExitCommandImpl(Command):
    """Command to exit the application."""
    async def execute(self, controller: "Controller", args: list[str]):
        from controller import ExitCommand as ExitException  # Avoid name clash
        raise ExitException()

class SwitchCommand(Command):
    """Command to switch to a different agent."""
    async def execute(self, controller: "Controller", args: list[str]):
        if not args:
            await controller.model.set_state(ErrorState(), error_message="Please provide an agent name: /switch <agent_name>")
            return
        
        agent_name = args[0]
        # Import here to avoid circular imports
        from agent_definitions import list_available_agents
        available_agents = list_available_agents()
        
        if agent_name not in available_agents:
            await controller.model.set_state(
                ErrorState(), 
                error_message=f"Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}"
            )
            return
        
        await controller.model.set_state(IdleState(), success_message=f"Switching to {agent_name} agent...")
        raise SwitchAgentCommand(agent_name)

class ListAgentsCommand(Command):
    """Command to list available agents."""
    async def execute(self, controller: "Controller", args: list[str]):
        from agent_definitions import list_available_agents
        available_agents = list_available_agents()
        await controller.model.set_state(
            IdleState(), 
            success_message=f"Available agents: {', '.join(available_agents)}"
        )

class SaveCommand(Command):
    """Command to save conversation history to a file."""
    async def execute(self, controller: "Controller", args: list[str]):
        filename = args[0] if args else None
        # If filename provided, ensure it's in the context directory
        if filename and not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = controller.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        else:
            filename = controller.model.user_preferences["auto_save_filename"]
        
        success = await save_history(controller.model.conversation_history, filename)
        if success:
            await controller.model.set_state(IdleState(), success_message="History saved successfully.")
        else:
            await controller.model.set_state(ErrorState(), error_message="Failed to save history.")

class LoadCommand(Command):
    """Command to load conversation history from a file."""
    async def execute(self, controller: "Controller", args: list[str]):
        if not args:
            await controller.model.set_state(ErrorState(), error_message="Please provide a filename: /load <filename>")
            return
        filename = args[0]
        # If filename doesn't start with path separator, assume it's in context directory
        if not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = controller.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        
        loaded_history = await load_history(filename)
        if loaded_history:
            controller.model.conversation_history = loaded_history
            await controller.model._notify_listeners()
            await controller.model.set_state(IdleState(), success_message="History loaded successfully.")
        else:
            await controller.model.set_state(ErrorState(), error_message="Failed to load history.")

class ClearCommand(Command):
    """Command to clear conversation history."""
    async def execute(self, controller: "Controller", args: list[str]):
        await controller.model.clear_history()
        await controller.model.set_state(IdleState(), success_message="Conversation history cleared.")

# MAIN CONTROLLER CLASS

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
        # The command map now holds INSTANCES of our command classes
        self.command_map = {
            'exit': ExitCommandImpl(),
            'quit': ExitCommandImpl(),
            'save': SaveCommand(),
            'load': LoadCommand(),
            'clear': ClearCommand(),
            'switch': SwitchCommand(),
            'agents': ListAgentsCommand(),
        }

    # PUBLIC INTERFACE

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

    # PRIVATE METHODS

    async def _handle_command(self, command_str: str):
        """Parses and executes client-side commands."""
        parts = command_str.lower().split()
        command_name = parts[0][1:]  # remove the '/'
        args = parts[1:]

        command = self.command_map.get(command_name)
        if command:
            # Polymorphism in action! We just call execute() on whatever object we get.
            await command.execute(self, args)
        else:
            await self.model.set_state(ErrorState(), error_message=f"Unknown command: /{command_name}")

    async def _handle_agent_prompt(self, user_prompt: str):
        """
        Manages the full lifecycle of a conversational turn with the agent,
        now with a retry mechanism.
        """
        await self.model.set_state(AgentIsThinkingState())
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
                await self.model.set_state(IdleState())
                if self.model.user_preferences.get("auto_save_enabled"):
                    await save_history(self.model.conversation_history, self.model.user_preferences["auto_save_filename"])
                return # Exit the method on success

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    await self.model.set_state(ErrorState(), error_message=f"Agent Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    await self.model.set_state(ErrorState(), error_message=f"Agent Error after {max_retries} attempts: {e}")
                    await self.model.pop_last_message() # Roll back the user message
                    return # Exit after final failure