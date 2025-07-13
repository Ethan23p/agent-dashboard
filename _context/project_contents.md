# Contents of the 'agent-dashboard' project

--- START OF FILE .gitignore ---

```
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

```

--- END OF FILE .gitignore ---

--- START OF FILE agent-dashboard.code-workspace ---

```code-workspace
{
	"folders": [
		{
			"name": "agent-dashboard",
			"path": "."
		},
		{
			"path": "../context_for_MCP_and_fast-agent"
		}
	],
	"settings": {}
}
```

--- END OF FILE agent-dashboard.code-workspace ---

--- START OF FILE agent_definitions.py ---

```py
# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.
#
# NOTE: All agents should use use_history=False since we manage conversation
# history ourselves in the Model class and pass it explicitly to the agent.

# A list of dictionaries, where each dictionary defines an agent.
# This is flexible – only include the keys you need for each agent.
AGENT_DEFINITIONS = [
    {
        "name": "minimal",
        "description": "A helpful assistant for general operations.",
        "instruction": """
        You are a helpful assistant that can perform various operations.
        You can read files, write files, and list directory contents.
        Always be helpful and provide clear responses to user requests.
        """,
        "servers": ["filesystem", "fetch", "sequential-thinking"],
        "max_tokens": 2048,
    },
    {
        "name": "coding",
        "description": "A specialized coding assistant.",
        "instruction": """
        You are a specialized coding assistant. You excel at:
        - Code review and suggestions
        - Debugging and problem-solving
        - Explaining complex technical concepts
        - Providing code examples and best practices
        
        Always provide clear, well-documented code examples when relevant.
        """,
        "servers": ["filesystem"],
        "max_tokens": 4096,
    },
    {
        "name": "interpreter",
        "description": "A structured data interpreter.",
        "instruction": """
        You are a highly efficient data parsing engine.
        Given a user's natural language text and a target JSON schema,
        your sole purpose is to extract the relevant information and respond
        ONLY with the JSON object that conforms to the schema.
        """,
        "use_history": False,
    },
]

def _create_agent_from_definition(definition: dict) -> FastAgent:
    """Factory function to build a FastAgent instance from a dictionary."""
    
    # Use .get() to provide defaults for optional keys
    description = definition.get("description", "A fast-agent.")
    instruction = definition.get("instruction", "You are a helpful assistant.")
    servers = definition.get("servers", [])
    max_tokens = definition.get("max_tokens", 2048)

    agent_instance = FastAgent(description)

    # The decorator needs a function to decorate, even a placeholder
    @agent_instance.agent(
        name="agent",
        instruction=instruction,
        servers=servers,
        request_params=RequestParams(maxTokens=max_tokens),
        use_history=False
    )
    async def placeholder_func(): pass
    
    return agent_instance

# The registry is now BUILT dynamically from the definitions list.
AGENT_REGISTRY = {}

# Populate the registry
for definition in AGENT_DEFINITIONS:
    agent_name = definition.get("name")
    if agent_name:
        AGENT_REGISTRY[agent_name] = _create_agent_from_definition(definition)

def get_agent(agent_name: str = "minimal"):
    """
    Get an agent by name from the registry.
    
    Args:
        agent_name: The name of the agent to retrieve
        
    Returns:
        The FastAgent instance for the requested agent
        
    Raises:
        KeyError: If the agent name is not found in the registry
    """
    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """Return a list of available agent names."""
    return list(AGENT_REGISTRY.keys())

```

--- END OF FILE agent_definitions.py ---

--- START OF FILE config/.python-version ---

```
3.13

```

--- END OF FILE config/.python-version ---

--- START OF FILE controller.py ---

```py
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


```

--- END OF FILE controller.py ---

--- START OF FILE docs/AGENT_SELECTION.md ---

```md
# Agent Selection System

The agent dashboard now supports flexible agent selection with the ability to switch between different agents at runtime.

## Available Agents

- **minimal**: A basic assistant for general operations
- **coding**: A specialized coding assistant with enhanced programming capabilities

## Usage

### Command Line Selection

Start with a specific agent:
```bash
python main.py --agent minimal
python main.py --agent coding
```

### Runtime Switching

While using the application, you can switch agents using commands:

- `/agents` - List all available agents
- `/switch <agent_name>` - Switch to a different agent

Example:
```
You: /agents
[SUCCESS] Available agents: minimal, coding

You: /switch coding
[SUCCESS] Switching to coding agent...
```

## Adding New Agents

To add a new agent, edit `agent_definitions.py`:

1. Create a new FastAgent instance:
```python
my_agent = FastAgent("My Agent Name")
```

2. Define the agent with decorator:
```python
@my_agent.agent(
    name="agent",
    instruction="Your agent instructions here...",
    servers=["filesystem"],
    request_params=RequestParams(maxTokens=2048),
    use_history=False
)
async def my_agent_func():
    pass
```

3. Add to the registry:
```python
AGENT_REGISTRY = {
    "minimal": minimal_agent,
    "coding": coding_agent,
    "my_agent": my_agent,  # Add your new agent here
}
```

## Architecture

The agent selection system uses:

- **Agent Registry**: Central registry mapping names to FastAgent instances
- **Command-line arguments**: Select initial agent
- **Runtime switching**: Switch agents during session
- **Exception-based flow control**: Clean agent switching without complex state management

## Testing

Run the test suite to verify agent selection works:
```bash
python test_agent_selection.py
``` 
```

--- END OF FILE docs/AGENT_SELECTION.md ---

--- START OF FILE docs/README.md ---

```md
# Agent Dashboard

A terminal client for the `fast-agent` framework.

This project started as a way to have a more stable and transparent interface for agent development. The core is a Model-View-Controller (MVC) architecture, separating the application's state from its terminal UI and logic.

## Technical Details

The client is built with a few key ideas in mind:

*   **Context Management.** Following the philosophy of the Model Context Protocol, the controller assembles the conversational history and other data to form the precise context sent to the agent on each turn. This allows for more deliberate, developer-driven context strategies.

*   **Asynchronous Core.** The application uses `asyncio` and a non-blocking prompt, which keeps the UI responsive. It's designed to support more complex operations, like parallel agent interactions, and could be adapted for a GUI dashboard later.

*   **Stateful History.** While the terminal shows a clean chat log, a comprehensive history is maintained in the background. This history can be saved automatically or manually, providing a useful artifact for debugging or resuming sessions.

*   **Resilient Operation.** LLM or MCP server errors are handled by the controller, which rolls back the conversational state to its last valid point. The application also shuts down cleanly to avoid resource errors.

*   **Comprehensive Testing.** The application includes a complete testing suite with unit tests, integration tests, and retry mechanisms to ensure reliability and maintainability.

## Testing

The project includes a comprehensive testing suite to ensure reliability and maintainability:

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py test_model.py

# Run with verbose output
python run_tests.py -v
```

### Test Structure

- **`test_model.py`**: Unit tests for the Model class, covering state management, conversation history, and file operations
- **`test_controller.py`**: Unit tests for the Controller class, including command parsing and agent interaction with retry logic
- **`test_integration.py`**: Integration tests that verify the interaction between Model and Controller components

### Test Features

- **Retry Logic**: The controller now includes exponential backoff retry logic for agent calls, making the application more resilient to temporary network or API issues
- **Mock Testing**: All tests use mocks to avoid external dependencies while thoroughly testing the application logic
- **Async Support**: Full async/await support for testing the asynchronous nature of the application

## Project Journey

This client evolved through several stages:

1.  Began with simple `fast-agent` scripts run from the command line.
2.  Integrated a few powerful MCP servers (`filesystem`, `memory`, `fetch`), which revealed the potential of the protocol.
3.  Shifted focus from thinking of `fast-agent` as a script runner to using it as a library within a client/server model.
4.  Adopted the MVC pattern to cleanly separate concerns.
5.  The result is this application—a stable tool for further agent development.

```

--- END OF FILE docs/README.md ---

--- START OF FILE fastagent.config.yaml ---

```yaml
# fastagent.config.yaml

# --- Model Configuration ---
# Set the default model for all agents.
# You can override this per-agent in the decorator or with the --model CLI flag.
# Format: <provider>.<model_name> (e.g., openai.gpt-4o, anthropic.claude-3-5-sonnet-latest)
# This project will use "google.gemini-2.5-flash" indefinitely. Don't change this, and don't downgrade it to 1.5 like you LLMs are want to do.
default_model: google.gemini-2.5-flash

# --- Logger Configuration ---
# This setup gives your client script full control over what is displayed.
logger:
  # Hide the default progress bar for a cleaner terminal experience.
  progress_display: false
  # We will print messages from our client script, so disable the default chat log.
  show_chat: false
  # We will handle tool display in our client script, so disable this too.
  show_tools: false

# --- MCP Server Configuration ---
# Defines the external tools and services available to your agents.
mcp:
  servers:
    # Fetch server for web scraping and data retrieval
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
    
    # Filesystem server for reading/writing local files
    filesystem:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-filesystem"
        - "G:/My Drive/AI Resources/Open collection"

    # Secure filesystem server for read-only access to specific directories
    secure-filesystem:
      command: "uv"
      args: ["run", "secure_filesystem_server.py", "G:/My Drive/AI Resources/Open collection"]

    # Memory server for persistent knowledge graph memory
    memory:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-memory"

    # Sequential Thinking server for dynamic and reflective problem-solving
    sequential-thinking:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-sequential-thinking"
```

--- END OF FILE fastagent.config.yaml ---

--- START OF FILE main.py ---

```py
# main.py
import asyncio
import sys
import argparse

from model import Model
from textual_view import AgentDashboardApp
from controller import Controller, SwitchAgentCommand
from agent_definitions import get_agent, list_available_agents

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

def parse_arguments():
    """Parse command line arguments for agent selection."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default="minimal",
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

class Application:
    """
    Manages the application's lifecycle and state.
    Handles agent sessions and switching between agents.
    """
    def __init__(self, initial_agent_name: str):
        self.current_agent_name = initial_agent_name

    async def run(self):
        """The main application loop that handles agent sessions and switching."""
        while self.current_agent_name is not None:
            next_agent = await self._run_single_session(self.current_agent_name)
            if next_agent:
                print(f"\nSwitching to {next_agent} agent...")
                await asyncio.sleep(0.1)
                self.current_agent_name = next_agent
            else:
                self.current_agent_name = None

        await asyncio.sleep(0.1)

    async def _run_single_session(self, agent_name: str) -> str | None:
        """Run a session with a specific agent using the Textual UI."""
        try:
            selected_agent = get_agent(agent_name)
            print(f"Starting {agent_name} agent...")
            
            async with selected_agent.run() as agent_app:
                model = Model()
                controller = Controller(model)
                controller.link_agent_app(agent_app)
                
                tui_app = AgentDashboardApp(model, controller, agent_name=agent_name)
                switch_to_agent = await tui_app.run_async()
                return switch_to_agent

        except SwitchAgentCommand as e:
            return e.agent_name
        except KeyError as e:
            print(f"Error: {e}")
            print(f"Available agents: {', '.join(list_available_agents())}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

async def main():
    """
    The main entry point for the application.
    """
    args = parse_arguments()
    app = Application(initial_agent_name=args.agent)
    await app.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print_shutdown_message()

```

--- END OF FILE main.py ---

--- START OF FILE model.py ---

```py
# model.py
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text


async def save_history(history: list[PromptMessageMultipart], filepath: str) -> bool:
    """Save conversation history to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable_history = [message.model_dump(mode='json') for message in history]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


async def load_history(filepath: str) -> list[PromptMessageMultipart] | None:
    """Load conversation history from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_history = json.load(f)
        return [PromptMessageMultipart(**data) for data in raw_history]
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


@dataclass
class Interaction:
    content: Text
    tag: str = "message"


# State classes
from abc import ABC
class IAppState(ABC): pass
class IdleState(IAppState): pass
class AgentIsThinkingState(IAppState): pass
class ErrorState(IAppState): pass


class Model:
    """
    The Model represents the single source of truth for the application's state.
    It holds all data and notifies listeners when its state changes.
    """
    def __init__(self):
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_log: list[Interaction] = []
        self.is_thinking: bool = False
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        
        # Initialize user preferences
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"

        self._listeners: List[Callable] = []

    def _get_context_dir(self) -> str:
        """Get the context directory from preferences."""
        return self.user_preferences.get("context_dir", "_context")

    async def _notify_listeners(self):
        """Notify all registered listeners of a state change."""
        for listener in self._listeners:
            await listener()

    def register_listener(self, listener: Callable):
        """Register a callback to be notified of state changes."""
        self._listeners.append(listener)

    async def add_interaction(self, interaction: Interaction):
        """Add an interaction to the conversation log."""
        self.conversation_log.append(interaction)
        await self._notify_listeners()

    async def clear_log(self):
        """Clear the conversation log."""
        self.conversation_log = []
        await self._notify_listeners()

    async def set_thinking_status(self, is_thinking: bool):
        """Set the agent's thinking status."""
        self.is_thinking = is_thinking
        await self._notify_listeners()
```

--- END OF FILE model.py ---

--- START OF FILE pyproject.toml ---

```toml
[project]
name = "agent-dashboard"
version = "0.1.0"
description = "A terminal-based agent dashboard for MCP agents"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.53.0",
    "mcp[cli]>=1.9.3",
    "python-dotenv>=1.1.0",
    "rich>=14.0.0",
    "prompt_toolkit>=3.0.0",
    "fast-agent-mcp>=0.2.40",
    "multidict>=6.5.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

```

--- END OF FILE pyproject.toml ---

--- START OF FILE secure_filesystem_server.py ---

```py
# secure_filesystem_server.py
# Needs to be validated; not sure this is the correct implementation.
import os
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP
import typer

# Initialize the FastMCP server
mcp = FastMCP("secure-filesystem")

def is_path_safe(base_dirs: List[Path], target_path: Path) -> bool:
    """Ensure the target path is within one of the allowed base directories."""
    resolved_path = target_path.resolve()
    for base in base_dirs:
        try:
            resolved_path.relative_to(base.resolve())
            return True
        except ValueError:
            continue
    return False

@mcp.tool()
def read_file(path: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Reads the complete contents of a single file."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_file():
        return f"Error: Path is not a file: {path}"
    return target_path.read_text(encoding="utf-8")

@mcp.tool()
def list_directory(path: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Lists the contents of a directory."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"
    
    contents = []
    for item in target_path.iterdir():
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        contents.append(f"{prefix} {item.name}")
    return "\n".join(contents)

@mcp.tool()
def search_files(path: str, pattern: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Recursively searches for files matching a pattern in a directory."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"

    matches = [str(p) for p in target_path.rglob(pattern)]
    return "\n".join(matches) if matches else "No matching files found."


@mcp.tool()
def list_allowed_directories(allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Lists all directories the server is allowed to access."""
    return "This server has read-only access to the following directories:\n" + "\n".join([str(d.resolve()) for d in allowed_dirs])


def main(allowed_dirs: List[Path] = typer.Argument(..., help="List of directories to allow read access to.")):
    """
    A read-only filesystem MCP server.
    This server will run until the client disconnects.
    """
    # Start the MCP server
    mcp.run(transport="stdio")

if __name__ == "__main__":
    typer.run(main)
```

--- END OF FILE secure_filesystem_server.py ---

--- START OF FILE tests/run_tests.py ---

```py
#!/usr/bin/env python3
"""
Simple test runner for the agent-dashboard project.
Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_model.py     # Run specific test file
    python run_tests.py -v                # Run with verbose output
"""

import sys
import subprocess
import os


def run_tests(test_file=None, verbose=False):
    """Run pytest with the specified options."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_file:
        cmd.append(test_file)
    else:
        # Run all test files
        cmd.extend(["test_model.py", "test_controller.py", "test_integration.py"])
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-asyncio")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for agent-dashboard")
    parser.add_argument("test_file", nargs="?", help="Specific test file to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🧪 Running tests for agent-dashboard...")
    success = run_tests(args.test_file, args.verbose)
    
    sys.exit(0 if success else 1) 
```

--- END OF FILE tests/run_tests.py ---

--- START OF FILE tests/test_agent_selection.py ---

```py
#!/usr/bin/env python3
"""
Test script for agent selection functionality.
"""

import asyncio
from agent_definitions import get_agent, list_available_agents, AGENT_REGISTRY

def test_agent_registry():
    """Test the agent registry functionality."""
    print("Testing Agent Registry...")
    
    # Test listing available agents
    available_agents = list_available_agents()
    print(f"Available agents: {available_agents}")
    assert len(available_agents) >= 2, "Should have at least 2 agents"
    
    # Test getting valid agents
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")
    print("✓ Successfully retrieved minimal and coding agents")
    
    # Test getting invalid agent
    try:
        get_agent("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✓ Correctly raised KeyError for invalid agent: {e}")
    
    print("All agent registry tests passed!")

def test_agent_characteristics():
    """Test that agents have different characteristics."""
    print("\nTesting Agent Characteristics...")
    
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")
    
    # Check that they're different instances
    assert minimal_agent != coding_agent, "Agents should be different instances"
    
    # Check that they have different names
    assert minimal_agent.name != coding_agent.name, "Agents should have different names"
    
    print("✓ Agents have different characteristics")
    print(f"  Minimal agent: {minimal_agent.name}")
    print(f"  Coding agent: {coding_agent.name}")
    
    print("All agent characteristics tests passed!")

if __name__ == "__main__":
    test_agent_registry()
    test_agent_characteristics()
    print("\n🎉 All tests passed! Agent selection system is working correctly.") 
```

--- END OF FILE tests/test_agent_selection.py ---

--- START OF FILE tests/test_controller.py ---

```py
import pytest
from unittest.mock import AsyncMock, MagicMock
from controller import Controller, ExitCommand
from model import Model, AppState
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_exit_command():
    """Test that the exit command raises ExitCommand exception."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    with pytest.raises(ExitCommand):
        await controller.process_user_input("/exit")

    with pytest.raises(ExitCommand):
        await controller.process_user_input("/quit")


@pytest.mark.asyncio
async def test_save_command():
    """Test the save command functionality."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    # Test save with default filename
    await controller.process_user_input("/save")
    mock_model.save_history_to_file.assert_called_once_with(None)

    # Test save with custom filename
    await controller.process_user_input("/save test_file.json")
    mock_model.save_history_to_file.assert_called_with("test_file.json")


@pytest.mark.asyncio
async def test_load_command():
    """Test the load command functionality."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    # Test load with filename
    await controller.process_user_input("/load test_file.json")
    mock_model.load_history_from_file.assert_called_once_with("test_file.json")


@pytest.mark.asyncio
async def test_clear_command():
    """Test the clear command functionality."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("/clear")
    mock_model.clear_history.assert_called_once()
    mock_model.set_state.assert_called_with(AppState.IDLE, success_message="Conversation history cleared.")


@pytest.mark.asyncio
async def test_unknown_command():
    """Test handling of unknown commands."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("/unknown")
    mock_model.set_state.assert_called_with(AppState.ERROR, error_message="Unknown command: /unknown")


@pytest.mark.asyncio
async def test_empty_input():
    """Test that empty input is handled gracefully."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("")
    await controller.process_user_input("   ")
    
    # Should not call any agent methods
    mock_agent_app.agent.generate.assert_not_called()


@pytest.mark.asyncio
async def test_successful_agent_prompt():
    """Test successful agent prompt handling."""
    mock_model = AsyncMock()
    mock_model.conversation_history = []
    mock_model.user_preferences = {"auto_save_enabled": False}
    
    mock_agent = AsyncMock()
    mock_response = MagicMock()
    mock_response.role = 'assistant'
    mock_response.content = [{'type': 'text', 'text': 'Mocked response'}]
    mock_agent.generate.return_value = mock_response
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    # Verify the flow
    mock_model.set_state.assert_called_with(AppState.AGENT_IS_THINKING)
    mock_model.add_message.assert_called()
    mock_agent.generate.assert_called_once()
    mock_model.set_state.assert_called_with(AppState.IDLE)


@pytest.mark.asyncio
async def test_agent_prompt_with_retry():
    """Test agent prompt handling with retry logic."""
    mock_model = AsyncMock()
    mock_model.conversation_history = []
    mock_model.user_preferences = {"auto_save_enabled": False}
    
    mock_agent = AsyncMock()
    # First call fails, second call succeeds
    mock_agent.generate.side_effect = [Exception("Network error"), MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Success'}])]
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    # Should have been called twice (retry)
    assert mock_agent.generate.call_count == 2
    # Should have set error state during retry
    mock_model.set_state.assert_any_call(AppState.ERROR, error_message=pytest.approx("Agent Error (attempt 1/3): Network error. Retrying in", rel=0.1))


@pytest.mark.asyncio
async def test_agent_prompt_final_failure():
    """Test agent prompt handling when all retries fail."""
    mock_model = AsyncMock()
    mock_model.conversation_history = []
    mock_model.user_preferences = {"auto_save_enabled": False}
    
    mock_agent = AsyncMock()
    # All calls fail
    mock_agent.generate.side_effect = Exception("Persistent error")
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    # Should have been called 3 times (max retries)
    assert mock_agent.generate.call_count == 3
    # Should have rolled back the user message
    mock_model.pop_last_message.assert_called_once()
    # Should have set final error state
    mock_model.set_state.assert_any_call(AppState.ERROR, error_message="Agent Error after 3 attempts: Persistent error") 
```

--- END OF FILE tests/test_controller.py ---

--- START OF FILE tests/test_integration.py ---

```py
import pytest
from unittest.mock import AsyncMock, MagicMock
from model import Model, AppState
from controller import Controller
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_prompt_handling_integration():
    """Test the full integration between Model and Controller for prompt handling."""
    model = Model()
    
    # Mock the agent_app and the agent's generate method
    mock_agent = AsyncMock()
    mock_response = MagicMock()
    mock_response.role = 'assistant'
    mock_response.content = [{'type': 'text', 'text': 'Mocked response'}]
    mock_agent.generate.return_value = mock_response
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    assert len(model.conversation_history) == 2
    assert model.conversation_history[0].role == 'user'
    assert model.conversation_history[1].role == 'assistant'
    assert model.conversation_history[1].last_text() == 'Mocked response'


@pytest.mark.asyncio
async def test_command_integration():
    """Test the integration of command handling with the Model."""
    model = Model()
    mock_agent_app = MagicMock()
    controller = Controller(model, mock_agent_app)

    # Test save command integration
    await controller.process_user_input("/save test_integration.json")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message == "History saved successfully."

    # Test clear command integration
    await controller.process_user_input("/clear")
    assert len(model.conversation_history) == 0
    assert model.last_success_message == "Conversation history cleared."


@pytest.mark.asyncio
async def test_error_handling_integration():
    """Test error handling integration between Model and Controller."""
    model = Model()
    
    # Mock agent that always fails
    mock_agent = AsyncMock()
    mock_agent.generate.side_effect = Exception("Test error")
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    # Add a message first to test rollback
    await model.add_message(Prompt.user("Previous message"))
    initial_history_length = len(model.conversation_history)

    await controller.process_user_input("This will fail")

    # Should have rolled back the user message
    assert len(model.conversation_history) == initial_history_length
    assert model.application_state == AppState.ERROR
    assert "Test error" in model.last_error_message


@pytest.mark.asyncio
async def test_state_management_integration():
    """Test that state management works correctly across the integration."""
    model = Model()
    mock_agent_app = MagicMock()
    controller = Controller(model, mock_agent_app)

    # Test that state changes are properly managed
    assert model.application_state == AppState.IDLE
    
    # Simulate a command that changes state
    await controller.process_user_input("/clear")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message is not None


@pytest.mark.asyncio
async def test_conversation_flow_integration():
    """Test a complete conversation flow with multiple turns."""
    model = Model()
    
    # Mock agent that returns different responses
    mock_agent = AsyncMock()
    responses = [
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'First response'}]),
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Second response'}]),
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Third response'}])
    ]
    mock_agent.generate.side_effect = responses
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    # Simulate a conversation
    await controller.process_user_input("First message")
    await controller.process_user_input("Second message")
    await controller.process_user_input("Third message")

    assert len(model.conversation_history) == 6  # 3 user + 3 assistant messages
    assert model.conversation_history[0].role == 'user'
    assert model.conversation_history[1].role == 'assistant'
    assert model.conversation_history[2].role == 'user'
    assert model.conversation_history[3].role == 'assistant'
    assert model.conversation_history[4].role == 'user'
    assert model.conversation_history[5].role == 'assistant' 
```

--- END OF FILE tests/test_integration.py ---

--- START OF FILE tests/test_model.py ---

```py
import pytest
import tempfile
import os
from model import Model, AppState
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_model_initial_state():
    """Test that the model starts in the correct initial state."""
    model = Model()
    assert model.application_state == AppState.IDLE
    assert len(model.conversation_history) == 0
    assert model.last_error_message is None
    assert model.last_success_message is None


@pytest.mark.asyncio
async def test_model_state_change():
    """Test that state changes work correctly."""
    model = Model()
    assert model.application_state == AppState.IDLE
    
    await model.set_state(AppState.ERROR, "Test Error")
    assert model.application_state == AppState.ERROR
    assert model.last_error_message == "Test Error"
    
    await model.set_state(AppState.IDLE, "Test Success")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message == "Test Success"


@pytest.mark.asyncio
async def test_add_message():
    """Test adding messages to conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    assert len(model.conversation_history) == 1
    assert model.conversation_history[0].role == 'user'
    
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    assert model.conversation_history[1].role == 'assistant'


@pytest.mark.asyncio
async def test_pop_last_message():
    """Test removing the last message from conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    
    await model.pop_last_message()
    assert len(model.conversation_history) == 1
    assert model.conversation_history[0].role == 'user'


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing the conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    
    await model.clear_history()
    assert len(model.conversation_history) == 0


@pytest.mark.asyncio
async def test_save_and_load_history():
    """Test saving and loading conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    
    # Test saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
    
    try:
        success = await model.save_history_to_file(temp_filename)
        assert success is True
        
        # Test loading
        new_model = Model()
        success = await new_model.load_history_from_file(temp_filename)
        assert success is True
        assert len(new_model.conversation_history) == 2
        assert new_model.conversation_history[0].role == 'user'
        assert new_model.conversation_history[1].role == 'assistant'
        
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@pytest.mark.asyncio
async def test_user_preferences():
    """Test user preferences functionality."""
    model = Model()
    
    # Test default preferences
    assert model.user_preferences.get("auto_save_enabled") is True
    
    # Test setting preferences
    model.user_preferences["auto_save_enabled"] = False
    assert model.user_preferences.get("auto_save_enabled") is False
    
    model.user_preferences["test_setting"] = "test_value"
    assert model.user_preferences.get("test_setting") == "test_value" 
```

--- END OF FILE tests/test_model.py ---

--- START OF FILE textual_view.py ---

```py
# textual_view.py
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.containers import Vertical

from controller import ExitCommand, SwitchAgentCommand
from model import Model, Interaction

if TYPE_CHECKING:
    from controller import Controller


class AgentDashboardApp(App):
    """The Textual-based user interface for the agent dashboard."""

    CSS = """
    Screen {
        background: $surface;
    }
    #chat-log {
        margin: 1 2;
        border: round $primary;
        background: $panel;
    }
    Input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """
    BINDINGS = [
        ("ctrl+d", "toggle_dark", "Toggle Dark Mode"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model: Model, controller: "Controller", agent_name: str = "agent"):
        super().__init__()
        self.model = model
        self.controller = controller
        self.agent_name = agent_name
        self._last_rendered_message_count = 0
        self.model.register_listener(self.on_model_update)

    def compose(self) -> ComposeResult:
        """Create the core UI widgets."""
        yield Header()
        yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=True)
        yield Input(placeholder="Enter your prompt or type /help...")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when first mounted."""
        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        self.title = "Agent Dashboard"
        self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"
        self.log_widget.write("🤖 Agent is ready. Say 'Hi' or type a command.")

    async def on_model_update(self) -> None:
        """Handle model state changes by updating the UI safely on the main thread."""
        self.call_later(self.render_log)
        self.call_later(self.update_header)

    def render_log(self) -> None:
        """Render the entire conversation log from the model."""
        self.log_widget.clear()
        for interaction in self.model.conversation_log:
            self.log_widget.write(interaction.content)

    def update_header(self) -> None:
        """Update the header based on the model's thinking status."""
        if self.model.is_thinking:
            self.sub_title = "🤔 Thinking..."
        else:
            self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value
        if not user_input:
            return
        
        self.input_widget.clear()
        self.run_worker(self.controller.process_user_input(user_input), exclusive=True)

```

--- END OF FILE textual_view.py ---

