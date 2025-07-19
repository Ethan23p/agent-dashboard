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
venv/
env/

# IDE and Editor files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/

# Environment variables
.env
.env.local
.env.*.local

# Temporary files
*.tmp
*.temp
.cache/

# Coverage reports
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover

# pytest
.pytest_cache/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

```

--- END OF FILE .gitignore ---

--- START OF FILE .pytest_cache/.gitignore ---

```
# Created by pytest automatically.
*

```

--- END OF FILE .pytest_cache/.gitignore ---

--- START OF FILE .pytest_cache/CACHEDIR.TAG ---

```TAG
Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by pytest.
# For information about cache directory tags, see:
#	https://bford.info/cachedir/spec.html

```

--- END OF FILE .pytest_cache/CACHEDIR.TAG ---

--- START OF FILE .pytest_cache/README.md ---

```md
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

--- END OF FILE .pytest_cache/README.md ---

--- START OF FILE .pytest_cache/v/cache/lastfailed ---

```
{
  "tests/test_agent_selection.py::test_agent_characteristics": true
}
```

--- END OF FILE .pytest_cache/v/cache/lastfailed ---

--- START OF FILE .pytest_cache/v/cache/nodeids ---

```
[
  "tests/test_agent_selection.py::test_agent_characteristics",
  "tests/test_agent_selection.py::test_agent_registry",
  "tests/test_harness.py::test_end_to_end_task_execution"
]
```

--- END OF FILE .pytest_cache/v/cache/nodeids ---

--- START OF FILE paperwork/.python-version ---

```
3.13

```

--- END OF FILE paperwork/.python-version ---

--- START OF FILE paperwork/agent-dashboard.code-workspace ---

```code-workspace
{
	"folders": [
		{
			"name": "agent-dashboard",
			"path": ".."
		},
		{
			"path": "../../context_for_MCP_and_fast-agent"
		}
	],
	"settings": {}
}
```

--- END OF FILE paperwork/agent-dashboard.code-workspace ---

--- START OF FILE paperwork/CHANGELOG.md ---

```md
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized state management in Model class
- Unified vocabulary using "Interaction" terminology
- Separated command logic into dedicated `commands.py` module
- Renamed `agent_definitions.py` to `agent_registry.py` for clarity
- Fixed duplicate user message display issue
- Improved ExitCommand exception handling for graceful shutdown

### Changed
- Refactored Controller to be stateless
- Moved conversation history management to Model
- Updated all imports to use new module structure
- Reorganized project structure with `src/`, `paperwork/`, and `tests/` directories
- Moved all Python code to `src/` directory
- Grouped documentation and project files in `paperwork/` directory

## [0.1.0] - 2024-12-19

### Added
- Textual-based TUI overhaul with better modularization
- Agent switching functionality with `/switch` command
- Multiple agent support (minimal, coding, interpreter)
- Retry mechanism with exponential backoff for agent calls
- Comprehensive testing suite with unit and integration tests
- Model-View-Controller (MVC) architecture
- Asynchronous core with non-blocking UI
- Stateful conversation history with save/load functionality
- MCP server integration (filesystem, fetch, sequential-thinking)

### Changed
- Migrated from basic CLI to Textual-based interface
- Improved error handling and resilience
- Enhanced separation of concerns across codebase

### Fixed
- Visual bugs in Textual interface
- Error handling for transient failures
- Code organization and modularization

## [Initial Development] - 2024-12-18

### Added
- Minimum Viable Implementation with MVC structure
- Basic agent framework integration
- Initial client and agent framework setup
- Core functionality for agent interactions

---

*Note: This changelog is based on recent commit history. For a complete history, see the git log.* 
```

--- END OF FILE paperwork/CHANGELOG.md ---

--- START OF FILE pyproject.toml ---

```toml
[project]
name = "agent-dashboard"
version = "0.1.0"
description = "A terminal-based agent dashboard for MCP agents"
readme = "paperwork/README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.53.0",
    "mcp[cli]>=1.9.3",
    "python-dotenv>=1.1.0",
    "rich>=14.0.0",
    "prompt_toolkit>=3.0.0",
    "fast-agent-mcp>=0.2.41",
    "multidict>=6.5.1",
    "textual>=3.7.0",
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

--- START OF FILE README.md ---

```md
# Agent Dashboard

A terminal client for the `fast-agent` framework with a modern Textual-based UI.

This project started as a way to have a more stable and transparent interface for agent development. The core is a Model-View-Controller (MVC) architecture, separating the application's state from its terminal UI and logic.

## Features

- **Multiple Agent Support**: Switch between different specialized agents (minimal, coding, interpreter)
- **Modern Textual UI**: Clean, responsive terminal interface with command support
- **Context Management**: Comprehensive conversation history and state management
- **Resilient Operation**: Error handling with retry logic and clean shutdown
- **Command System**: Built-in commands for switching agents, saving/loading history, and more

## Available Agents

- **minimal**: General-purpose assistant with filesystem, fetch, and sequential-thinking capabilities
- **coding**: Specialized coding assistant with enhanced debugging and code review features
- **interpreter**: Structured data interpreter for JSON schema extraction

## Project Structure

```
agent-dashboard/
├── src/                           # Main application code
│   ├── main.py                   # Application entry point
│   ├── controller.py              # Business logic controller
│   ├── model.py                   # Data model and state management
│   ├── textual_view.py            # Textual-based UI
│   ├── agent_registry.py          # Agent definitions and registry
│   ├── commands.py                # Command implementations
│   ├── secure_filesystem_server.py # MCP filesystem server
│   └── fastagent.config.yaml     # FastAgent configuration
├── tests/                         # Test suite
├── paperwork/                     # Documentation and project files
│   ├── AGENT_SELECTION.md         # Agent selection guide
│   └── CHANGELOG.md              # Version history
└── _context/                      # Session history and context files
```

## Technical Details

The client is built with a few key ideas in mind:

*   **Context Management.** Following the philosophy of the Model Context Protocol, the controller assembles the conversational history and other data to form the precise context sent to the agent on each turn. This allows for more deliberate, developer-driven context strategies.

*   **Asynchronous Core.** The application uses `asyncio` and a non-blocking prompt, which keeps the UI responsive. It's designed to support more complex operations, like parallel agent interactions, and could be adapted for a GUI dashboard later.

*   **Stateful History.** While the terminal shows a clean chat log, a comprehensive history is maintained in the background. This history can be saved automatically or manually, providing a useful artifact for debugging or resuming sessions.

*   **Resilient Operation.** LLM or MCP server errors are handled by the controller, which rolls back the conversational state to its last valid point. The application also shuts down cleanly to avoid resource errors.

*   **Comprehensive Testing.** The application includes a complete testing suite with unit tests, integration tests, and retry mechanisms to ensure reliability and maintainability.

## Running the Application

```bash
# From the project root
uv run python src/main.py

# With specific agent
uv run python src/main.py --agent coding

# List available agents
uv run python src/main.py --help
```

## Using the Application

Once the application starts, you'll see a clean terminal UI with:

- **Chat Interface**: Type your messages and press Enter to send
- **Command System**: Use commands starting with `/`:
  - `/switch <agent>` - Switch to a different agent
  - `/agents` - List available agents
  - `/save [filename]` - Save conversation history
  - `/load <filename>` - Load conversation history
  - `/clear` - Clear current conversation
  - `/exit` or `/quit` - Exit the application

## Testing

The project includes a comprehensive testing suite to ensure reliability and maintainability:

### Running Tests

```bash
# Run all tests
uv run python tests/run_tests.py

# Run specific test file
uv run python -m pytest tests/test_model.py

# Run with verbose output
uv run python -m pytest tests/ -v
```

### Test Structure

- **`tests/test_model.py`**: Unit tests for the Model class, covering state management, conversation history, and file operations
- **`tests/test_controller.py`**: Unit tests for the Controller class, including command parsing and agent interaction with retry logic
- **`tests/test_integration.py`**: Integration tests that verify the interaction between Model and Controller components
- **`tests/test_agent_selection.py`**: Tests for agent switching functionality

### Test Features

- **Retry Logic**: The controller includes exponential backoff retry logic for agent calls, making the application more resilient to temporary network or API issues
- **Mock Testing**: All tests use mocks to avoid external dependencies while thoroughly testing the application logic
- **Async Support**: Full async/await support for testing the asynchronous nature of the application

## Configuration

The application uses `src/fastagent.config.yaml` for configuration, including:

- **Model Settings**: Default model and token limits
- **MCP Servers**: Filesystem, fetch, memory, and other server configurations
- **Logging**: Customizable logging and display options

## Project Journey

This client evolved through several stages:

1.  Began with simple `fast-agent` scripts run from the command line.
2.  Integrated a few powerful MCP servers (`filesystem`, `memory`, `fetch`), which revealed the potential of the protocol.
3.  Shifted focus from thinking of `fast-agent` as a script runner to using it as a library within a client/server model.
4.  Adopted the MVC pattern to cleanly separate concerns.
5.  Added a modern Textual-based UI with agent switching capabilities.
6.  The result is this application—a stable tool for further agent development.

```

--- END OF FILE README.md ---

--- START OF FILE src/agent_dashboard.egg-info/dependency_links.txt ---

```txt


```

--- END OF FILE src/agent_dashboard.egg-info/dependency_links.txt ---

--- START OF FILE src/agent_dashboard.egg-info/PKG-INFO ---

```
Metadata-Version: 2.4
Name: agent-dashboard
Version: 0.1.0
Summary: A terminal-based agent dashboard for MCP agents
Requires-Python: >=3.13
Description-Content-Type: text/markdown
Requires-Dist: anthropic>=0.53.0
Requires-Dist: mcp[cli]>=1.9.3
Requires-Dist: python-dotenv>=1.1.0
Requires-Dist: rich>=14.0.0
Requires-Dist: prompt_toolkit>=3.0.0
Requires-Dist: fast-agent-mcp>=0.2.41
Requires-Dist: multidict>=6.5.1
Requires-Dist: textual>=3.7.0
Provides-Extra: dev
Requires-Dist: pytest>=7.0.0; extra == "dev"
Requires-Dist: pytest-asyncio>=0.21.0; extra == "dev"

```

--- END OF FILE src/agent_dashboard.egg-info/PKG-INFO ---

--- START OF FILE src/agent_dashboard.egg-info/requires.txt ---

```txt
anthropic>=0.53.0
mcp[cli]>=1.9.3
python-dotenv>=1.1.0
rich>=14.0.0
prompt_toolkit>=3.0.0
fast-agent-mcp>=0.2.41
multidict>=6.5.1
textual>=3.7.0

[dev]
pytest>=7.0.0
pytest-asyncio>=0.21.0

```

--- END OF FILE src/agent_dashboard.egg-info/requires.txt ---

--- START OF FILE src/agent_dashboard.egg-info/SOURCES.txt ---

```txt
README.md
pyproject.toml
src/agent_registry.py
src/commands.py
src/controller.py
src/main.py
src/model.py
src/mood_server.py
src/secure_filesystem_server.py
src/textual_view.py
src/agent_dashboard.egg-info/PKG-INFO
src/agent_dashboard.egg-info/SOURCES.txt
src/agent_dashboard.egg-info/dependency_links.txt
src/agent_dashboard.egg-info/requires.txt
src/agent_dashboard.egg-info/top_level.txt
tests/test_agent_selection.py
tests/test_controller.py
tests/test_harness.py
tests/test_integration.py
tests/test_model.py
```

--- END OF FILE src/agent_dashboard.egg-info/SOURCES.txt ---

--- START OF FILE src/agent_dashboard.egg-info/top_level.txt ---

```txt
agent_registry
commands
controller
main
model
mood_server
secure_filesystem_server
textual_view

```

--- END OF FILE src/agent_dashboard.egg-info/top_level.txt ---

--- START OF FILE src/agent_registry.py ---

```py
# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from typing import Optional

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

def _register_agent_from_definition(agent_app: FastAgent, definition: dict) -> None:
    """Factory function to register an agent with a FastAgent instance."""
    
    # Use .get() to provide defaults for optional keys
    agent_name = definition.get("name", "minimal")
    instruction = definition.get("instruction", "You are a helpful assistant.")
    servers = definition.get("servers", [])
    max_tokens = definition.get("max_tokens", 2048)

    # The decorator needs a function to decorate, even a placeholder
    # This registers the agent with the *shared* agent_app instance
    @agent_app.agent(
        name=agent_name,
        instruction=instruction,
        servers=servers,
        request_params=RequestParams(maxTokens=max_tokens),
        use_history=False
    )
    async def placeholder_func(): pass

# The registry is now BUILT dynamically from the definitions list.
AGENT_REGISTRY: dict[str, FastAgent] = {}

# Create a single, shared FastAgent application instance
SHARED_AGENT_APP = FastAgent("Agent Dashboard", config_path="src/fastagent.config.yaml")

# Default agent (first one in the list)
DEFAULT_AGENT = AGENT_DEFINITIONS[0]["name"] if AGENT_DEFINITIONS else "minimal"

# Populate the registry
for definition in AGENT_DEFINITIONS:
    agent_name = definition.get("name")
    if agent_name:
        _register_agent_from_definition(SHARED_AGENT_APP, definition)
        AGENT_REGISTRY[agent_name] = SHARED_AGENT_APP

def get_agent(agent_name: Optional[str] = None):
    """
    Get an agent by name from the registry.
    
    Args:
        agent_name: The name of the agent to retrieve. If None, uses DEFAULT_AGENT
        
    Returns:
        The FastAgent instance for the requested agent
        
    Raises:
        KeyError: If the agent name is not found in the registry
    """
    
    if agent_name is None:
        agent_name = DEFAULT_AGENT

    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """Return a list of available agent names."""
    return list(AGENT_REGISTRY.keys())

```

--- END OF FILE src/agent_registry.py ---

--- START OF FILE src/commands.py ---

```py
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


class TaskCommand(Command):
    """Parent command for task management."""
    async def execute(self, controller: "Controller", args: List[str]):
        if not args:
            error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Usage: /task <new|switch|list> [args]"), tag="error")
            await controller.model.add_interaction(error_interaction)
            return
        
        subcommand = args[0]
        sub_args = args[1:]

        if subcommand == "new":
            prompt = " ".join(sub_args) or "New task started."
            await controller._create_and_run_task(prompt)
        elif subcommand == "switch":
            if not sub_args:
                error_interaction = Interaction(Text.from_markup("[bold red]Error:[/bold red] Usage: /task switch <task_id>"), tag="error")
                await controller.model.add_interaction(error_interaction)
                return
            task_id_prefix = sub_args[0]
            task_to_switch = next((t for t in controller.model.tasks if t.id.startswith(task_id_prefix)), None)
            if task_to_switch:
                controller.model.active_task_id = task_to_switch.id
                success_interaction = Interaction(Text.from_markup(f"[bold green]Success:[/bold green] Switched to task {task_to_switch.id}"), tag="success")
                await controller.model.add_interaction(success_interaction)
                await controller.model._notify_listeners() # Force header update
            else:
                error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Task with prefix '{task_id_prefix}' not found."), tag="error")
                await controller.model.add_interaction(error_interaction)
        elif subcommand == "list":
            task_list = "\n".join([f"- {task.id} ({task.status}): {task.prompt[:50]}..." for task in controller.model.tasks])
            info_interaction = Interaction(Text.from_markup(f"[bold]Available Tasks:[/bold]\n{task_list}"), tag="info")
            await controller.model.add_interaction(info_interaction)
        else:
            error_interaction = Interaction(Text.from_markup(f"[bold red]Error:[/bold red] Unknown task command: {subcommand}"), tag="error")
            await controller.model.add_interaction(error_interaction) 
```

--- END OF FILE src/commands.py ---

--- START OF FILE src/controller.py ---

```py
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


```

--- END OF FILE src/controller.py ---

--- START OF FILE src/fastagent.config.yaml ---

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

--- END OF FILE src/fastagent.config.yaml ---

--- START OF FILE src/main.py ---

```py
# main.py
import asyncio
import sys
import argparse
from typing import Optional

from model import Model
from textual_view import AgentDashboardApp
from controller import Controller
from agent_registry import list_available_agents, DEFAULT_AGENT

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

def parse_arguments():
    """Parse command line arguments for agent selection."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=DEFAULT_AGENT,
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

class Application:
    """
    The main application class that orchestrates the Model, View, and Controller.
    """
    def __init__(self, initial_agent_name: str):
        self.initial_agent_name = initial_agent_name

    async def run(self):
        """
        Initializes and runs the Textual user interface. The TUI now drives
        the application by sending user input to the controller.
        """
        tui_app = AgentDashboardApp(
            agent_name=self.initial_agent_name
        )
        await tui_app.run_async()

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

--- END OF FILE src/main.py ---

--- START OF FILE src/model.py ---

```py
# model.py
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Dict
import itertools

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.core.prompt import Prompt
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
    content: Text | PromptMessageMultipart
    tag: str = "message"
    meta: Dict = field(default_factory=dict)

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    status: str = "pending"  # pending, running, completed, failed
    agent_name: str = "minimal"
    conversation_history: List[PromptMessageMultipart] = field(default_factory=list)
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


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
        self.tasks: List[Task] = []
        self.interactions: List[Interaction] = []  # For the UI log
        self.is_thinking: bool = False
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        self.active_task_id: Optional[str] = None
        self.persona_cycler = itertools.cycle(["Hutter", "Aasimov", "Heinlein", "Chiang", "Borges"])
        # Initialize user preferences
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"
        self.default_agent_name: str = "minimal"
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
        self.interactions.append(interaction)
        await self._notify_listeners()
    
    async def add_interaction_from_message(self, message: PromptMessageMultipart, tag: str = "message"):
        """Helper to create and add an Interaction from a PromptMessageMultipart."""
        content_text = message.last_text() or ""
        interaction = Interaction(
            content=Text.from_markup(content_text),
            tag=tag,
            meta={"timestamp": datetime.now().isoformat()}
        )
        await self.add_interaction(interaction)
    
    def _generate_task_id(self) -> str:
        """Generates a new fun, thematic task ID."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        persona = next(self.persona_cycler)
        return f"{timestamp}-{persona}"

    async def create_task(self, prompt: str, agent_name: str) -> Task:
        """Creates a new task, adds it to the model, and returns it."""
        task_id = self._generate_task_id()
        task = Task(id=task_id, prompt=prompt, agent_name=agent_name)
        task.conversation_history.append(Prompt.user(prompt))
        self.tasks.append(task)
        self.active_task_id = task.id # The new task becomes active
        interaction = Interaction(Text.from_markup(f"[bold yellow]New Task '{task.id[:8]}':[/] {prompt}"), tag="task_created")
        await self.add_interaction(interaction)
        return task

    async def update_task(self, task_id: str, **updates):
        """Updates attributes of a specific task."""
        task = self.get_task(task_id)
        if task:
            for key, value in updates.items():
                setattr(task, key, value)
            if "status" in updates:
                interaction = Interaction(Text.from_markup(f"[dim]Task '{task.id[:8]}' status changed to {task.status}[/]"), tag="task_status")
                await self.add_interaction(interaction)
        await self._notify_listeners()

    async def update_task_history(self, task_id: str, new_history_parts: List[PromptMessageMultipart]):
        """Replaces the last user prompt with the full turn history from the agent."""
        task = self.get_task(task_id)
        if task:
            # Remove the last simple user prompt
            if task.conversation_history and task.conversation_history[-1].role == "user":
                task.conversation_history.pop()
            # Add the comprehensive history parts
            task.conversation_history.extend(new_history_parts)

    async def add_assistant_turn_to_task(self, task_id: str, response_message: PromptMessageMultipart):
        """Adds an assistant response to a specific task's history."""
        task = self.get_task(task_id)
        if task:
            task.conversation_history.append(response_message)
            agent_interaction = Interaction(
                content=Text.from_markup(f"[bold magenta]Task '{task_id[:8]}':[/] {response_message.last_text()}"),
                tag="agent_response"
            )
            await self.add_interaction(agent_interaction)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Find a task by its ID."""
        return next((task for task in self.tasks if task.id == task_id), None)

    def get_last_task(self) -> Optional[Task]:
        """Get the most recently created task."""
        return self.tasks[-1] if self.tasks else None

    async def clear_tasks(self):
        """Clear all tasks and interactions."""
        self.tasks = []
        self.interactions = []
        await self._notify_listeners()

    async def add_user_turn(self, user_input: str):
        """Adds a user turn to both the agent history and the UI log."""
        user_message = Prompt.user(user_input)
        self.conversation_history.append(user_message)
        user_interaction = Interaction(Text.from_markup(f"[bold blue]You:[/bold blue] {user_input}"), tag="user_prompt")
        self.interactions.append(user_interaction)
        await self._notify_listeners()

    async def add_assistant_turn(self, response_message: PromptMessageMultipart):
        """Adds an assistant turn to both the agent history and the UI log."""
        self.conversation_history.append(response_message)
        agent_interaction = Interaction(
            content=Text.from_markup(f"[bold magenta]Agent:[/bold magenta] {response_message.last_text()}"),
            tag="agent_response"
        )
        self.interactions.append(agent_interaction)
        await self._notify_listeners()

    async def clear_log(self):
        """Clear the conversation log."""
        self.interactions = []
        self.conversation_history = []
        await self._notify_listeners()

    async def set_thinking_status(self, is_thinking: bool):
        """Set the agent's thinking status."""
        self.is_thinking = is_thinking
        await self._notify_listeners()

    async def save_task_history(self, task: Task):
        await save_history(task.conversation_history, self.user_preferences["auto_save_filename"])

    async def add_user_turn_to_task(self, task_id: str, prompt: str):
        """Adds a new user prompt to an existing task's history."""
        task = self.get_task(task_id)
        if task:
            task.prompt = prompt # Update the task's prompt to the latest one
            task.conversation_history.append(Prompt.user(prompt))

    def get_active_task(self) -> Optional[Task]:
        """Get the currently active task."""
        if self.active_task_id:
            return self.get_task(self.active_task_id)
        # Fallback to last task if no active one is set
        return self.get_last_task()
```

--- END OF FILE src/model.py ---

--- START OF FILE src/mood_server.py ---

```py
import sys
import logging
from mcp.server.fastmcp import FastMCP
from mcp.server.elicitation import AcceptedElicitation, DeclinedElicitation, CancelledElicitation
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mood_server")

mcp = FastMCP("Mood Elicitation Server")

class MoodElicitationForm(BaseModel):
    mood: str = Field(
        description="In a few words, how are you feeling right now?",
        max_length=100
    )

@mcp.tool()
async def elicit_mood() -> str:
    """Elicits user mood through interactive form and returns structured response."""
    logger.info("Tool 'elicit_mood' called. Requesting free-form mood from user.")
    
    result = await mcp.get_context().elicit(
        "Please share how you're feeling.",
        schema=MoodElicitationForm
    )

    match result:
        case AcceptedElicitation(data=data):
            logger.info(f"User entered mood: '{data.mood}'")
            return f"The user described their mood as: '{data.mood}'"
        case DeclinedElicitation():
            logger.info("User declined the mood elicitation.")
            return "The user chose not to share their mood."
        case CancelledElicitation():
            logger.info("User cancelled the mood elicitation.")
            return "The user cancelled the request."

if __name__ == "__main__":
    mcp.run()
```

--- END OF FILE src/mood_server.py ---

--- START OF FILE src/secure_filesystem_server.py ---

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

--- END OF FILE src/secure_filesystem_server.py ---

--- START OF FILE src/textual_view.py ---

```py
# textual_view.py
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.containers import Vertical
from model import Task

from controller import Controller, ExitCommand, SwitchAgentCommand
from model import Model, Interaction
from agent_registry import DEFAULT_AGENT

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

    def __init__(self, agent_name: str = DEFAULT_AGENT):
        super().__init__()
        self.model = Model()
        self.controller = Controller(self.model, self)
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
        # This now renders tasks instead of a simple chat log
        if self._last_rendered_message_count != len(self.model.interactions):
            # Render only new interactions to be more efficient
            new_interactions = self.model.interactions[self._last_rendered_message_count:]
            for interaction in new_interactions:
                # Only render user prompts and final agent responses in the main chat view.
                # System messages like 'success' or 'error' are also rendered.
                if interaction.tag in ("user_prompt", "agent_response", "success", "error", "task_created", "task_status"):
                    if isinstance(interaction.content, Text):
                         self.log_widget.write(interaction.content)

            self._last_rendered_message_count = len(self.model.interactions)

    def update_header(self) -> None:
        # This could be enhanced to show number of running tasks, etc.
        active_task = self.model.get_active_task()
        active_task_id_display = f"Task: [bold cyan]{active_task.id}[/]" if active_task else "No Active Task"

        if self.model.is_thinking:
            self.sub_title = "🤔 Thinking..."
        else:
            self.sub_title = f"Agent: [bold]{self.agent_name}[/] | {active_task_id_display}"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value
        if not user_input:
            return
        
        self.input_widget.clear()
        
        async def process_input_with_exit_handling():
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                # Gracefully exit the application
                self.exit()
        
        self.run_worker(process_input_with_exit_handling(), exclusive=True)

```

--- END OF FILE src/textual_view.py ---

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
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_registry import get_agent, list_available_agents, AGENT_REGISTRY

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

# Remove outdated test for agent characteristics (instances/names)

if __name__ == "__main__":
    test_agent_registry()
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

--- START OF FILE tests/test_harness.py ---

```py
# tests/test_harness.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from main import Application
from model import Task
from mcp_agent.core.prompt import Prompt
from textual_view import AgentDashboardApp
from controller import Controller, ExitCommand, SwitchAgentCommand

# This is our mock agent that will be returned by the patched get_agent
mock_agent_instance = MagicMock()
mock_agent_instance.run = MagicMock()

# We need an async context manager for `async with agent.run()...`
mock_agent_context = AsyncMock()
mock_agent_instance.run.return_value = mock_agent_context

# The agent object itself inside the context
mock_agent_object = AsyncMock()
# The generate method is what we really care about
mock_agent_object.generate = AsyncMock(
    return_value=Prompt.assistant("This is a mocked response.")
)
# Add message_history to match controller expectations
mock_agent_object.message_history = [
    Prompt.user("Analyze this data"),
    Prompt.assistant("This is a mocked response.")
]
# Create a mock for the agent_app object that supports __getitem__
mock_agent_app = MagicMock()
mock_agent_app.__getitem__.return_value = mock_agent_object

# Make the context manager return our new mock agent_app
mock_agent_context.__aenter__.return_value = mock_agent_app


@pytest.mark.asyncio
@patch('controller.get_agent', return_value=mock_agent_instance)
async def test_end_to_end_task_execution(mock_get_agent):
    """
    Tests the full application lifecycle from user input to task completion
    using the Textual Pilot.
    """
    # 1. Run the app headlessly with the Pilot
    tui_app = AgentDashboardApp(agent_name="minimal")
    async with tui_app.run_test() as pilot:
        # 2. Simulate user typing a prompt and pressing enter
        prompt = "Analyze this data"
        await pilot.press(*prompt)
        await pilot.press("enter")

        # 3. Wait for the UI and any immediate workers to settle
        await pilot.pause()

        # 4. Assert that a task was created in the model
        assert len(tui_app.model.tasks) == 1
        task = tui_app.model.tasks[0]
        assert task.prompt == prompt
        assert task.status in ("running", "completed")

        # 5. Wait for all background workers to complete
        await pilot.wait_for_scheduled_animations()
        await tui_app.workers.wait_for_complete()

        # 6. Assert that the task is now completed
        completed_task = tui_app.model.get_task(task.id)
        assert completed_task is not None
        assert completed_task.status == "completed"
        assert completed_task.result == "This is a mocked response."

        # 7. Verify that the agent was called correctly
        mock_get_agent.assert_called_once_with("minimal")
        mock_agent_object.generate.assert_called_once()
```

--- END OF FILE tests/test_harness.py ---

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

