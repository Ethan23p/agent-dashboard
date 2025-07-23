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

--- START OF FILE paperwork/.python-version ---

```
3.13

```

--- END OF FILE paperwork/.python-version ---

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

```

--- END OF FILE README.md ---

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

def _create_agent_from_definition(definition: dict) -> FastAgent:
    """Factory function to build a FastAgent instance from a dictionary."""
    
    # Use .get() to provide defaults for optional keys
    agent_name = definition.get("name", "minimal")
    description = definition.get("description", "A fast-agent.")
    instruction = definition.get("instruction", "You are a helpful assistant.")
    servers = definition.get("servers", [])
    max_tokens = definition.get("max_tokens", 2048)

    agent_instance = FastAgent(description, config_path="src/fastagent.config.yaml")

    # The decorator needs a function to decorate, even a placeholder
    @agent_instance.agent(
        name=agent_name,
        instruction=instruction,
        servers=servers,
        request_params=RequestParams(maxTokens=max_tokens),
        use_history=False
    )
    async def placeholder_func(): pass
    
    return agent_instance

# The registry is now BUILT dynamically from the definitions list.
AGENT_REGISTRY = {}

# Default agent (first one in the list)
DEFAULT_AGENT = AGENT_DEFINITIONS[0]["name"] if AGENT_DEFINITIONS else "minimal"

# Populate the registry
for definition in AGENT_DEFINITIONS:
    agent_name = definition.get("name")
    if agent_name:
        AGENT_REGISTRY[agent_name] = _create_agent_from_definition(definition)

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
```

--- END OF FILE src/commands.py ---

--- START OF FILE src/controller.py ---

```py
# /src/controller.py
import asyncio
import logging
import random
from typing import TYPE_CHECKING

from agent_registry import get_agent
from commands import (ClearCommand, ExitCommand, ExitCommandImpl,
                      ListAgentsCommand, LoadCommand, SaveCommand,
                      SwitchAgentCommand, SwitchCommand)
from mcp_agent.core.prompt import Prompt
from model import Interaction
from primitives import Session
from rich.text import Text

if TYPE_CHECKING:
    from textual_view import AgentDashboardApp
    from model import Model

logger = logging.getLogger(__name__)

class Controller:
    """
    The Controller contains the application's business logic. It responds
    to user input from the View and orchestrates interactions between the
    Model and the Agent.
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
            await self._handle_prompt(stripped_input)

    async def _handle_command(self, command_str: str):
        """Parse and execute client-side commands."""
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
        """
        Handles a user prompt by creating an Interaction and starting a
        background worker to get the agent's response.
        """
        # 1. Create and store the user's interaction
        user_interaction = Interaction(
            contents=[Prompt.user(user_prompt)],
            metadata={"user-facing": True, "type": "user_prompt"}
        )
        await self.model.add_interaction_to_active_session(user_interaction)

        # 2. Save the session record *after* the user's turn
        if self.model.user_preferences.get("auto_save_enabled"):
            await self.model.save_active_session()

        # 3. Start a background worker to execute the agent turn
        self.app.run_worker(self._execute_agent_turn(), exclusive=False, group="agent_turns")

    async def _execute_agent_turn(self):
        """
        The background worker that executes a single agent turn.
        This includes the full retry logic and state management.
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
                # Get the latest history from the model for the agent
                agent_history = self.model.get_agent_history_for_active_session()

                async with agent_instance.run() as agent_app:
                    agent = agent_app[active_session.agent_name]
                    response_message = await agent.generate(agent_history)

                # Create an interaction for the agent's full response, including tool calls
                agent_interaction = Interaction(
                    contents=[response_message],
                    metadata={"user-facing": True, "type": "agent_response", "agent_name": active_session.agent_name}
                )
                await self.model.add_interaction_to_active_session(agent_interaction)

                # Save the session record *after* the agent's turn
                if self.model.user_preferences.get("auto_save_enabled"):
                    await self.model.save_active_session()

                await self.model.set_thinking_status(False)
                return  # Success

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
                    return # Failure
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
import logging
from typing import Optional

from textual_view import AgentDashboardApp
from agent_registry import list_available_agents, DEFAULT_AGENT

def setup_logging():
    """Configures file-based logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="agent-dashboard.log",
        filemode="a" # Append to the log file
    )
    # Add a handler to also print critical errors to the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(console_handler)
    logging.info("--- Application session started ---")

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")
    logging.info("--- Application session ended ---")

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
    setup_logging()
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
# /src/model.py - note to future Ethan... You've had to leave this mid refactor
import asyncio
import json
import logging
import os
from typing import Callable, List, Optional

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from primitives import Interaction, Session

# Set up a logger for this module for developer diagnostics (the "session log")
logger = logging.getLogger(__name__)

async def save_session(session: Session, filepath: str) -> bool:
    """
    Saves a session record to a JSON file.
    This uses the `to_dict` method from the Session primitive for robust serialization.
    """
    try:
        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Session record saved successfully to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session record to {filepath}: {e}", exc_info=True)
        return False

async def load_session(filepath: str) -> Optional[Session]:
    """
    Loads a session record from a JSON file.
    This uses the `from_dict` method from the Session primitive for robust deserialization.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        session = Session.from_dict(session_data)
        logger.info(f"Session record loaded successfully from {filepath}")
        return session
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to load session record from {filepath}: {e}", exc_info=True)
        return None

class Model:
    """
    The Model represents the single source of truth for the application's state.
    It manages all sessions and notifies listeners when its state changes.
    """
    def __init__(self, default_agent_name: str):
        self.sessions: List[Session] = []
        self.active_session_id: Optional[str] = None
        self.is_thinking: bool = False
        self.default_agent_name: str = default_agent_name

        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        self._listeners: List[Callable] = []

    # --- Listener Pattern for UI Updates ---
    async def _notify_listeners(self):
        """Notify all registered listeners of a state change."""
        for listener in self._listeners:
            # Use asyncio.create_task to avoid blocking the model's operations
            asyncio.create_task(listener())

    def register_listener(self, listener: Callable):
        """Register a callback to be notified of state changes."""
        self._listeners.append(listener)

    # --- Session Management ---
    async def create_session(self, agent_name: Optional[str] = None) -> Session:
        """Creates a new session, sets it as active, and returns it."""
        agent = agent_name or self.default_agent_name
        new_session = Session(agent_name=agent)
        self.sessions.append(new_session)
        self.active_session_id = new_session.id
        logger.info(f"Created and activated new session {new_session.id} for agent '{agent}'.")
        await self._notify_listeners()
        return new_session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Finds a session by its ID."""
        return next((s for s in self.sessions if s.id == session_id), None)

    def get_active_session(self) -> Optional[Session]:
        """Returns the currently active session."""
        if self.active_session_id:
            return self.get_session(self.active_session_id)
        return None

    async def set_active_session(self, session_id: str):
        """Sets the active session by ID."""
        if self.get_session(session_id):
            self.active_session_id = session_id
            logger.info(f"Switched active session to {session_id}.")
            await self._notify_listeners()
        else:
            logger.warning(f"Attempted to switch to non-existent session {session_id}.")

    async def clear_sessions(self):
        """Clears all sessions and creates a new default one."""
        self.sessions = []
        self.active_session_id = None
        await self.create_session(self.default_agent_name)
        logger.info("All sessions cleared. A new default session has been created.")
        # create_session already notifies listeners

    # --- Interaction Management ---
    async def add_interaction_to_active_session(self, interaction: Interaction):
        """Adds an interaction to the active session's record."""
        active_session = self.get_active_session()
        if active_session:
            active_session.interactions.append(interaction)
            await self._notify_listeners()
        else:
            logger.warning("Attempted to add interaction, but no active session.")

    # --- History Preparation for Agent and View ---
    def get_agent_history_for_active_session(self) -> List[PromptMessageMultipart]:
        """
        Constructs the conversation history for the agent from the active session record.
        This includes all user and assistant turns.
        """
        active_session = self.get_active_session()
        if not active_session:
            return []
        
        history: List[PromptMessageMultipart] = []
        for interaction in active_session.interactions:
            # Only include interactions that are lists of PromptMessageMultipart
            if isinstance(interaction.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in interaction.contents):
                history.extend(interaction.contents)
        return history

    @property
    def display_history(self) -> List[Interaction]:
        """
        A computed property that returns a filtered list of interactions
        for the UI to display. This keeps the view simple.
        """
        active_session = self.get_active_session()
        if not active_session:
            return []
        
        return [
            interaction for interaction in active_session.interactions
            if interaction.metadata.get("user-facing", False)
        ]

    # --- State Management ---
    async def set_thinking_status(self, is_thinking: bool):
        """Sets the agent's thinking status and notifies listeners."""
        if self.is_thinking != is_thinking:
            self.is_thinking = is_thinking
            await self._notify_listeners()

    # --- Persistence ---
    async def save_active_session(self):
        """Saves the current active session to a file."""
        active_session = self.get_active_session()
        if not active_session:
            logger.warning("Attempted to save, but no active session.")
            return

        context_dir = self.user_preferences.get("context_dir", "_context")
        filename = f"{context_dir}/{active_session.id}.json"
        await save_session(active_session, filename)
```

--- END OF FILE src/model.py ---

--- START OF FILE src/primitives.py ---

```py
# src/primitives.py
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Union

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text

# Define a flexible type for the contents of an interaction.
# This allows an Interaction to hold agent messages, system notifications, or other data.
ContentsType = Union[List[PromptMessageMultipart], Text, str, Dict[str, Any]]

@dataclass
class Interaction:
    """
    Represents a single event or exchange within a session. This is the atomic
    unit of the session record.
    """
    contents: ContentsType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Interaction object to a JSON-friendly dictionary."""
        serialized_contents: Any
        if isinstance(self.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in self.contents):
            # Handle list of PromptMessageMultipart for agent turns
            serialized_contents = [msg.model_dump(mode='json') for msg in self.contents]
            # Add a type hint to the metadata for robust deserialization
            self.metadata['_content_type'] = 'prompt_message_multipart_list'
        elif isinstance(self.contents, Text):
            # Serialize Rich Text to a string with markup
            serialized_contents = self.contents.markup
            self.metadata['_content_type'] = 'rich_text'
        else:
            # For str, dict, etc., which are already JSON-serializable
            serialized_contents = self.contents
            self.metadata['_content_type'] = 'primitive'

        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "contents": serialized_contents,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Deserializes a dictionary back into an Interaction object."""
        metadata = data.get("metadata", {})
        content_type = metadata.get('_content_type')
        
        deserialized_contents: ContentsType
        if content_type == 'prompt_message_multipart_list':
            deserialized_contents = [PromptMessageMultipart(**msg_data) for msg_data in data["contents"]]
        elif content_type == 'rich_text':
            deserialized_contents = Text.from_markup(data["contents"])
        else: # 'primitive' or fallback
            deserialized_contents = data["contents"]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            contents=deserialized_contents,
            metadata=metadata,
        )

@dataclass
class Session:
    """
    Represents a complete, continuous context of interactions. This replaces
    the previous concepts of 'Task' and 'Conversation'.
    """
    id: str = field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: datetime = field(default_factory=datetime.now)
    agent_name: str = "minimal"
    status: str = "active"  # e.g., active, archived, completed
    interactions: List[Interaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Session object to a JSON-friendly dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "agent_name": self.agent_name,
            "status": self.status,
            "interactions": [interaction.to_dict() for interaction in self.interactions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserializes a dictionary back into a Session object."""
        return cls(
            id=data.get("id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_name=data.get("agent_name", "minimal"),
            status=data.get("status", "completed"),
            interactions=[Interaction.from_dict(interaction_data) for interaction_data in data["interactions"]],
        )
```

--- END OF FILE src/primitives.py ---

--- START OF FILE src/textual_view.py ---

```py
# textual_view.py
import logging
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from agent_registry import DEFAULT_AGENT
from commands import ExitCommand, SwitchAgentCommand
from controller import Controller
from model import Model

if TYPE_CHECKING:
    from model import Interaction

logger = logging.getLogger(__name__)

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
        height: 1fr;
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
        self.model = Model(default_agent_name=agent_name)
        self.controller = Controller(self.model, self)
        self._last_rendered_interaction_count = 0
        self.model.register_listener(self.on_model_update)

    def compose(self) -> ComposeResult:
        """Create the core UI widgets."""
        yield Header()
        yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=True)
        yield Input(placeholder="Enter your prompt or type /help...")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app when first mounted."""
        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        # Create the initial session
        await self.model.create_session()
        
        self.title = "Agent Dashboard"
        await self.on_model_update() # Initial render
        
        welcome_interaction = Interaction(
            Text.from_markup("🤖 Agent is ready. Say 'Hi' or type a command."),
            metadata={"user-facing": True, "type": "info"}
        )
        await self.model.add_interaction_to_active_session(welcome_interaction)

    async def on_model_update(self) -> None:
        """Handle model state changes by updating the UI safely on the main thread."""
        self.call_later(self.render_log)
        self.call_later(self.update_header)

    def render_log(self) -> None:
        """
        Renders the display_history from the model. This is a pure rendering
        method with no filtering logic.
        """
        display_history = self.model.display_history
        if self._last_rendered_interaction_count != len(display_history):
            self.log_widget.clear()
            for interaction in display_history:
                # The view now only needs to know how to render the content,
                # not what the content means.
                if isinstance(interaction.contents, Text):
                    self.log_widget.write(interaction.contents)
                elif isinstance(interaction.contents, list): # Assumes PromptMessageMultipart list
                    for msg in interaction.contents:
                        # A simple rendering strategy for now. This could be enhanced.
                        role = msg.role.capitalize()
                        color = "blue" if msg.role == "user" else "magenta"
                        self.log_widget.write(Text.from_markup(f"[bold {color}]{role}:[/] {msg.last_text()}"))
                else:
                    self.log_widget.write(str(interaction.contents))

            self._last_rendered_interaction_count = len(display_history)

    def update_header(self) -> None:
        """Updates the header based on the model's state."""
        active_session = self.model.get_active_session()
        agent_name = active_session.agent_name if active_session else "N/A"
        
        if self.model.is_thinking:
            self.sub_title = f"Agent: [bold]{agent_name}[/] 🤔 Thinking..."
        else:
            self.sub_title = f"Active Agent: [bold]{agent_name}[/]"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value
        if not user_input:
            return
        
        self.input_widget.clear()
        
        async def process_input_with_exception_handling():
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                self.exit()
            except SwitchAgentCommand as e:
                # Handle agent switching by creating a new session
                await self.model.create_session(agent_name=e.agent_name)
                switch_interaction = Interaction(
                    Text.from_markup(f"[bold green]Success:[/bold green] Switched to agent '{e.agent_name}'."),
                    metadata={"user-facing": True, "type": "success"}
                )
                await self.model.add_interaction_to_active_session(switch_interaction)
            except Exception as e:
                logger.critical("Unhandled exception in input processing", exc_info=True)
                error_text = Text.from_markup(f"[bold red]Critical Error:[/bold red] An unexpected error occurred: {e}")
                error_interaction = Interaction(error_text, metadata={"user-facing": True, "type": "error"})
                await self.model.add_interaction_to_active_session(error_interaction)

        self.run_worker(process_input_with_exception_handling(), exclusive=True)
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
# Create a mock for the agent_app object that supports dictionary access
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

