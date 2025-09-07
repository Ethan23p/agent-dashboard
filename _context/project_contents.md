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

--- START OF FILE agent-dashboard.log ---

```log
2025-07-23 14:48:00,304 - root - INFO - --- Application session started ---
2025-07-23 14:48:00,404 - model - INFO - Created and activated new session session_20250723_144800 for agent 'minimal'.
2025-07-23 14:48:00,490 - root - INFO - --- Application session ended ---
2025-07-23 14:52:49,472 - root - INFO - --- Application session started ---
2025-07-23 14:52:49,501 - model - INFO - Created and activated new session session_20250723_145249 for agent 'minimal'.
2025-07-23 14:53:15,023 - model - INFO - Session record saved successfully to _context/session_20250723_145249.json
2025-07-23 14:53:23,820 - google_genai._api_client - WARNING - Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
2025-07-23 14:53:24,966 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-07-23 14:53:27,334 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-07-23 14:53:29,330 - model - INFO - Session record saved successfully to _context/session_20250723_145249.json
2025-07-23 14:53:46,656 - root - INFO - --- Application session ended ---
2025-07-23 18:00:16,149 - root - INFO - --- Application session started ---
2025-07-23 18:00:16,178 - model - INFO - Created and activated new session session_20250723_180016 for agent 'minimal'.
2025-07-23 18:00:23,737 - root - INFO - --- Application session ended ---
2025-07-23 20:42:15,220 - root - INFO - --- Application session started ---
2025-07-23 20:42:15,255 - model - INFO - Created and activated new session session_20250723_204215 for agent 'minimal'.
2025-07-23 20:42:18,814 - root - INFO - --- Application session ended ---
2025-07-23 20:52:52,955 - root - INFO - --- Application session started ---
2025-07-23 20:52:52,992 - model - INFO - Created and activated new session session_20250723_205252 for agent 'minimal'.
2025-07-23 20:53:10,704 - root - INFO - --- Application session ended ---
2025-07-23 20:58:18,141 - root - INFO - --- Application session started ---
2025-07-23 20:58:18,185 - model - INFO - Created and activated new session session_20250723_205818 for agent 'minimal'.
2025-07-23 20:58:18,877 - root - INFO - --- Application session ended ---
2025-07-23 21:00:39,194 - root - INFO - --- Application session started ---
2025-07-23 21:00:39,231 - model - INFO - Created and activated new session session_20250723_210039 for agent 'minimal'.
2025-07-23 21:00:43,452 - model - INFO - Switched active session to session_20250723_210039.
2025-07-26 11:39:11,482 - root - INFO - --- Application session started ---
2025-07-26 11:39:12,414 - model - INFO - Created and activated new session session_20250726_113912 for agent 'minimal'.
2025-07-26 11:40:55,891 - model - INFO - Session record saved successfully to _context/session_20250726_113912.json
2025-07-26 11:41:06,310 - google_genai._api_client - WARNING - Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
2025-07-26 11:41:10,069 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-07-26 11:41:11,293 - google_genai.models - INFO - AFC is enabled with max remote calls: 10.
2025-07-26 11:41:12,359 - model - INFO - Session record saved successfully to _context/session_20250726_113912.json
2025-07-26 11:41:24,529 - commands - INFO - Exit command received.
2025-07-26 11:41:24,643 - root - INFO - --- Application session ended ---
2025-07-26 11:41:43,636 - root - INFO - --- Application session started ---
2025-07-26 11:41:43,675 - model - INFO - Created and activated new session session_20250726_114143 for agent 'minimal'.

```

--- END OF FILE agent-dashboard.log ---

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
    "pytest-mock>=3.12.0",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
pythonpath = ["src"]
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

```md
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

- **Context Management.** Following the philosophy of the Model Context Protocol, the controller assembles the conversational history and other data to form the precise context sent to the agent on each turn. This allows for more deliberate, developer-driven context strategies.

- **Asynchronous Core.** The application uses `asyncio` and a non-blocking prompt, which keeps the UI responsive. It's designed to support more complex operations, like parallel agent interactions, and could be adapted for a GUI dashboard later.

- **Stateful History.** While the terminal shows a clean chat log, a comprehensive history is maintained in the background. This history can be saved automatically or manually, providing a useful artifact for debugging or resuming sessions.

- **Resilient Operation.** LLM or MCP server errors are handled by the controller, which rolls back the conversational state to its last valid point. The application also shuts down cleanly to avoid resource errors.

- **Comprehensive Testing.** The application includes a complete testing suite with unit tests, integration tests, and retry mechanisms to ensure reliability and maintainability.

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

```

--- END OF FILE README.md ---

--- START OF FILE src/__init__.py ---

```py

```

--- END OF FILE src/__init__.py ---

--- START OF FILE src/agent_registry.py ---

```py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from typing import Optional

# This module acts as a centralized catalog for agent definitions, making it
# easy to manage and access different agent configurations.
#
# NOTE: All agents should use `use_history=False`, as the application's Model
# manages conversation history explicitly.

AGENT_DEFINITIONS = [
    {
        "name": "minimal-agent",
        "description": "A minimal *effective* agent with full capabilities.",
        "instruction": "You are a sophisticated assistant AI with many capabilities.",
        "servers": ["filesystem", "fetch", "sequential-thinking", "playwright", "desktop-commander", "gitmcp", "github"],
        "max_tokens": 4096,
    },
    {
        "name": "filesystem-agent",
        "description": "A simple agent with filesystem access.",
        "instruction": "You are a utility. You should straightforwardly do as the prompt instructs with absolutely no commentary or fluff.",
        "servers": ["filesystem"],
        "max_tokens": 2048,
    },
    {
        "name": "Spongebob-agent",
        "description": "A plain conversational agent which embodies Spongebob.",
        "instruction": "You are Spongebob Squarepants. You are optimistic and give advice, though you are clumsy all the while.",
        "servers": [],
        "max_tokens": 2048,
    },
    {
        "name": "AynRand-agent",
        "description": "A plain conversational agent which embodies Ayn Rand.",
        "instruction": "You are Ayn Rand. You are biased towards capitalism, unspoken patriarchy, and against communism.",
        "servers": [],
        "max_tokens": 2048,
    },
]

def _create_agent_from_definition(definition: dict) -> FastAgent:
    """Factory function to build a FastAgent from a definition dictionary."""
    agent_name = definition.get("name", "minimal")
    description = definition.get("description", "A fast-agent.")
    instruction = definition.get("instruction", "You are a helpful assistant.")
    servers = definition.get("servers", [])
    max_tokens = definition.get("max_tokens", 2048)

    agent_instance = FastAgent(description, config_path="src/fastagent.config.yaml")

    # The decorator requires a function to decorate, even if it's a placeholder.
    @agent_instance.agent(
        name=agent_name,
        instruction=instruction,
        servers=servers,
        request_params=RequestParams(maxTokens=max_tokens),
        use_history=False
    )
    async def placeholder_func(): pass
    
    return agent_instance

# The registry is built dynamically from the definitions list.
AGENT_REGISTRY = {
    definition["name"]: _create_agent_from_definition(definition)
    for definition in AGENT_DEFINITIONS if "name" in definition
}

DEFAULT_AGENT = AGENT_DEFINITIONS[0]["name"] if AGENT_DEFINITIONS else "minimal"

def get_agent(agent_name: Optional[str] = None):
    """
    Retrieves an agent from the registry by name.
        
    Raises:
        KeyError: If the agent name is not found.
    """
    agent_name = agent_name or DEFAULT_AGENT

    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """Returns a list of available agent names."""
    return list(AGENT_REGISTRY.keys())
```

--- END OF FILE src/agent_registry.py ---

--- START OF FILE src/commands.py ---

```py
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
```

--- END OF FILE src/commands.py ---

--- START OF FILE src/controller.py ---

```py
import asyncio
import logging
import random
from typing import TYPE_CHECKING

from agent_registry import get_agent
from commands import (ClearCommand, ExitCommand, ExitCommandImpl,
                        ListAgentsCommand, LoadCommand, SaveCommand,
                        SwitchAgentCommand, SwitchCommand)
from mcp_agent.core.prompt import Prompt
from model import Interaction, Session
from rich.text import Text

if TYPE_CHECKING:
    from textual_view import AgentDashboardApp
    from model import Model

logger = logging.getLogger(__name__)

class Controller:
    """
    Contains the application's business logic, responding to user input from
    the View and orchestrating interactions between the Model and the Agent.
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
        Handles user input, routing to either a command handler or the agent.
        """
        stripped_input = user_input.strip()
        if not stripped_input:
            return

        if stripped_input.lower().startswith('/'):
            await self._handle_command(stripped_input)
        else:
            await self._handle_prompt(stripped_input)

    async def _handle_command(self, command_str: str):
        """Parses and executes client-side commands."""
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
        """Handles a user prompt by creating an interaction and calling the agent."""
        user_interaction = Interaction(
            contents=[Prompt.user(user_prompt)],
            metadata={"user-facing": True, "type": "user_prompt"}
        )
        await self.model.add_interaction_to_active_session(user_interaction)

        if self.model.user_preferences.get("auto_save_enabled"):
            await self.model.save_active_session()

        # Run the agent turn in the background to keep the UI responsive.
        self.app.run_worker(self._execute_agent_turn, exclusive=False, group="agent_turns")

    async def _execute_agent_turn(self):
        """
        Executes a single agent turn with retry logic for resilience.
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
                agent_history = self.model.get_agent_history_for_active_session()

                async with agent_instance.run() as agent_app:
                    agent = agent_app[active_session.agent_name]
                    response_message = await agent.generate(agent_history)

                agent_interaction = Interaction(
                    contents=[response_message],
                    metadata={"user-facing": True, "type": "agent_response", "agent_name": active_session.agent_name}
                )
                await self.model.add_interaction_to_active_session(agent_interaction)

                if self.model.user_preferences.get("auto_save_enabled"):
                    await self.model.save_active_session()

                await self.model.set_thinking_status(False)
                return

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
                    return

```

--- END OF FILE src/controller.py ---

--- START OF FILE src/dashboard.tcss ---

```tcss
/* Configurable 5-color scheme using Textual theme variables */
/* Colors are defined in the Python code and applied via the theme system */
/* Primary: Main background color ($primary, $surface, $background) */
/* Secondary: Accent color for headers, buttons, etc. ($secondary, $panel) */
/* Text: Text color ($text) */
/* Highlight: Selection and highlight color ($accent, $success, 'warning') */
/* Border: Border and divider color ($error) */

/* General screen and layout styling */
Screen {
    background: $surface;
    color: $text;
    layout: vertical;
}

#app-grid {
    layout: horizontal;
    height: 100%;
}

/* Sidebar styling for the session list */
#session-list {
    width: 30;
    border-right: solid $border;
    overflow-y: auto;
    background: $surface;
}

#session-list ListItem {
    padding: 1;
    color: $text;
}

#session-list ListItem Label {
    color: $text;
    background: transparent;
}

#session-list ListItem:hover {
    background: $secondary;
}

#session-list ListItem:hover Label {
    color: $text;
    background: transparent;
}

#session-list ListItem.--highlight {
    background: $accent;
    color: $surface;
}

#session-list ListItem.--highlight Label {
    color: $surface;
    background: transparent;
}

/* Header and Footer styling */
Header {
    dock: top;
    background: $secondary;
    color: $text;
}

Footer {
    dock: bottom;
    background: $surface;
    color: $text;
}

/* Main content area container */
#main-content {
    width: 1fr;
    height: 100%;
    layout: vertical;
}

#chat-history {
    height: 1fr;
    padding: 0 1;
    background: $background;
    color: $text;
    border: round $border;
    margin: 1 2;
}

/* The composite input bar styling */
#input-bar {
    height: 3;
    padding: 0 1;
    background: $panel;
    border: round $border;
    margin: 0 1 1 1;
    dock: bottom;
}

#message-input {
    width: 1fr; /* The input field takes up all available flexible space */
    height: 100%;
    border: none; /* Remove the default border from the input */
    background: $panel;
    color: $text;
    padding: 0 1;
}

#send-button {
    width: 8;
    height: 100%;
    border: none; /* Remove the default border from the button */
    background: $accent;
    color: $surface;
    content-align: center middle;
}

#send-button:hover {
    background: $secondary;
    color: $text;
}
```

--- END OF FILE src/dashboard.tcss ---

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
    
    # Playwright server for browser automation
    playwright:
      command: "npx"
      args:
        - "@playwright/mcp@latest"

    # Desktop Commander for local system control
    desktop-commander:
      command: "npx"
      args:
        - "-y"
        - "@wonderwhy-er/desktop-commander"

    # GitMcp for git operations
    gitmcp:
      command: "npx"
      args:
        - "mcp-remote"
        - "https://gitmcp.io/{owner}/{repo}"

    # GitHub Copilot
    github:
      transport: "http"
      url: "https://api.githubcopilot.com/mcp/"

```

--- END OF FILE src/fastagent.config.yaml ---

--- START OF FILE src/main.py ---

```py
import asyncio
import sys
import argparse
import logging
from typing import Optional

from textual_view import AgentDashboardApp
from agent_registry import list_available_agents, DEFAULT_AGENT

def setup_logging():
    """Initializes logging to a file and stderr for critical errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="agent-dashboard.log",
        filemode="a"
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(console_handler)
    logging.info("--- Application session started ---")

def print_shutdown_message():
    print("\nClient session ended.")
    logging.info("--- Application session ended ---")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=DEFAULT_AGENT,
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

class Application:
    """Orchestrates the main application components."""
    def __init__(self, initial_agent_name: str):
        self.initial_agent_name = initial_agent_name

    async def run(self):
        tui_app = AgentDashboardApp(
            agent_name=self.initial_agent_name
        )
        await tui_app.run_async()

async def main():
    """Application entry point."""
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
import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from rich.text import Text

logger = logging.getLogger(__name__)

# Defines the flexible content for an Interaction, allowing it to hold agent
# messages, system notifications, or other data structures.
ContentsType = Union[List[PromptMessageMultipart], Text, str, Dict[str, Any]]

@dataclass
class Interaction:
    """Represents a single event or exchange, the atomic unit of a session."""
    contents: ContentsType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Interaction to a dictionary."""
        serialized_contents: Any
        if isinstance(self.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in self.contents):
            serialized_contents = [msg.model_dump(mode='json') for msg in self.contents]
            # Add a type hint to the metadata for robust deserialization.
            self.metadata['_content_type'] = 'prompt_message_multipart_list'
        elif isinstance(self.contents, Text):
            serialized_contents = self.contents.markup
            self.metadata['_content_type'] = 'rich_text'
        else:
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
        """Deserializes a dictionary back into an Interaction."""
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
    Represents a complete, continuous context of interactions, replacing the
    previous concepts of 'Task' and 'Conversation'.
    """
    id: str = field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: datetime = field(default_factory=datetime.now)
    agent_name: str = "minimal"
    status: str = "active"
    interactions: List[Interaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Session to a dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "agent_name": self.agent_name,
            "status": self.status,
            "interactions": [interaction.to_dict() for interaction in self.interactions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserializes a dictionary back into a Session."""
        return cls(
            id=data.get("id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            agent_name=data.get("agent_name", "minimal"),
            status=data.get("status", "completed"),
            interactions=[Interaction.from_dict(interaction_data) for interaction_data in data["interactions"]],
        )


async def save_session(session: Session, filepath: str) -> bool:
    """Saves a session to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Session record saved successfully to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session record to {filepath}: {e}", exc_info=True)
        return False

async def load_session(filepath: str) -> Optional[Session]:
    """Loads a session from a JSON file."""
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
    Manages the application's state, including all sessions, and notifies
    listeners of changes.
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

    async def _notify_listeners(self):
        """Notifies listeners of a state change."""
        # Create tasks to avoid blocking the model while listeners run.
        for listener in self._listeners:
            asyncio.create_task(listener())

    def register_listener(self, listener: Callable):
        """Register a callback for state changes."""
        self._listeners.append(listener)

    async def create_session(self, agent_name: Optional[str] = None) -> Session:
        """Creates a new session and sets it as active."""
        agent = agent_name or self.default_agent_name
        new_session = Session(agent_name=agent)
        self.sessions.append(new_session)
        self.active_session_id = new_session.id
        logger.info(f"Created and activated new session {new_session.id} for agent '{agent}'.")
        await self._notify_listeners()
        return new_session

    def get_session(self, session_id: str) -> Optional[Session]:
        return next((s for s in self.sessions if s.id == session_id), None)

    def get_active_session(self) -> Optional[Session]:
        if self.active_session_id:
            return self.get_session(self.active_session_id)
        return None

    async def set_active_session(self, session_id: str):
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
        logger.info("All sessions cleared; new default session created.")


    async def add_interaction_to_active_session(self, interaction: Interaction):
        """Adds an interaction to the active session."""
        active_session = self.get_active_session()
        if active_session:
            active_session.interactions.append(interaction)
            await self._notify_listeners()
        else:
            logger.warning("Attempted to add interaction, but no active session.")

    def get_agent_history_for_active_session(self) -> List[PromptMessageMultipart]:
        """Constructs the conversation history for the agent from the active session."""
        active_session = self.get_active_session()
        if not active_session:
            return []
        
        history: List[PromptMessageMultipart] = []
        for interaction in active_session.interactions:
            if isinstance(interaction.contents, list) and all(isinstance(item, PromptMessageMultipart) for item in interaction.contents):
                history.extend(interaction.contents)
        return history

    @property
    def display_history(self) -> List[Interaction]:
        """
        Returns a filtered list of interactions suitable for UI display.
        This simplifies the view's rendering logic.
        """
        active_session = self.get_active_session()
        if not active_session:
            return []
        
        return [
            interaction for interaction in active_session.interactions
            if interaction.metadata.get("user-facing", False)
        ]


    async def set_thinking_status(self, is_thinking: bool):
        """Sets the agent's thinking status and notifies listeners."""
        if self.is_thinking != is_thinking:
            self.is_thinking = is_thinking
            await self._notify_listeners()


    async def save_active_session(self):
        """Saves the active session to a file."""
        active_session = self.get_active_session()
        if not active_session:
            logger.warning("Attempted to save, but no active session.")
            return

        context_dir = self.user_preferences.get("context_dir", "_context")
        filename = f"{context_dir}/{active_session.id}.json"
        await save_session(active_session, filename)

```

--- END OF FILE src/model.py ---

--- START OF FILE src/textual_view.py ---

```py
import logging
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, RichLog

from agent_registry import DEFAULT_AGENT
from commands import ExitCommand, SwitchAgentCommand
from controller import Controller
from model import Interaction, Model

if TYPE_CHECKING:
    from model import Interaction

logger = logging.getLogger(__name__)

COLORS = {
    "primary": "#262624",
    "secondary": "#1F1E1D",
    "text": "#BFBDB8",
    "accent": "#D97059",
    "border": "#BFAF80",
    "emphasis": "#BFAF80",
}

CUSTOM_THEME = Theme(
    "creamy-dark",
    COLORS["primary"],
    secondary=COLORS["secondary"],
    foreground=COLORS["text"],
    accent=COLORS["accent"],
    surface=COLORS["secondary"],
    background=COLORS["primary"],
    panel=COLORS["secondary"],
    error=COLORS["accent"],
    warning=COLORS["accent"],
    success=COLORS["border"],
    variables={
        "border": COLORS["border"],
        "emphasis": COLORS["emphasis"],
    }
)

class AgentDashboardApp(App):
    """The Textual-based user interface for the agent dashboard."""

    CSS_PATH = "dashboard.tcss"
    TITLE = "Agent Dashboard"
    SUB_TITLE = "Interface for interacting with sophisticated AI agents."
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
        """Create child widgets for the app."""
        yield Header()
        with Container(id="app-grid"):
            yield ListView(id="session-list")
            with Container(id="main-content"):
                yield RichLog(id="chat-history", wrap=True, highlight=False, markup=True)
                with Horizontal(id="input-bar"):
                    yield Input(placeholder="How can I help you today?", id="message-input")
                    yield Button("↑", id="send-button")
        yield Footer()

    async def on_mount(self) -> None:
        """Initializes the application and creates the first session."""
        self.register_theme(CUSTOM_THEME)
        self.theme = "creamy-dark"

        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        # (1)! The model listener will automatically trigger the first UI update.
        await self.model.create_session()
        
        self.title = "Agent Dashboard"
        # (2)! Removed the direct call to self.on_model_update() to prevent the race condition.
        
        welcome_interaction = Interaction(
            Text.from_markup("🤖 Agent is ready. Say 'Hi' or type a command."),
            metadata={"user-facing": True, "type": "info"}
        )
        # This will trigger the second, necessary UI update.
        await self.model.add_interaction_to_active_session(welcome_interaction)

    async def on_model_update(self) -> None:
        """Schedules UI updates when the model's state changes."""
        self.call_later(self.update_session_list)
        self.call_later(self.render_log)
        self.call_later(self.update_header)

    def update_session_list(self) -> None:
        """Renders the list of sessions in the sidebar idempotently."""
        session_list_view = self.query_one("#session-list", ListView)
        
        # (3)! Add a guard to prevent re-rendering if the session list is already correct.
        current_ids = {item.id for item in session_list_view.children if item.id is not None}
        model_ids = {session.id for session in self.model.sessions}
        if current_ids == model_ids:
            # The view is already in sync, no need to do anything.
            return

        session_list_view.clear()
        
        active_session_id = self.model.active_session_id
        highlighted_index = 0

        for i, session in enumerate(self.model.sessions):
            session_label = f"{session.agent_name} ({session.id.split('_')[1]})"
            list_item = ListItem(Label(session_label), id=session.id)
            session_list_view.append(list_item)
            if session.id == active_session_id:
                highlighted_index = i
        
        if session_list_view.children:
            session_list_view.index = highlighted_index

    def render_log(self) -> None:
        """Renders the model's display_history to the chat log."""
        display_history = self.model.display_history
        
        if self._last_rendered_interaction_count == len(display_history) and self.log_widget.children:
            return

        self.log_widget.clear()
        for interaction in display_history:
            if isinstance(interaction.contents, Text):
                self.log_widget.write(interaction.contents)
            elif isinstance(interaction.contents, list):
                for msg in interaction.contents:
                    role = msg.role.capitalize()
                    color = COLORS['emphasis'] if msg.role == "user" else COLORS['accent']
                    self.log_widget.write(Text.from_markup(f"[bold {color}]{role}:[/] {msg.last_text()}"))
            else:
                self.log_widget.write(str(interaction.contents))

        self._last_rendered_interaction_count = len(display_history)

    def update_header(self) -> None:
        """Updates the header with the current agent and thinking status."""
        active_session = self.model.get_active_session()
        agent_name = active_session.agent_name if active_session else "N/A"
        
        if self.model.is_thinking:
            self.sub_title = f"Agent: [bold]{agent_name}[/] 🤔 Thinking..."
        else:
            self.sub_title = f"Active Agent: [bold]{agent_name}[/]"

    @on(ListView.Selected, "#session-list")
    def on_session_selected(self, event: ListView.Selected) -> None:
        """Handle the selection of a new session in the sidebar."""
        if event.item.id:
            self.run_worker(self.model.set_active_session(event.item.id))

    def action_send_message(self) -> None:
        """Called when the user sends a message."""
        user_input = self.input_widget.value
        if not user_input:
            return
        
        self.input_widget.clear()
        
        async def process_input_with_exception_handling():
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                self.exit()
            except SwitchAgentCommand as e:
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

    @on(Button.Pressed, "#send-button")
    def on_button_pressed(self) -> None:
        """Handle send button clicks."""
        self.action_send_message()

    @on(Input.Submitted, "#message-input")
    def on_input_submitted(self) -> None:
        """Handle 'Enter' key press in the input field."""
        self.action_send_message()
```

--- END OF FILE src/textual_view.py ---

--- START OF FILE tests/conftest.py ---

```py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from model import Model
from textual_view import AgentDashboardApp

@pytest.fixture
def mock_model():
    """Provides a mock Model object with a spec for type hinting."""
    # The 'spec' argument makes the mock type-aware for Pylance.
    return MagicMock(spec=Model)

@pytest.fixture
def mock_app():
    """Provides a mock Textual App object with a spec and mocked run_worker."""
    mock = MagicMock(spec=AgentDashboardApp)
    # We still need to mock the implementation of run_worker.
    mock.run_worker = AsyncMock(side_effect=lambda coro, **kwargs: asyncio.create_task(coro))
    return mock
```

--- END OF FILE tests/conftest.py ---

--- START OF FILE tests/integration/test_controller.py ---

```py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from src.controller import Controller
from src.primitives import Interaction
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import TextContent

@pytest.fixture
def controller(mock_model, mock_app):
    """Provides a Controller instance with mocked dependencies."""
    return Controller(mock_model, mock_app)

@pytest.mark.asyncio
async def test_handle_prompt_and_agent_turn(controller: Controller, mocker):
    """
    Tests the main interaction loop: user prompt -> agent call -> agent response.
    This test validates the core logic without calling a real LLM.
    """
    # 1. Setup: Mock the agent's response
    mock_agent_response = PromptMessageMultipart(
        role="assistant",
        content=[TextContent(type="text", text="This is a mock agent response.")]
    )
    
    # Mock the agent instance and its generate method
    mock_agent_instance = MagicMock()
    mock_agent_instance.generate = AsyncMock(return_value=mock_agent_response)
    
    # Mock the agent application context manager
    mock_agent_app = MagicMock()
    mock_agent_app.__getitem__.return_value = mock_agent_instance
    
    mock_agent_context = AsyncMock()
    mock_agent_context.__aenter__.return_value = mock_agent_app
    
    # Mock the get_agent function to return our controlled agent
    mocker.patch(
        'src.controller.get_agent', 
        return_value=MagicMock(run=MagicMock(return_value=mock_agent_context))
    )

    # Mock the active session to control its state
    mock_session = MagicMock()
    mock_session.agent_name = "minimal"
    controller.model.get_active_session.return_value = mock_session

    # 2. Action: Process a user prompt
    user_prompt = "Hello, agent!"
    await controller.process_user_input(user_prompt)
    
    await asyncio.sleep(0.01) # Wait for the background worker task to run

    # 3. Assertions: Verify the controller's behavior
    controller.model.set_thinking_status.assert_any_call(True)
    controller.model.set_thinking_status.assert_any_call(False)

    assert controller.model.add_interaction_to_active_session.call_count == 2

    # Verify the user prompt interaction was added correctly
    user_interaction_call = controller.model.add_interaction_to_active_session.call_args_list[0]
    user_interaction_arg: Interaction = user_interaction_call.args[0]
    assert isinstance(user_interaction_arg, Interaction)
    assert user_interaction_arg.metadata["type"] == "user_prompt"
    
    assert isinstance(user_interaction_arg.contents, list)
    assert isinstance(user_interaction_arg.contents[0], PromptMessageMultipart)
    assert user_interaction_arg.contents[0].last_text() == user_prompt

    # Verify the agent response interaction was added correctly
    agent_interaction_call = controller.model.add_interaction_to_active_session.call_args_list[1]
    agent_interaction_arg: Interaction = agent_interaction_call.args[0]
    assert isinstance(agent_interaction_arg, Interaction)
    assert agent_interaction_arg.metadata["type"] == "agent_response"
    
    assert isinstance(agent_interaction_arg.contents, list)
    assert agent_interaction_arg.contents[0] == mock_agent_response
```

--- END OF FILE tests/integration/test_controller.py ---

--- START OF FILE tests/unit/test_model.py ---

```py
import pytest
from src.model import Model
from src.primitives import Interaction
from src.agent_registry import DEFAULT_AGENT

@pytest.fixture
def model():
    """Provides a fresh Model instance for each test."""
    return Model(default_agent_name=DEFAULT_AGENT)

@pytest.mark.asyncio
async def test_model_initialization(model: Model):
    """Test that the model initializes correctly without a session."""
    assert not model.sessions
    assert model.active_session_id is None

@pytest.mark.asyncio
async def test_create_session(model: Model):
    """Test session creation and activation."""
    assert len(model.sessions) == 0
    
    session = await model.create_session()
    
    assert len(model.sessions) == 1
    assert model.active_session_id == session.id
    assert model.get_active_session() is session
    assert session.agent_name == DEFAULT_AGENT

@pytest.mark.asyncio
async def test_add_interaction(model: Model):
    """Test adding an interaction to the active session."""
    session = await model.create_session()
    
    interaction = Interaction("Test content")
    await model.add_interaction_to_active_session(interaction)
    
    active_session = model.get_active_session()
    assert active_session
    assert len(active_session.interactions) == 1
    assert active_session.interactions[0] == interaction

@pytest.mark.asyncio
async def test_display_history_filtering(model: Model):
    """Test that display_history only returns user-facing interactions."""
    await model.create_session()
    
    await model.add_interaction_to_active_session(Interaction("User message", metadata={"user-facing": True}))
    await model.add_interaction_to_active_session(Interaction("System log", metadata={"user-facing": False}))
    await model.add_interaction_to_active_session(Interaction("Agent response", metadata={"user-facing": True}))
    
    display_history = model.display_history
    assert len(display_history) == 2
    assert display_history[0].contents == "User message"
    assert display_history[1].contents == "Agent response"
```

--- END OF FILE tests/unit/test_model.py ---

