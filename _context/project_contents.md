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

--- START OF FILE .python-version ---

```
3.13

```

--- END OF FILE .python-version ---

--- START OF FILE agent_definitions.py ---

```py
# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.

# Simple single agent for basic operations
minimal_agent = FastAgent("Minimal Agent")

@minimal_agent.agent(
    name="agent",
    instruction="""
    You are a helpful assistant that can perform various operations.
    You can read files, write files, and list directory contents.
    Always be helpful and provide clear responses to user requests.
    """,
    servers=["filesystem"],
    request_params=RequestParams(maxTokens=2048),
    use_history=False  # <-- THIS IS THE KEY CHANGE
)

async def agent():
    """ This function is a placeholder for the decorator. """
    pass

```

--- END OF FILE agent_definitions.py ---

--- START OF FILE controller.py ---

```py
# controller.py
from typing import TYPE_CHECKING

from mcp_agent.core.prompt import Prompt
from model import AppState, Model

if TYPE_CHECKING:
    from mcp_agent.core.agent_app import AgentApp

class ExitCommand(Exception):
    """Custom exception to signal a graceful exit from the main loop."""
    pass

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

    async def _handle_command(self, command_str: str):
        """Parses and executes client-side commands like /save or /exit."""
        parts = command_str.lower().split()
        command_name = parts[0][1:]  # remove the '/'
        args = parts[1:]

        command_map = {
            'exit': self._cmd_exit,
            'quit': self._cmd_exit,
            'save': self._cmd_save,
            'load': self._cmd_load,
            'clear': self._cmd_clear,
        }

        handler = command_map.get(command_name)
        if handler:
            await handler(args)
        else:
            await self.model.set_state(AppState.ERROR, error_message=f"Unknown command: /{command_name}")

    async def _cmd_exit(self, args):
        raise ExitCommand()

    async def _cmd_save(self, args):
        filename = args[0] if args else None
        # If filename provided, ensure it's in the context directory
        if filename and not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = self.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        success = await self.model.save_history_to_file(filename)
        if success:
            await self.model.set_state(AppState.IDLE, success_message="History saved successfully.")
        else:
            await self.model.set_state(AppState.ERROR, error_message="Failed to save history.")

    async def _cmd_load(self, args):
        if not args:
            await self.model.set_state(AppState.ERROR, error_message="Please provide a filename: /load <filename>")
            return
        filename = args[0]
        # If filename doesn't start with path separator, assume it's in context directory
        if not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = self.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        success = await self.model.load_history_from_file(filename)
        if success:
            await self.model.set_state(AppState.IDLE, success_message="History loaded successfully.")
        else:
            await self.model.set_state(AppState.ERROR, error_message="Failed to load history.")

    async def _cmd_clear(self, args):
        await self.model.clear_history()
        await self.model.set_state(AppState.IDLE, success_message="Conversation history cleared.")


    async def _handle_agent_prompt(self, user_prompt: str):
        """
        Manages the full lifecycle of a conversational turn with the agent.
        """
        await self.model.set_state(AppState.AGENT_IS_THINKING)
        user_message = Prompt.user(user_prompt)
        await self.model.add_message(user_message)

        try:
            response_message = await self.agent.generate(
                self.model.conversation_history
            )
            await self.model.add_message(response_message)
        except Exception as e:
            await self.model.set_state(AppState.ERROR, error_message=f"Agent Error: {e}")
            await self.model.pop_last_message()
        finally:
            if self.model.application_state != AppState.ERROR:
                await self.model.set_state(AppState.IDLE)
            if self.model.user_preferences.get("auto_save_enabled"):
                await self.model.save_history_to_file()
```

--- END OF FILE controller.py ---

--- START OF FILE fastagent.config.yaml ---

```yaml
# fastagent.config.yaml

# --- Model Configuration ---
# Set the default model for all agents.
# You can override this per-agent in the decorator or with the --model CLI flag.
# Format: <provider>.<model_name> (e.g., openai.gpt-4o, anthropic.claude-3-5-sonnet-latest)
# Aliases like 'sonnet' or 'haiku' are also supported.
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
    # Fetch server for web scraping and data retrieval.
    fetch:
      # Use the Python runner 'uvx'
      command: "uvx"
      # Use the Python package name 'mcp-server-fetch'
      args: ["mcp-server-fetch"]
    
    # Filesystem server for reading/writing local files.
    filesystem:
      # The command to run the server. 'npx' is a good cross-platform choice.
      command: "npx"
      # Arguments for the command.
      args:
        - "-y" # Automatically say yes to npx prompts
        - "@modelcontextprotocol/server-filesystem"
        # IMPORTANT: Replace this with the ABSOLUTE path to the directory
        # you want the agent to have access to. Using absolute paths is crucial
        # for reliability.
        - "G:/My Drive/AI Resources/Open collection"

    # New read-only server for the planner
    readonly_fs:
      command: "uv" # or "python"
      args: ["run", "readonly_filesystem_server.py", "G:/My Drive/AI Resources/Open collection"] # Allow access to current dir

    # Memory server for persistent knowledge graph memory.
    memory:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-memory"

    # Sequential Thinking server for dynamic and reflective problem-solving.
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

from model import Model
from view import View
from controller import Controller, ExitCommand
from agent_definitions import minimal_agent

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

async def main():
    """
    The main entry point for the application.
    """
    # Run the minimal agent
    async with minimal_agent.run() as agent_app:
        print("Starting minimal agent...")
        
        # Initialize MVC components
        model = Model()
        controller = Controller(model, agent_app)
        view = View(model, controller)
        
        # Run the main loop until exit
        await view.run_main_loop()

    # This delay happens AFTER minimal_agent.run() has closed, giving background
    # tasks time to finalize their shutdown before the script terminates.
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # We no longer catch SystemExit here, but keep it for robustness.
        pass
    finally:
        # The final message is printed after everything has shut down.
        print_shutdown_message()
```

--- END OF FILE main.py ---

--- START OF FILE model.py ---

```py
# model.py
import asyncio
import json
import os
from datetime import datetime
from enum import Enum, auto
from typing import Callable, List, Optional

# Core types from the fast-agent framework.
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

class AppState(Enum):
    """Defines the possible states of the client application."""
    IDLE = auto()
    AGENT_IS_THINKING = auto()
    WAITING_FOR_USER_INPUT = auto()
    ERROR = auto()

class Model:
    """
    The Model represents the single source of truth for the application's state.
    It holds all data and notifies listeners when its state changes. It contains
    no business logic and is entirely passive.
    """
    def __init__(self):
        # --- State Data ---
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history: List[PromptMessageMultipart] = []
        self.application_state: AppState = AppState.IDLE
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        
        # Corrected initialization sequence:
        # 1. Initialize the dictionary with static keys first.
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        # 2. Now that self.user_preferences exists, we can safely use its
        #    values to construct and add the dynamic key.
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"

        # --- Notification System ---
        self._listeners: List[Callable] = []

    def _get_context_dir(self) -> str:
        """Helper to access the context directory from preferences."""
        return self.user_preferences.get("context_dir", "_context")

    async def _notify_listeners(self):
        """Asynchronously notify all registered listeners of a state change."""
        for listener in self._listeners:
            await listener()

    def register_listener(self, listener: Callable):
        """
        Allows other components (like the View) to register a callback
        to be notified of state changes.
        """
        self._listeners.append(listener)

    # --- Methods to Mutate State (Instructed by the Controller) ---

    async def add_message(self, message: PromptMessageMultipart):
        """Appends a new message to the conversation history."""
        self.conversation_history.append(message)
        await self._notify_listeners()

    async def pop_last_message(self) -> Optional[PromptMessageMultipart]:
        """
        Removes and returns the last message from the history.
        Crucial for rolling back state on agent failure.
        """
        if not self.conversation_history:
            return None
        last_message = self.conversation_history.pop()
        await self._notify_listeners()
        return last_message

    async def clear_history(self):
        """Clears the entire conversation history."""
        self.conversation_history = []
        await self._notify_listeners()

    async def set_state(self, new_state: AppState, error_message: Optional[str] = None, success_message: Optional[str] = None):
        """Updates the application's current state and notifies listeners."""
        self.application_state = new_state
        if new_state == AppState.ERROR:
            self.last_error_message = error_message
            self.last_success_message = None
        else:
            self.last_error_message = None # Clear error on non-error states.
            self.last_success_message = success_message
        await self._notify_listeners()

    async def load_history_from_file(self, filepath: str) -> bool:
        """
        Loads conversation history from a JSON file, replacing the current history.
        Returns True on success, False on failure.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)
            # Re-create the rich PromptMessageMultipart objects from the raw dicts.
            self.conversation_history = [
                PromptMessageMultipart(**data) for data in raw_history
            ]
            await self._notify_listeners()
            return True
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            # We don't change state on failure, just report it.
            await self.set_state(AppState.ERROR, f"Failed to load history: {e}")
            return False

    # --- Methods for Actions (Instructed by the Controller) ---

    async def save_history_to_file(self, filepath: Optional[str] = None) -> bool:
        """
        Saves the current conversation history to a specified JSON file.
        This method does not mutate the model's state.
        Returns True on success, False on failure.
        """
        target_filepath = filepath or self.user_preferences["auto_save_filename"]
        context_dir = self._get_context_dir()
        os.makedirs(context_dir, exist_ok=True)

        try:
            serializable_history = [
                message.model_dump(mode='json') for message in self.conversation_history
            ]
            with open(target_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            # If saving fails, we set an error state to inform the user.
            await self.set_state(AppState.ERROR, f"Could not write to file {target_filepath}")
            return False
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
    "fast-agent-mcp",
]

```

--- END OF FILE pyproject.toml ---

--- START OF FILE README.md ---

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

## Project Journey

This client evolved through several stages:

1.  Began with simple `fast-agent` scripts run from the command line.
2.  Integrated a few powerful MCP servers (`filesystem`, `memory`, `fetch`), which revealed the potential of the protocol.
3.  Shifted focus from thinking of `fast-agent` as a script runner to using it as a library within a client/server model.
4.  Adopted the MVC pattern to cleanly separate concerns.
5.  The result is this application—a stable tool for further agent development.

```

--- END OF FILE README.md ---

--- START OF FILE readonly_filesystem_server.py ---

```py
# readonly_filesystem_server.py

import os
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP
import typer

# Initialize the FastMCP server
mcp = FastMCP("readonly_fs")

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


# The `run` function provided by FastMCP will automatically handle CLI arguments.
# Any arguments defined here (like `allowed_dirs`) that are not tools themselves
# will be passed to all tool functions that require them.
@mcp.run(transport="stdio")
def main(allowed_dirs: List[Path] = typer.Argument(..., help="List of directories to allow read access to.")):
    """
    A read-only filesystem MCP server.
    This server will run until the client disconnects.
    """
    # The typer decorator and mcp.run handle the server lifecycle.
    pass
```

--- END OF FILE readonly_filesystem_server.py ---

--- START OF FILE view.py ---

```py
# view.py
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from controller import ExitCommand # Import our custom exception
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from model import AppState, Model

if TYPE_CHECKING:
    from controller import Controller

class View:
    """
    The View is responsible for the presentation layer of the application.
    It renders the model's state to the terminal and captures user input.
    """
    def __init__(self, model: Model, controller: "Controller"):
        self.model = model
        self.controller = controller
        self._last_rendered_message_count = 0
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self.model.register_listener(self.on_model_update)

    async def on_model_update(self):
        """Callback triggered when the model's state changes."""
        self._render_status()
        self._render_new_messages()

    def _render_status(self):
        """Renders status messages like 'thinking' or errors."""
        if self.model.application_state == AppState.AGENT_IS_THINKING:
            print("...")
        elif self.model.application_state == AppState.ERROR:
            error_msg = self.model.last_error_message or "An unknown error occurred."
            print(f"\n[ERROR] {error_msg}")
        elif self.model.application_state == AppState.IDLE and self.model.last_success_message:
            # Show success messages
            print(f"\n[SUCCESS] {self.model.last_success_message}")
            # Clear the message after showing it
            self.model.last_success_message = None

    def _render_new_messages(self):
        """Renders only new messages from the agent."""
        current_message_count = len(self.model.conversation_history)
        if current_message_count > self._last_rendered_message_count:
            new_messages = self.model.conversation_history[self._last_rendered_message_count:]
            for message in new_messages:
                # We only print the assistant's messages to avoid duplication.
                if message.role == 'assistant':
                    self._print_message(message)
            self._last_rendered_message_count = current_message_count

    def _print_message(self, message: PromptMessageMultipart):
        """Formats and prints a single message from the agent."""
        print("\n" + "---" * 20)
        print("Agent:")
        text_content = message.last_text()
        indented_text = "\n".join(["    " + line for line in text_content.splitlines()])
        print(indented_text)

    async def _get_user_input_async(self) -> str:
        """Asynchronously captures user input."""
        print("\n" + "---" * 20 + "\n")
        print("You:")
        try:
            return await self._prompt_session.prompt_async("")
        except (KeyboardInterrupt, EOFError):
            return "/exit"

    def print_startup_message(self):
        """Prints the initial welcome message."""
        print("Agent is ready. Type a message or '/exit' to quit.")
        prefs = self.model.user_preferences
        if prefs.get("auto_save_enabled"):
            filename = prefs.get("auto_save_filename", "the context directory.")
            print(f"Auto-saving is ON. History will be saved to '{filename}'")

    async def run_main_loop(self):
        """The main loop to capture user input."""
        self.print_startup_message()
        while True:
            user_input = await self._get_user_input_async()
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                # Break the loop gracefully when the controller signals to exit.
                break
```

--- END OF FILE view.py ---

