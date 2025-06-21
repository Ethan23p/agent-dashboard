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

fast = FastAgent("Minimal Controllable Agent")

@fast.agent(
    name="base_agent",
    instruction="You are a helpful and concise assistant. You have access to a filesystem.",
    servers=["filesystem"],
    use_history=False,
    request_params=RequestParams(max_tokens=2048),
)
async def define_agents():
    """
    This function is a placeholder for the decorator. The `fast.run()`
    context manager will discover any agents defined in this file.
    defined in this file.
    """
    pass

```

--- END OF FILE agent_definitions.py ---

--- START OF FILE brain.py ---

```py
# brain.py
import asyncio
import json
import os
from datetime import datetime
from typing import List

# --- Module Imports ---
# We import our custom modules first, following best practices.
from agent_definitions import fast  # The `FastAgent` instance and its agent definitions.
from ui_manager import (
    get_user_input_async,
    print_agent_response,
    print_message,
    print_shutdown_message,
    print_startup_message,
    print_user_prompt_indicator,
)

# Core types from the fast-agent framework.
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# --- Application Configuration ---
# This section centralizes developer-facing settings for easy tweaking.
ENABLE_AUTO_SAVE = True
CONTEXT_DIR = "_context"

# --- Core Application Logic ---

class ChatSession:
    """
    Manages the state and logic for a single, self-contained conversation.
    """
    def __init__(self, agent, session_id: str):
        self.agent = agent
        self.session_id = session_id
        self.history: List[PromptMessageMultipart] = []
        self.auto_save_filename = f"{CONTEXT_DIR}/{self.session_id}.json"

    async def process_user_turn(self, user_input: str) -> str:
        """
        Tells the session to process one full turn of conversation. This includes
        updating its history and interacting with its assigned agent.

        Args:
            user_input: The text input from the user.

        Returns:
            The final text response from the agent.
        """
        user_message = Prompt.user(user_input)
        self.history.append(user_message)

        try:
            # The agent is given the entire history, allowing it to understand
            # the full context of the conversation for this turn.
            response_message = await self.agent.generate(self.history)
            self.history.append(response_message)
            return response_message.last_text()
        except Exception as e:
            print_message(f"An error occurred: {e}", style="error")
            # To maintain a clean state, we remove the user's last message if
            # the agent failed to generate a response.
            self.history.pop()
            return "I encountered an error. Please try again."

    async def save_history(self, filename: str = None):
        """
        Tells the session to save its current history to a file. This method
        encapsulates the logic for serialization and file I/O.
        """
        target_filename = filename or self.auto_save_filename
        os.makedirs(CONTEXT_DIR, exist_ok=True)

        print_message(f"Saving conversation to '{target_filename}'...")
        try:
            # `model_dump()` serializes the rich MCP message objects into a
            # standard dictionary format, suitable for JSON.
            serializable_history = [message.model_dump() for message in self.history]
            with open(target_filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            print_message(f"Conversation history saved.", style="success")
        except Exception as e:
            print_message(f"Could not write to file {target_filename}: {e}", style="error")


async def main():
    """The main client application loop and entry point."""
    # `fast.run()` is a context manager that handles the startup and
    # shutdown of all configured MCP servers.
    async with fast.run() as agent_app:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = ChatSession(agent=agent_app.base_agent, session_id=session_id)

        print_startup_message(ENABLE_AUTO_SAVE, session.auto_save_filename)

        while True:
            print_user_prompt_indicator()
            user_input = await get_user_input_async()

            # Command handling is checked before regular processing.
            if user_input.strip().lower().startswith(('/exit', '/quit')):
                break

            if user_input.strip().lower().startswith('/save'):
                parts = user_input.strip().split()
                manual_filename = parts[1] if len(parts) > 1 else None
                await session.save_history(filename=manual_filename)
                continue # Skip the rest of the loop for commands.

            # Tell the session to handle the core logic for the turn.
            agent_text = await session.process_user_turn(user_input)

            # Tell the UI manager to display the result.
            if agent_text:
                print_agent_response(agent_text)

            if ENABLE_AUTO_SAVE:
                await session.save_history()

    # A small delay to prevent shutdown race conditions.
    # This should be removed/resolved gracefully if/when we transition to a frontend UI.
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_shutdown_message() # Use UI manager for a consistent exit message.
```

--- END OF FILE brain.py ---

--- START OF FILE fastagent.config.yaml ---

```yaml
# fastagent.config.yaml

# --- Model Configuration ---
# Set the default model for all agents.
# You can override this per-agent in the decorator or with the --model CLI flag.
# Format: <provider>.<model_name> (e.g., openai.gpt-4o, anthropic.claude-3-5-sonnet-latest)
# Aliases like 'sonnet' or 'haiku' are also supported.
default_model: google.gemini-2.5-flash-preview-05-20

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
        - "H:\\Hume General\\Programming\\Repos\\agent-dashboard"
        - "H:\\Hume General\\AI Resources"
```

--- END OF FILE fastagent.config.yaml ---

--- START OF FILE pyproject.toml ---

```toml
[project]
name = "agent-dashboard"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.53.0",
    "mcp>=1.9.3",
    "python-dotenv>=1.1.0",
    "rich>=14.0.0",
]

```

--- END OF FILE pyproject.toml ---

--- START OF FILE ui_manager.py ---

```py
# ui_manager.py
from prompt_toolkit import PromptSession

# This module is responsible for the "Presentation Layer" of the application.
# It handles all direct interaction with the terminal, abstracting the details
# of input and output away from the core application logic in 'brain.py'.
# If we were to build a web UI, this file would be replaced, but 'brain.py'
# and 'agent_definitions.py' could remain largely unchanged.

# We create a single, module-level session object. This allows `prompt_toolkit`
# to remember user input history (e.g., using the up/down arrow keys)
# for the entire duration of the application run.
_prompt_session = PromptSession()

def print_startup_message(auto_save_enabled: bool, filename: str):
    """Prints the initial welcome and instructions to the user."""
    print("Agent is ready. Type '/save [filename]' to save history, or '/exit' to quit.")
    if auto_save_enabled:
        print(f"Auto-saving is ON. History will be saved to '{filename}' after each turn.")

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

def print_agent_response(text: str):
    """
    Formats and prints the agent's text response.
    The primary goal here is to visually distinguish the agent's output
    from the user's input, making the conversation easy to follow.
    """
    print("\nAgent:")
    # Indenting the response is a simple but effective way to create this distinction.
    indented_text = "\n".join(["    " + line for line in text.splitlines()])
    print(indented_text)

def print_user_prompt_indicator():
    """
    Prints the separator and the 'You:' prompt indicator. This clearly marks
    the beginning of the user's turn.
    """
    print("\n" + "---" * 20 + "\n")
    print("You:")

def print_message(message: str, style: str = "info"):
    """
    A general-purpose function for printing formatted system messages,
    such as status updates or errors.
    """
    prefix = ""
    if style.lower() == "error":
        prefix = "[ERROR] "
    elif style.lower() == "success":
        prefix = "[SUCCESS] "
    print(f"{prefix}{message}")

async def get_user_input_async() -> str:
    """
    Asynchronously gets input from the user.
    Using `prompt_toolkit` instead of the standard `input()` is crucial for
    an `asyncio` application, as it doesn't block the entire event loop.
    """
    try:
        # We pass an empty string to `prompt_async` because the "You:" prompt
        # is handled by `print_user_prompt_indicator` for consistent formatting.
        return await _prompt_session.prompt_async("")
    except (KeyboardInterrupt, EOFError):
        # A user pressing Ctrl+C or Ctrl+D is a signal to exit gracefully.
        # We return a specific command string that the main loop in 'brain.py'
        # can check for, rather than letting the exception crash the program.
        return "/exit"
```

--- END OF FILE ui_manager.py ---

