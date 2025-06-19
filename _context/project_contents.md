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

from agent_definitions import fast  # Import the fast_agent instance
from ui_manager import (
    get_user_input_async,
    print_agent_response,
    print_message,
    print_shutdown_message,
    print_startup_message,
    print_user_prompt_indicator,
)

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# --- Application Configuration ---
# This is now the central, developer-facing config area.
ENABLE_AUTO_SAVE = True
CONTEXT_DIR = "_context"

# --- Core Application Logic ---

class ChatSession:
    """
    Manages the state and logic for a single conversation.
    This class embodies the "Tell, Don't Ask" principle by encapsulating
    its own history and behavior.
    """
    def __init__(self, agent, session_id: str):
        self.agent = agent
        self.session_id = session_id
        self.history: List[PromptMessageMultipart] = []
        self.auto_save_filename = f"{CONTEXT_DIR}/{self.session_id}.json"

    async def process_user_turn(self, user_input: str) -> str:
        """
        Tells the session to process one full turn of conversation.
        Returns the final text from the agent.
        """
        user_message = Prompt.user(user_input)
        self.history.append(user_message)

        try:
            response_message = await self.agent.generate(self.history)
            self.history.append(response_message)
            return response_message.last_text()
        except Exception as e:
            print_message(f"An error occurred: {e}", style="error")
            # Remove the failed user message to prevent it from poisoning the history.
            self.history.pop()
            return "I encountered an error. Please try again."

    async def save_history(self, filename: str = None):
        """Tells the session to save its current history to a file."""
        # If no filename is provided, use the default auto-save filename.
        target_filename = filename or self.auto_save_filename

        # Ensure the context directory exists.
        import os
        os.makedirs(CONTEXT_DIR, exist_ok=True)

        print_message(f"Saving conversation to '{target_filename}'...")
        try:
            serializable_history = [message.model_dump() for message in self.history]
            with open(target_filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            print_message(f"Conversation history saved.", style="success")
        except Exception as e:
            print_message(f"Could not write to file {target_filename}: {e}", style="error")


async def main():
    """The main client application loop."""
    async with fast.run() as agent_app:
        # Create a unique ID for this chat session.
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Instantiate our ChatSession, telling it which agent to use.
        session = ChatSession(agent=agent_app.base_agent, session_id=session_id)

        print_startup_message(ENABLE_AUTO_SAVE, session.auto_save_filename)

        while True:
            print_user_prompt_indicator()
            user_input = await get_user_input_async()

            if user_input.strip().lower().startswith(('/exit', '/quit')):
                break

            if user_input.strip().lower().startswith('/save'):
                parts = user_input.strip().split()
                manual_filename = parts[1] if len(parts) > 1 else None
                await session.save_history(filename=manual_filename)
                continue

            # Tell the session to process the turn and get the final text.
            agent_text = await session.process_user_turn(user_input)

            # Tell the UI manager to display the response.
            if agent_text:
                print_agent_response(agent_text)

            # Tell the session to auto-save itself if enabled.
            if ENABLE_AUTO_SAVE:
                await session.save_history()

    # A small delay to prevent shutdown race conditions.
    # This should be removed/resolved gracefully if/when we transition to a frontend UI.
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_shutdown_message() # Use UI manager for consistent shutdown message
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

--- START OF FILE history_manager.py ---

```py
# history_manager.py
import json
from datetime import datetime
from typing import List

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

async def handle_save_command(user_input: str, history: List[PromptMessageMultipart]):
    """
    Parses the /save command and saves the conversation history to a file.
    """
    parts = user_input.strip().split()
    default_filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename = parts[1] if len(parts) > 1 else default_filename

    if not filename.endswith('.json'):
        filename += '.json'

    print(f"Saving conversation to '{filename}'...")
    await save_conversation_to_file(history, filename)

async def save_conversation_to_file(history: List[PromptMessageMultipart], filename: str):
    """
    Serializes the conversation history to a JSON file.
    This saves the full, structured MCP message data, which is ideal for
    perfect state restoration later.
    """
    try:
        serializable_history = [message.model_dump() for message in history]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        #print(f"[SUCCESS] Conversation history saved to {filename}")
    except IOError as e:
        print(f"[ERROR] Could not write to file {filename}: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during serialization: {e}")

# --- Future Feature: State Restoration ---
# async def load_conversation_from_file(filename: str) -> List[PromptMessageMultipart]:
#     """Loads and deserializes conversation history from a file."""
#     # This would be the counterpart to the save function.
#     pass
```

--- END OF FILE history_manager.py ---

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

# This module handles all terminal input and output.
# It defines HOW to display information, abstracting these details from other parts of the application.

# A single session object enables command history across multiple inputs within a run.
_prompt_session = PromptSession()

def print_startup_message(auto_save_enabled: bool, filename: str):
    """Prints the initial welcome and instructions."""
    print("Agent is ready. Type '/save [filename.json]' to save history, or '/exit' to quit.")
    if auto_save_enabled:
        print(f"Auto-saving is ON. History will be saved to '{filename}' after each turn.")

def print_shutdown_message():
    """Prints a clean shutdown message."""
    print("Shutting down...")

def print_agent_response(text: str):
    """Formats and prints the agent's text response.
    Indentation is used for visual distinction of the agent's output.
    """
    print("\nAgent:")
    indented_text = "\n".join(["    " + line for line in text.splitlines()])
    print(indented_text)

def print_user_prompt_indicator():
    """Prints the separator and the 'You:' prompt indicator for clarity."""
    print("\n" + "---" * 20 + "\n")
    print("You:")

def print_message(message: str, style: str = "info"):
    """Prints a formatted message, with optional styling prefixes."""
    prefix = ""
    if style.lower() == "error":
        prefix = "[ERROR] "
    elif style.lower() == "success":
        prefix = "[SUCCESS] "
    print(f"{prefix}{message}")

async def get_user_input_async() -> str:
    """Asynchronously gets input, returning '/exit' on interrupt for graceful shutdown."""
    try:
        return await _prompt_session.prompt_async("") # Empty prompt for a clean input line
    except (KeyboardInterrupt, EOFError):
        return "/exit" # Signals the main loop to exit

```

--- END OF FILE ui_manager.py ---

