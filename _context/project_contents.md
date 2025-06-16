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

--- START OF FILE agent.py ---

```py
# agent.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# --- AGENT DEFINITION ---
# This file now only defines the agent and its capabilities.
# The client logic that runs it is in cli.py.

# Instantiate the main fast-agent application object.
fast = FastAgent("Minimal Controllable Agent")

# Decorator to define our agent.
@fast.agent(
    name="base_agent",
    instruction="You are a helpful and concise assistant. You have access to a filesystem.",
    servers=["filesystem"],
    use_history=False,
    request_params=RequestParams(max_tokens=2048)
)
async def define_agents():
    """
    This function is now just a placeholder for the decorator.
    The `fast.run()` context manager in cli.py will discover any agents
    defined in this file.
    """
    # We don't need any logic here anymore.
    pass
```

--- END OF FILE agent.py ---

--- START OF FILE cli.py ---

```py
# cli.py
import asyncio
from typing import List

from prompt_toolkit import PromptSession

# Import our modularized components
from agent import fast  # Import the fast_agent instance from our agent definition file
from history_manager import handle_save_command

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


async def main():
    """
    The main client application loop.
    """
    async with fast.run() as agent_app:
        conversation_history: List[PromptMessageMultipart] = []
        prompt_session = PromptSession()

        print("Agent is ready. Type '/save [filename.json]' to save history, or '/exit' to quit.")

        while True:
            # --- NEW: Enhanced UI Formatting ---
            print("\n" + "---" * 20 + "\n")
            print("You:")
            try:
                # The prompt is now just an empty string for a clean input line.
                user_input = await prompt_session.prompt_async("")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if user_input.strip().lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Session ended.")
                break

            if user_input.strip().lower().startswith('/save'):
                await handle_save_command(user_input, conversation_history)
                continue

            user_message = Prompt.user(user_input)
            conversation_history.append(user_message)

            try:
                response_message = await agent_app.base_agent.generate(conversation_history)
                conversation_history.append(response_message)

                # --- NEW: Enhanced UI Formatting for Agent Response ---
                print("\nAgent:")
                for content_part in response_message.content:
                    if content_part.type == "text":
                        # Indent the agent's response for clarity.
                        indented_text = "\n".join(["    " + line for line in content_part.text.splitlines()])
                        print(indented_text)

            except Exception as e:
                print(f"\n[ERROR] An error occurred: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient interrupted.")
    finally:
        # --- NEW: Graceful Shutdown Pause ---
        # This pause helps prevent the "I/O operation on closed pipe" error on Windows
        # by giving background MCP server processes a moment to terminate cleanly.
        # This can be removed if a more sophisticated shutdown signal is implemented later.
        print("Shutting down...")
        # Note: time.sleep() is blocking and should not be used in async code.
        # We use asyncio.sleep() instead.
        asyncio.run(asyncio.sleep(0.5))
```

--- END OF FILE cli.py ---

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
        print(f"[SUCCESS] Conversation history saved to {filename}")
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

