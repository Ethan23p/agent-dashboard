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