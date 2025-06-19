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