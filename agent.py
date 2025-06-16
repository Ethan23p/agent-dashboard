import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List

# prompt_toolkit is used for async input in the terminal
from prompt_toolkit import PromptSession

# Core fast-agent components
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams

# MCP type for handling conversation history
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# --- 1. AGENT DEFINITION (Unchanged) ---

fast = FastAgent("Minimal Controllable Agent")

@fast.agent(
    name="base_agent",
    instruction="You are a helpful and concise assistant. You have access to a filesystem.",
    servers=["filesystem"],
    use_history=False,
    request_params=RequestParams(max_tokens=2048)
)
async def main():
    """
    This function serves as the main entry point and contains the client logic
    for interacting with our defined agent.
    """
    async with fast.run() as agent_app:
        # --- 2. CLIENT LOGIC (Updated) ---
        conversation_history: List[PromptMessageMultipart] = []

        # Create a PromptSession for handling asynchronous user input.
        prompt_session = PromptSession()

        print("Agent is ready. Type '/save [filename.json]' to save history, or 'exit' to quit.")

        while True:
            # --- New Feature: UI Boundary ---
            print("\n" + "---" * 20)

            # --- New Feature: Asynchronous Input ---
            # This is now a non-blocking call that works with the asyncio event loop.
            try:
                user_input = await prompt_session.prompt_async("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if user_input.lower() in ["exit", "quit"]:
                print("Session ended.")
                break

            # --- New Feature: Save Conversation History ---
            if user_input.strip().lower().startswith('/save'):
                await handle_save_command(user_input, conversation_history)
                continue # Continue to the next loop iteration without sending '/save' to the agent

            user_message = Prompt.user(user_input)
            conversation_history.append(user_message)

            try:
                response_message = await agent_app.base_agent.generate(conversation_history)
                conversation_history.append(response_message)

                # --- Updated Response Handling with UI Boundary ---
                print("Agent:", end=" ", flush=True)
                for content_part in response_message.content:
                    if content_part.type == "text":
                        print(content_part.text, end="")
                print()

            except Exception as e:
                print(f"\n[ERROR] An error occurred: {e}")


async def handle_save_command(user_input: str, history: List[PromptMessageMultipart]):
    """
    Parses the /save command and saves the conversation history to a file.
    """
    parts = user_input.strip().split()
    # Use a timestamp for the default filename if none is provided.
    default_filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename = parts[1] if len(parts) > 1 else default_filename

    # Ensure the filename ends with .json
    if not filename.endswith('.json'):
        filename += '.json'

    print(f"Saving conversation to '{filename}'...")
    await save_conversation_to_file(history, filename)


async def save_conversation_to_file(history: List[PromptMessageMultipart], filename: str):
    """
    Serializes the conversation history to a JSON file.
    This saves the full, structured MCP message data.
    """
    # --- Future Feature: Context Management & State Restoration ---
    # This function is the counterpart to a future "load_conversation" function.
    # It saves the state in a structured way that can be perfectly restored.
    try:
        # Convert each PromptMessageMultipart object into a serializable dictionary
        serializable_history = [message.model_dump() for message in history]

        with open(filename, 'w', encoding='utf-8') as f:
            # Use json.dump for pretty-printing the output
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Conversation history saved to {filename}")
    except IOError as e:
        print(f"[ERROR] Could not write to file {filename}: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during serialization: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient interrupted. Exiting.")