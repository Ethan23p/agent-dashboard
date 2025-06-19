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