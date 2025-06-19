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
