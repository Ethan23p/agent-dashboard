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