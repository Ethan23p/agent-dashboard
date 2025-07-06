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