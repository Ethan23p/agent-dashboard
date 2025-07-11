# textual_view.py
from typing import TYPE_CHECKING

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from controller import ExitCommand, SwitchAgentCommand
from model import AppState, Model

if TYPE_CHECKING:
    from controller import Controller

class AgentDashboardApp(App):
    """The Textual-based user interface for the agent dashboard."""

    TITLE = "Agent Dashboard"
    SUB_TITLE = "A fast-agent client"
    CSS = """
    Screen {
        background: $surface;
    }
    #chat-log {
        margin: 1 2;
        border: round $primary;
        background: $panel;
    }
    Input {
        dock: bottom;
        margin: 0 1 1 1;
    }
    """
    BINDINGS = [
        ("ctrl+d", "toggle_dark", "Toggle Dark Mode"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model: Model, controller: "Controller"):
        super().__init__()
        self.model = model
        self.controller = controller
        self._last_rendered_message_count = 0
        # Register the view as a listener to the model
        self.model.register_listener(self.on_model_update)

    def compose(self) -> ComposeResult:
        """Create the core UI widgets."""
        yield Header()
        yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=True)
        yield Input(placeholder="Enter your prompt or type /help...")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        self.log_widget.write("🤖 Agent is ready. Say 'Hi' or type a command.")

    async def on_model_update(self) -> None:
        """
        Callback triggered when the model's state changes.
        
        This is now a standard awaitable coroutine. We use `call_later`
        to ensure the UI updates happen safely on the main app thread.
        """
        self.call_later(self._render_status)
        self.call_later(self._render_new_messages)

    def _render_status(self) -> None:
        """Renders status messages like 'thinking' or errors to the log."""
        if self.model.application_state == AppState.AGENT_IS_THINKING:
            self.log_widget.write("...")
        elif self.model.application_state == AppState.ERROR and self.model.last_error_message:
            self.log_widget.write(f"[bold red]💥 Error:[/bold red] {self.model.last_error_message}")
            self.model.last_error_message = None # Clear after displaying
        elif self.model.application_state == AppState.IDLE and self.model.last_success_message:
            self.log_widget.write(f"[bold green]✅ Success:[/bold green] {self.model.last_success_message}")
            self.model.last_success_message = None # Clear after displaying

    def _render_new_messages(self) -> None:
        """Renders only new messages from the conversation history to the log."""
        current_message_count = len(self.model.conversation_history)
        if current_message_count > self._last_rendered_message_count:
            new_messages = self.model.conversation_history[self._last_rendered_message_count:]
            for message in new_messages:
                if message.role == 'user':
                    self.log_widget.write(f"[bold blue]You:[/bold blue] {message.last_text()}")
                elif message.role == 'assistant':
                    self.log_widget.write(f"[bold magenta]Agent:[/bold magenta] {message.last_text()}")
            self._last_rendered_message_count = current_message_count

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles the submission of user input by starting a worker."""
        user_input = event.value
        if not user_input:
            return
        
        # Clear the input immediately for a responsive feel
        self.input_widget.clear()
        
        # Run the entire controller interaction in a background worker
        # to keep the UI from freezing during agent processing.
        self.run_worker(self.handle_submission(user_input), exclusive=True)

    async def handle_submission(self, user_input: str) -> None:
        """
        This method is run in a worker. It processes the user input
        and handles commands that control the app's lifecycle.
        """
        try:
            await self.controller.process_user_input(user_input)
        except ExitCommand:
            self.exit()
        except SwitchAgentCommand as e:
            self.exit(result=e.agent_name)
