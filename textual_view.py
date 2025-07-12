# textual_view.py
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from controller import ExitCommand, SwitchAgentCommand
from model import IdleState, AgentIsThinkingState, ErrorState, Model

if TYPE_CHECKING:
    from controller import Controller

class AgentDashboardApp(App):
    """The Textual-based user interface for the agent dashboard."""

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

    def __init__(self, model: Model, controller: "Controller", agent_name: str = "agent"):
        super().__init__()
        self.model = model
        self.controller = controller
        self.agent_name = agent_name  # Store the agent name
        self._last_rendered_message_count = 0
        # Register the view as a listener to the model
        self.model.register_listener(self.on_model_update)
        # Map state types to their rendering methods within the View
        self.state_renderers = {
            IdleState: self._render_idle,
            AgentIsThinkingState: self._render_thinking,
            ErrorState: self._render_error,
        }
        # Map message roles to their rendering methods
        self.message_renderers = {
            'user': self._render_user_message,
            'assistant': self._render_assistant_message,
        }

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
        
        # Set the initial title and sub_title
        self.title = "Agent Dashboard"
        self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"
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
        """Renders the status by dispatching to the correct method."""
        state_type = type(self.model.application_state)
        # Polymorphic call without the if/elif block
        renderer = self.state_renderers.get(state_type)
        if renderer:
            renderer()

    # --- Status methods now update self.sub_title ---

    def _render_idle(self) -> None:
        if self.model.last_success_message:
            self.sub_title = f"✅ {self.model.last_success_message}"
            self.model.last_success_message = None
        else:
            # Revert to showing the active agent when idle
            self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"

    def _render_thinking(self) -> None:
        self.sub_title = "🤔 Thinking..."

    def _render_error(self) -> None:
        if self.model.last_error_message:
            self.sub_title = f"💥 Error: {self.model.last_error_message}"
            self.model.last_error_message = None

    # --- Specific message rendering methods ---

    def _render_user_message(self, message) -> None:
        """Renders a user message."""
        log_message = Text.from_markup(f"[bold blue]You:[/bold blue] {message.last_text()}")
        self.log_widget.write(log_message)

    def _render_assistant_message(self, message) -> None:
        """Renders an assistant message."""
        log_message = Text.from_markup(f"[bold magenta]Agent:[/bold magenta] {message.last_text()}")
        self.log_widget.write(log_message)

    def _render_new_messages(self) -> None:
        """Renders only new messages from the conversation history to the log."""
        current_message_count = len(self.model.conversation_history)
        if current_message_count > self._last_rendered_message_count:
            new_messages = self.model.conversation_history[self._last_rendered_message_count:]
            for message in new_messages:
                # Polymorphic dispatch based on message role
                renderer = self.message_renderers.get(message.role)
                if renderer:
                    renderer(message)
                else:
                    # Fallback for unknown roles
                    self.log_widget.write(f"[dim]{message.role}:[/dim] {message.last_text()}")
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
