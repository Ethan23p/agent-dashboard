# textual_view.py
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.containers import Vertical

from controller import ExitCommand, SwitchAgentCommand
from model import Model, Interaction

class ElicitationPrompt(Static):
    """A temporary widget to display the elicitation prompt."""
    DEFAULT_CSS = """
    ElicitationPrompt {
        background: $boost; padding: 0 1; margin-bottom: 1; border: round yellow;
    }
    """

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
        self.agent_name = agent_name
        self._last_rendered_message_count = 0
        self.model.register_listener(self.on_model_update)

    def compose(self) -> ComposeResult:
        """Create the core UI widgets."""
        yield Header()
        yield Vertical(id="elicitation-container")
        yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=True)
        yield Input(placeholder="Enter your prompt or type /help...")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        self.title = "Agent Dashboard"
        self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"
        self.log_widget.write("🤖 Agent is ready. Say 'Hi' or type a command.")

    async def on_model_update(self) -> None:
        """
        Called when the model state changes. Uses call_later to ensure UI updates
        happen safely on the main app thread.
        """
        self.call_later(self.render_log)
        self.call_later(self.update_header) # This will handle thinking status
        self.call_later(self.render_elicitation) # This handles the elicitation prompt widget

    def render_log(self) -> None:
        """Renders the entire log from the model."""
        self.log_widget.clear()
        for interaction in self.model.conversation_log:
            self.log_widget.write(interaction.content)

    def render_elicitation(self) -> None:
        """Renders or removes the elicitation prompt based on model state."""
        container = self.query_one("#elicitation-container")
        context = self.model.active_elicitation_context

        has_prompt = len(container.children) > 0

        if context and not has_prompt:
            prompt_widget = ElicitationPrompt(f"🤖 {context.params.message}")
            container.mount(prompt_widget)
            self.input_widget.placeholder = "Respond to the agent's question..."
            self.input_widget.focus()
        elif not context and has_prompt:
            container.remove_children()
            self.input_widget.placeholder = "Enter your prompt or type /help..."

    def update_header(self) -> None:
        """Updates the header based on the model's thinking status."""
        if self.model.is_thinking:
            self.sub_title = "🤔 Thinking..."
        else:
            self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value
        if not user_input:
            return
        
        self.input_widget.clear()
        self.run_worker(self.controller.process_user_input(user_input), exclusive=True)
