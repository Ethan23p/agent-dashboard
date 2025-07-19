# textual_view.py
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.containers import Vertical
from model import Task

from controller import Controller, ExitCommand, SwitchAgentCommand
from model import Model, Interaction
from agent_registry import DEFAULT_AGENT

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

    def __init__(self, agent_name: str = DEFAULT_AGENT):
        super().__init__()
        self.model = Model()
        self.controller = Controller(self.model, self)
        self.agent_name = agent_name
        self._last_rendered_message_count = 0
        self.model.register_listener(self.on_model_update)

    def compose(self) -> ComposeResult:
        """Create the core UI widgets."""
        yield Header()
        yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=True)
        yield Input(placeholder="Enter your prompt or type /help...")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when first mounted."""
        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        self.title = "Agent Dashboard"
        self.sub_title = f"Active Agent: [bold]{self.agent_name}[/]"
        self.log_widget.write("🤖 Agent is ready. Say 'Hi' or type a command.")

    async def on_model_update(self) -> None:
        """Handle model state changes by updating the UI safely on the main thread."""
        self.call_later(self.render_log)
        self.call_later(self.update_header)

    def render_log(self) -> None:
        # This now renders tasks instead of a simple chat log
        if self._last_rendered_message_count != len(self.model.interactions):
            # Render only new interactions to be more efficient
            new_interactions = self.model.interactions[self._last_rendered_message_count:]
            for interaction in new_interactions:
                # Render primary dialogue and system messages, but hide verbose status updates.
                if interaction.tag in ("user_prompt", "agent_response", "success", "error", "task_created"):
                    content_text = interaction.content
                    if isinstance(content_text, Text) and interaction.meta.get("task_id"):
                        # Shorten the full task ID for display in the log
                        full_task_id = interaction.meta["task_id"]
                        parts = full_task_id.split('-')
                        if len(parts) > 1:
                            short_id = f"{parts[0][-6:]}-{parts[1][:4]}" # e.g., 125747-Hutt
                            content_text = content_text.copy()
                            content_text.plain = content_text.plain.replace(full_task_id, short_id)
                    self.log_widget.write(content_text)

            self._last_rendered_message_count = len(self.model.interactions)

    def update_header(self) -> None:
        active_task = self.model.get_active_task()
        if active_task:
            persona = active_task.id.split('-')[1] if '-' in active_task.id else "Default"
            active_task_id_display = f"Session: [bold cyan]{persona}[/]"
        else:
            active_task_id_display = "No Active Session"

        if self.model.is_thinking:
            self.sub_title = "🤔 Thinking..."
        else:
            self.sub_title = f"Agent: [bold]{self.agent_name}[/] | {active_task_id_display}"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value
        if not user_input:
            return
        
        self.input_widget.clear()
        
        async def process_input_with_exit_handling():
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                # Gracefully exit the application
                self.exit()
        
        self.run_worker(process_input_with_exit_handling(), exclusive=True)
