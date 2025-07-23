import logging
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from agent_registry import DEFAULT_AGENT
from commands import ExitCommand, SwitchAgentCommand
from controller import Controller
from model import Model, Interaction

if TYPE_CHECKING:
    from model import Interaction

logger = logging.getLogger(__name__)

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
        height: 1fr;
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
        self.model = Model(default_agent_name=agent_name)
        self.controller = Controller(self.model, self)
        self._last_rendered_interaction_count = 0
        self.model.register_listener(self.on_model_update)

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=True)
        yield Input(placeholder="Enter your prompt or type /help...")
        yield Footer()

    async def on_mount(self) -> None:
        """Initializes the application and creates the first session."""
        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        await self.model.create_session()
        
        self.title = "Agent Dashboard"
        await self.on_model_update()
        
        welcome_interaction = Interaction(
            Text.from_markup("🤖 Agent is ready. Say 'Hi' or type a command."),
            metadata={"user-facing": True, "type": "info"}
        )
        await self.model.add_interaction_to_active_session(welcome_interaction)

    async def on_model_update(self) -> None:
        """Schedules UI updates when the model's state changes."""
        self.call_later(self.render_log)
        self.call_later(self.update_header)

    def render_log(self) -> None:
        """Renders the model's display_history to the chat log."""
        display_history = self.model.display_history
        if self._last_rendered_interaction_count != len(display_history):
            self.log_widget.clear()
            for interaction in display_history:
                if isinstance(interaction.contents, Text):
                    self.log_widget.write(interaction.contents)
                elif isinstance(interaction.contents, list):
                    for msg in interaction.contents:
                        role = msg.role.capitalize()
                        color = "blue" if msg.role == "user" else "magenta"
                        self.log_widget.write(Text.from_markup(f"[bold {color}]{role}:[/] {msg.last_text()}"))
                else:
                    self.log_widget.write(str(interaction.contents))

            self._last_rendered_interaction_count = len(display_history)

    def update_header(self) -> None:
        """Updates the header with the current agent and thinking status."""
        active_session = self.model.get_active_session()
        agent_name = active_session.agent_name if active_session else "N/A"
        
        if self.model.is_thinking:
            self.sub_title = f"Agent: [bold]{agent_name}[/] 🤔 Thinking..."
        else:
            self.sub_title = f"Active Agent: [bold]{agent_name}[/]"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles user input, delegating to the controller."""
        user_input = event.value
        if not user_input:
            return
        
        self.input_widget.clear()
        
        async def process_input_with_exception_handling():
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                self.exit()
            except SwitchAgentCommand as e:
                # A new session is created to provide a clean slate for the new agent.
                await self.model.create_session(agent_name=e.agent_name)
                switch_interaction = Interaction(
                    Text.from_markup(f"[bold green]Success:[/bold green] Switched to agent '{e.agent_name}'."),
                    metadata={"user-facing": True, "type": "success"}
                )
                await self.model.add_interaction_to_active_session(switch_interaction)
            except Exception as e:
                logger.critical("Unhandled exception in input processing", exc_info=True)
                error_text = Text.from_markup(f"[bold red]Critical Error:[/bold red] An unexpected error occurred: {e}")
                error_interaction = Interaction(error_text, metadata={"user-facing": True, "type": "error"})
                await self.model.add_interaction_to_active_session(error_interaction)

        self.run_worker(process_input_with_exception_handling(), exclusive=True)
