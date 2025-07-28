import logging
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, RichLog

from agent_registry import DEFAULT_AGENT
from commands import ExitCommand, SwitchAgentCommand
from controller import Controller
from model import Interaction, Model

if TYPE_CHECKING:
    from model import Interaction

logger = logging.getLogger(__name__)

COLORS = {
    "primary": "#262624",
    "secondary": "#1F1E1D",
    "text": "#BFBDB8",
    "accent": "#D97059",
    "border": "#BFAF80",
    "emphasis": "#BFAF80",
}

CUSTOM_THEME = Theme(
    "creamy-dark",
    COLORS["primary"],
    secondary=COLORS["secondary"],
    foreground=COLORS["text"],
    accent=COLORS["accent"],
    surface=COLORS["secondary"],
    background=COLORS["primary"],
    panel=COLORS["secondary"],
    error=COLORS["accent"],
    warning=COLORS["accent"],
    success=COLORS["border"],
    variables={
        "border": COLORS["border"],
        "emphasis": COLORS["emphasis"],
    }
)

class AgentDashboardApp(App):
    """The Textual-based user interface for the agent dashboard."""

    CSS_PATH = "dashboard.tcss"
    TITLE = "Agent Dashboard"
    SUB_TITLE = "Interface for interacting with sophisticated AI agents."
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
        """Create child widgets for the app."""
        yield Header()
        with Container(id="app-grid"):
            yield ListView(id="session-list")
            with Container(id="main-content"):
                yield RichLog(id="chat-history", wrap=True, highlight=False, markup=True)
                with Horizontal(id="input-bar"):
                    yield Input(placeholder="How can I help you today?", id="message-input")
                    yield Button("↑", id="send-button")
        yield Footer()

    async def on_mount(self) -> None:
        """Initializes the application and creates the first session."""
        self.register_theme(CUSTOM_THEME)
        self.theme = "creamy-dark"

        self.log_widget = self.query_one(RichLog)
        self.input_widget = self.query_one(Input)
        self.input_widget.focus()
        
        # (1)! The model listener will automatically trigger the first UI update.
        await self.model.create_session()
        
        self.title = "Agent Dashboard"
        # (2)! Removed the direct call to self.on_model_update() to prevent the race condition.
        
        welcome_interaction = Interaction(
            Text.from_markup("🤖 Agent is ready. Say 'Hi' or type a command."),
            metadata={"user-facing": True, "type": "info"}
        )
        # This will trigger the second, necessary UI update.
        await self.model.add_interaction_to_active_session(welcome_interaction)

    async def on_model_update(self) -> None:
        """Schedules UI updates when the model's state changes."""
        self.call_later(self.update_session_list)
        self.call_later(self.render_log)
        self.call_later(self.update_header)

    def update_session_list(self) -> None:
        """Renders the list of sessions in the sidebar idempotently."""
        session_list_view = self.query_one("#session-list", ListView)
        
        # (3)! Add a guard to prevent re-rendering if the session list is already correct.
        current_ids = {item.id for item in session_list_view.children if item.id is not None}
        model_ids = {session.id for session in self.model.sessions}
        if current_ids == model_ids:
            # The view is already in sync, no need to do anything.
            return

        session_list_view.clear()
        
        active_session_id = self.model.active_session_id
        highlighted_index = 0

        for i, session in enumerate(self.model.sessions):
            session_label = f"{session.agent_name} ({session.id.split('_')[1]})"
            list_item = ListItem(Label(session_label), id=session.id)
            session_list_view.append(list_item)
            if session.id == active_session_id:
                highlighted_index = i
        
        if session_list_view.children:
            session_list_view.index = highlighted_index

    def render_log(self) -> None:
        """Renders the model's display_history to the chat log."""
        display_history = self.model.display_history
        
        if self._last_rendered_interaction_count == len(display_history) and self.log_widget.children:
            return

        self.log_widget.clear()
        for interaction in display_history:
            if isinstance(interaction.contents, Text):
                self.log_widget.write(interaction.contents)
            elif isinstance(interaction.contents, list):
                for msg in interaction.contents:
                    role = msg.role.capitalize()
                    color = COLORS['emphasis'] if msg.role == "user" else COLORS['accent']
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

    @on(ListView.Selected, "#session-list")
    def on_session_selected(self, event: ListView.Selected) -> None:
        """Handle the selection of a new session in the sidebar."""
        if event.item.id:
            self.run_worker(self.model.set_active_session(event.item.id))

    def action_send_message(self) -> None:
        """Called when the user sends a message."""
        user_input = self.input_widget.value
        if not user_input:
            return
        
        self.input_widget.clear()
        
        async def process_input_with_exception_handling():
            try:
                await self.controller.process_user_input(user_input)
            except ExitCommand:
                self.exit()
            except SwitchAgentCommand as e:
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

    @on(Button.Pressed, "#send-button")
    def on_button_pressed(self) -> None:
        """Handle send button clicks."""
        self.action_send_message()

    @on(Input.Submitted, "#message-input")
    def on_input_submitted(self) -> None:
        """Handle 'Enter' key press in the input field."""
        self.action_send_message()