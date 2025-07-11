# Contents of the 'agent-dashboard' project

--- START OF FILE .gitignore ---

```
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

```

--- END OF FILE .gitignore ---

--- START OF FILE agent-dashboard.code-workspace ---

```code-workspace
{
	"folders": [
		{
			"name": "agent-dashboard",
			"path": "."
		},
		{
			"path": "../context_for_MCP_and_fast-agent"
		}
	],
	"settings": {}
}
```

--- END OF FILE agent-dashboard.code-workspace ---

--- START OF FILE agent_definitions.py ---

```py
# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.
#
# NOTE: All agents should use use_history=False since we manage conversation
# history ourselves in the Model class and pass it explicitly to the agent.

# Simple single agent for basic operations
minimal_agent = FastAgent("Minimal Agent")

@minimal_agent.agent(
    name="agent",
    instruction="""
    You are a helpful assistant that can perform various operations.
    You can read files, write files, and list directory contents.
    Always be helpful and provide clear responses to user requests.
    """,
    servers=["filesystem", "fetch", "sequential-thinking"],
    request_params=RequestParams(maxTokens=2048),
    use_history=False 
)

async def agent():
    """ This function is a placeholder for the decorator. """
    pass

# Example of a second agent with different characteristics
coding_agent = FastAgent("Coding Assistant")

@coding_agent.agent(
    name="agent",
    instruction="""
    You are a specialized coding assistant. You excel at:
    - Code review and suggestions
    - Debugging and problem-solving
    - Explaining complex technical concepts
    - Providing code examples and best practices
    
    Always provide clear, well-documented code examples when relevant.
    """,
    servers=["filesystem"],
    request_params=RequestParams(maxTokens=4096),
    use_history=False
)

async def coding_agent_func():
    """ This function is a placeholder for the decorator. """
    pass

# Agent Registry - maps agent names to their FastAgent instances
AGENT_REGISTRY = {
    "minimal": minimal_agent,
    "coding": coding_agent,
}

def get_agent(agent_name: str = "minimal"):
    """
    Get an agent by name from the registry.
    
    Args:
        agent_name: The name of the agent to retrieve
        
    Returns:
        The FastAgent instance for the requested agent
        
    Raises:
        KeyError: If the agent name is not found in the registry
    """
    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """Return a list of available agent names."""
    return list(AGENT_REGISTRY.keys())

```

--- END OF FILE agent_definitions.py ---

--- START OF FILE config/.python-version ---

```
3.13

```

--- END OF FILE config/.python-version ---

--- START OF FILE controller.py ---

```py
# controller.py
import asyncio
import random
from typing import TYPE_CHECKING

from mcp_agent.core.prompt import Prompt
from model import AppState, Model

if TYPE_CHECKING:
    from mcp_agent.core.agent_app import AgentApp

class ExitCommand(Exception):
    """Custom exception to signal a graceful exit from the main loop."""
    pass

class SwitchAgentCommand(Exception):
    """Custom exception to signal switching to a different agent."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        super().__init__(f"Switch to agent: {agent_name}")

class Controller:
    """
    The Controller contains the application's business logic. It responds
    to user input from the View and orchestrates interactions between the
    Model and the Agent (fast-agent).
    """
    def __init__(self, model: Model, agent_app: "AgentApp"):
        self.model = model
        self.agent_app = agent_app
        # Get the first (default) agent without knowing its name
        # For now, we'll use the direct agent access since we only have one agent
        # This can be enhanced later when we support multiple agents
        self.agent = agent_app.agent

    async def process_user_input(self, user_input: str):
        """
        The main entry point for handling actions initiated by the user.
        It parses the input to determine if it's a command or a prompt
        for the agent.
        """
        stripped_input = user_input.strip()

        if not stripped_input:
            return

        if stripped_input.lower().startswith('/'):
            await self._handle_command(stripped_input)
        else:
            await self._handle_agent_prompt(stripped_input)

    async def _handle_command(self, command_str: str):
        """Parses and executes client-side commands like /save or /exit."""
        parts = command_str.lower().split()
        command_name = parts[0][1:]  # remove the '/'
        args = parts[1:]

        command_map = {
            'exit': self._cmd_exit,
            'quit': self._cmd_exit,
            'save': self._cmd_save,
            'load': self._cmd_load,
            'clear': self._cmd_clear,
            'switch': self._cmd_switch,
            'agents': self._cmd_list_agents,
        }

        handler = command_map.get(command_name)
        if handler:
            await handler(args)
        else:
            await self.model.set_state(AppState.ERROR, error_message=f"Unknown command: /{command_name}")

    async def _cmd_exit(self, args):
        raise ExitCommand()

    async def _cmd_switch(self, args):
        """Switch to a different agent."""
        if not args:
            await self.model.set_state(AppState.ERROR, error_message="Please provide an agent name: /switch <agent_name>")
            return
        
        agent_name = args[0]
        # Import here to avoid circular imports
        from agent_definitions import list_available_agents
        available_agents = list_available_agents()
        
        if agent_name not in available_agents:
            await self.model.set_state(
                AppState.ERROR, 
                error_message=f"Agent '{agent_name}' not found. Available agents: {', '.join(available_agents)}"
            )
            return
        
        await self.model.set_state(AppState.IDLE, success_message=f"Switching to {agent_name} agent...")
        raise SwitchAgentCommand(agent_name)

    async def _cmd_list_agents(self, args):
        """List available agents."""
        from agent_definitions import list_available_agents
        available_agents = list_available_agents()
        await self.model.set_state(
            AppState.IDLE, 
            success_message=f"Available agents: {', '.join(available_agents)}"
        )

    async def _cmd_save(self, args):
        filename = args[0] if args else None
        # If filename provided, ensure it's in the context directory
        if filename and not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = self.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        success = await self.model.save_history_to_file(filename)
        if success:
            await self.model.set_state(AppState.IDLE, success_message="History saved successfully.")
        else:
            await self.model.set_state(AppState.ERROR, error_message="Failed to save history.")

    async def _cmd_load(self, args):
        if not args:
            await self.model.set_state(AppState.ERROR, error_message="Please provide a filename: /load <filename>")
            return
        filename = args[0]
        # If filename doesn't start with path separator, assume it's in context directory
        if not filename.startswith('/') and not filename.startswith('\\'):
            context_dir = self.model._get_context_dir()
            filename = f"{context_dir}/{filename}"
        success = await self.model.load_history_from_file(filename)
        if success:
            await self.model.set_state(AppState.IDLE, success_message="History loaded successfully.")
        else:
            await self.model.set_state(AppState.ERROR, error_message="Failed to load history.")

    async def _cmd_clear(self, args):
        await self.model.clear_history()
        await self.model.set_state(AppState.IDLE, success_message="Conversation history cleared.")


    async def _handle_agent_prompt(self, user_prompt: str):
        """
        Manages the full lifecycle of a conversational turn with the agent,
        now with a retry mechanism.
        """
        await self.model.set_state(AppState.AGENT_IS_THINKING)
        user_message = Prompt.user(user_prompt)
        await self.model.add_message(user_message)

        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                # The core agent call
                response_message = await self.agent.generate(
                    self.model.conversation_history
                )
                await self.model.add_message(response_message)
                
                # If successful, break the loop
                await self.model.set_state(AppState.IDLE)
                if self.model.user_preferences.get("auto_save_enabled"):
                    await self.model.save_history_to_file()
                return # Exit the method on success

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    await self.model.set_state(AppState.ERROR, error_message=f"Agent Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    await self.model.set_state(AppState.ERROR, error_message=f"Agent Error after {max_retries} attempts: {e}")
                    await self.model.pop_last_message() # Roll back the user message
                    return # Exit after final failure
```

--- END OF FILE controller.py ---

--- START OF FILE docs/AGENT_SELECTION.md ---

```md
# Agent Selection System

The agent dashboard now supports flexible agent selection with the ability to switch between different agents at runtime.

## Available Agents

- **minimal**: A basic assistant for general operations
- **coding**: A specialized coding assistant with enhanced programming capabilities

## Usage

### Command Line Selection

Start with a specific agent:
```bash
python main.py --agent minimal
python main.py --agent coding
```

### Runtime Switching

While using the application, you can switch agents using commands:

- `/agents` - List all available agents
- `/switch <agent_name>` - Switch to a different agent

Example:
```
You: /agents
[SUCCESS] Available agents: minimal, coding

You: /switch coding
[SUCCESS] Switching to coding agent...
```

## Adding New Agents

To add a new agent, edit `agent_definitions.py`:

1. Create a new FastAgent instance:
```python
my_agent = FastAgent("My Agent Name")
```

2. Define the agent with decorator:
```python
@my_agent.agent(
    name="agent",
    instruction="Your agent instructions here...",
    servers=["filesystem"],
    request_params=RequestParams(maxTokens=2048),
    use_history=False
)
async def my_agent_func():
    pass
```

3. Add to the registry:
```python
AGENT_REGISTRY = {
    "minimal": minimal_agent,
    "coding": coding_agent,
    "my_agent": my_agent,  # Add your new agent here
}
```

## Architecture

The agent selection system uses:

- **Agent Registry**: Central registry mapping names to FastAgent instances
- **Command-line arguments**: Select initial agent
- **Runtime switching**: Switch agents during session
- **Exception-based flow control**: Clean agent switching without complex state management

## Testing

Run the test suite to verify agent selection works:
```bash
python test_agent_selection.py
``` 
```

--- END OF FILE docs/AGENT_SELECTION.md ---

--- START OF FILE docs/elicitations/elicitation_account_server.py ---

```py
"""
MCP Server for Account Creation Demo

This server provides an account signup form that can be triggered
by tools, demonstrating LLM-initiated elicitations.

Note: Following MCP spec, we don't collect sensitive information like passwords.
"""

import logging
import sys

from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_account_server")

# Create MCP server
mcp = FastMCP("Account Creation Server", log_level="INFO")


@mcp.tool()
async def create_user_account(service_name: str = "MyApp") -> str:
    """
    Create a new user account for the specified service.

    Args:
        service_name: The name of the service to create an account for

    Returns:
        Status message about the account creation
    """
    # This tool triggers the elicitation form
    logger.info(f"Creating account for service: {service_name}")

    class AccountSignup(BaseModel):
        username: str = Field(description="Choose a username", min_length=3, max_length=20)
        email: str = Field(description="Your email address", json_schema_extra={"format": "email"})
        full_name: str = Field(description="Your full name", max_length=30)

        language: str = Field(
            default="en",
            description="Preferred language",
            json_schema_extra={
                "enum": [
                    "en",
                    "zh",
                    "es",
                    "fr",
                    "de",
                    "ja",
                ],
                "enumNames": ["English", "中文", "Español", "Français", "Deutsch", "日本語"],
            },
        )
        agree_terms: bool = Field(description="I agree to the terms of service")
        marketing_emails: bool = Field(False, description="Send me product updates")

    result = await mcp.get_context().elicit(
        f"Create Your {service_name} Account", schema=AccountSignup
    )

    match result:
        case AcceptedElicitation(data=data):
            if not data.agree_terms:
                return "❌ Account creation failed: You must agree to the terms of service"
            else:
                return f"✅ Account created successfully for {service_name}!\nUsername: {data.username}\nEmail: {data.email}"
        case DeclinedElicitation():
            return f"❌ Account creation for {service_name} was declined by user"
        case CancelledElicitation():
            return f"❌ Account creation for {service_name} was cancelled by user"


if __name__ == "__main__":
    logger.info("Starting account creation server...")
    mcp.run()

```

--- END OF FILE docs/elicitations/elicitation_account_server.py ---

--- START OF FILE docs/elicitations/elicitation_forms_server.py ---

```py
"""
MCP Server for Basic Elicitation Forms Demo

This server provides various elicitation resources that demonstrate
different form types and validation patterns.
"""

import logging
import sys
from typing import Optional

from mcp import ReadResourceResult
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl, BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_forms_server")

# Create MCP server
mcp = FastMCP("Elicitation Forms Demo Server", log_level="INFO")


@mcp.resource(uri="elicitation://event-registration")
async def event_registration() -> ReadResourceResult:
    """Register for a tech conference event."""

    class EventRegistration(BaseModel):
        name: str = Field(description="Your full name", min_length=2, max_length=100)
        email: str = Field(description="Your email address", json_schema_extra={"format": "email"})
        company_website: Optional[str] = Field(
            None, description="Your company website (optional)", json_schema_extra={"format": "uri"}
        )
        event_date: str = Field(
            description="Which event date works for you?", json_schema_extra={"format": "date"}
        )
        dietary_requirements: Optional[str] = Field(
            None, description="Any dietary requirements? (optional)", max_length=200
        )

    result = await mcp.get_context().elicit(
        "Register for the fast-agent conference - fill out your details",
        schema=EventRegistration,
    )

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"✅ Registration confirmed for {data.name}",
                f"📧 Email: {data.email}",
                f"🏢 Company: {data.company_website or 'Not provided'}",
                f"📅 Event Date: {data.event_date}",
                f"🍽️ Dietary Requirements: {data.dietary_requirements or 'None'}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Registration declined - no ticket reserved"
        case CancelledElicitation():
            response = "Registration cancelled - please try again later"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://event-registration"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://product-review")
async def product_review() -> ReadResourceResult:
    """Submit a product review with rating and comments."""

    class ProductReview(BaseModel):
        rating: int = Field(description="Rate this product (1-5 stars)", ge=1, le=5)
        satisfaction: float = Field(
            description="Overall satisfaction score (0.0-10.0)", ge=0.0, le=10.0
        )
        category: str = Field(
            description="What type of product is this?",
            json_schema_extra={
                "enum": ["electronics", "books", "clothing", "home", "sports"],
                "enumNames": [
                    "Electronics",
                    "Books & Media",
                    "Clothing",
                    "Home & Garden",
                    "Sports & Outdoors",
                ],
            },
        )
        review_text: str = Field(
            description="Tell us about your experience", min_length=10, max_length=1000
        )

    result = await mcp.get_context().elicit(
        "Share your product review - Help others make informed decisions!", schema=ProductReview
    )

    match result:
        case AcceptedElicitation(data=data):
            stars = "⭐" * data.rating
            lines = [
                "🎯 Product Review Submitted!",
                f"⭐ Rating: {stars} ({data.rating}/5)",
                f"📊 Satisfaction: {data.satisfaction}/10.0",
                f"📦 Category: {data.category.replace('_', ' ').title()}",
                f"💬 Review: {data.review_text}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Review declined - no feedback submitted"
        case CancelledElicitation():
            response = "Review cancelled - you can submit it later"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://product-review"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://account-settings")
async def account_settings() -> ReadResourceResult:
    """Configure your account settings and preferences."""

    class AccountSettings(BaseModel):
        email_notifications: bool = Field(True, description="Receive email notifications?")
        marketing_emails: bool = Field(False, description="Subscribe to marketing emails?")
        theme: str = Field(
            description="Choose your preferred theme",
            json_schema_extra={
                "enum": ["light", "dark", "auto"],
                "enumNames": ["Light Theme", "Dark Theme", "Auto (System)"],
            },
        )
        privacy_public: bool = Field(False, description="Make your profile public?")
        items_per_page: int = Field(description="Items to show per page (10-100)", ge=10, le=100)

    result = await mcp.get_context().elicit("Update your account settings", schema=AccountSettings)

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                "⚙️ Account Settings Updated!",
                f"📧 Email notifications: {'On' if data.email_notifications else 'Off'}",
                f"📬 Marketing emails: {'On' if data.marketing_emails else 'Off'}",
                f"🎨 Theme: {data.theme.title()}",
                f"👥 Public profile: {'Yes' if data.privacy_public else 'No'}",
                f"📄 Items per page: {data.items_per_page}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Settings unchanged - keeping current preferences"
        case CancelledElicitation():
            response = "Settings update cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://account-settings"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://service-appointment")
async def service_appointment() -> ReadResourceResult:
    """Schedule a car service appointment."""

    class ServiceAppointment(BaseModel):
        customer_name: str = Field(description="Your full name", min_length=2, max_length=50)
        vehicle_type: str = Field(
            description="What type of vehicle do you have?",
            json_schema_extra={
                "enum": ["sedan", "suv", "truck", "motorcycle", "other"],
                "enumNames": ["Sedan", "SUV/Crossover", "Truck", "Motorcycle", "Other"],
            },
        )
        needs_loaner: bool = Field(description="Do you need a loaner vehicle?")
        appointment_time: str = Field(
            description="Preferred appointment date and time",
            json_schema_extra={"format": "date-time"},
        )
        priority_service: bool = Field(False, description="Is this an urgent repair?")

    result = await mcp.get_context().elicit(
        "Schedule your vehicle service appointment", schema=ServiceAppointment
    )

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                "🔧 Service Appointment Scheduled!",
                f"👤 Customer: {data.customer_name}",
                f"🚗 Vehicle: {data.vehicle_type.title()}",
                f"🚙 Loaner needed: {'Yes' if data.needs_loaner else 'No'}",
                f"📅 Appointment: {data.appointment_time}",
                f"⚡ Priority service: {'Yes' if data.priority_service else 'No'}",
            ]
            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Appointment cancelled - call us when you're ready!"
        case CancelledElicitation():
            response = "Appointment scheduling cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain",
                uri=AnyUrl("elicitation://service-appointment"),
                text=response,
            )
        ]
    )


if __name__ == "__main__":
    logger.info("Starting elicitation forms demo server...")
    mcp.run()

```

--- END OF FILE docs/elicitations/elicitation_forms_server.py ---

--- START OF FILE docs/elicitations/elicitation_game_server.py ---

```py
"""
MCP Server for Game Character Creation

This server provides a fun game character creation form
that can be used with custom handlers.
"""

import logging
import random
import sys

from mcp import ReadResourceResult
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl, BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_game_server")

# Create MCP server
mcp = FastMCP("Game Character Creation Server", log_level="INFO")


@mcp.resource(uri="elicitation://game-character")
async def game_character() -> ReadResourceResult:
    """Fun game character creation form for the whimsical example."""

    class GameCharacter(BaseModel):
        character_name: str = Field(description="Name your character", min_length=2, max_length=30)
        character_class: str = Field(
            description="Choose your class",
            json_schema_extra={
                "enum": ["warrior", "mage", "rogue", "ranger", "paladin", "bard"],
                "enumNames": [
                    "⚔️ Warrior",
                    "🔮 Mage",
                    "🗡️ Rogue",
                    "🏹 Ranger",
                    "🛡️ Paladin",
                    "🎵 Bard",
                ],
            },
        )
        strength: int = Field(description="Strength (3-18)", ge=3, le=18, default=10)
        intelligence: int = Field(description="Intelligence (3-18)", ge=3, le=18, default=10)
        dexterity: int = Field(description="Dexterity (3-18)", ge=3, le=18, default=10)
        charisma: int = Field(description="Charisma (3-18)", ge=3, le=18, default=10)
        lucky_dice: bool = Field(False, description="Roll for a lucky bonus?")

    result = await mcp.get_context().elicit("🎮 Create Your Game Character!", schema=GameCharacter)

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"🎭 Character Created: {data.character_name}",
                f"Class: {data.character_class.title()}",
                f"Stats: STR:{data.strength} INT:{data.intelligence} DEX:{data.dexterity} CHA:{data.charisma}",
            ]

            if data.lucky_dice:
                dice_roll = random.randint(1, 20)
                if dice_roll >= 15:
                    bonus = random.choice(
                        [
                            "🎁 Lucky! +2 to all stats!",
                            "🌟 Critical! Found a magic item!",
                            "💰 Jackpot! +100 gold!",
                        ]
                    )
                    lines.append(f"🎲 Dice Roll: {dice_roll} - {bonus}")
                else:
                    lines.append(f"🎲 Dice Roll: {dice_roll} - No bonus this time!")

            total_stats = data.strength + data.intelligence + data.dexterity + data.charisma
            if total_stats > 50:
                lines.append("💪 Powerful character build!")
            elif total_stats < 30:
                lines.append("🎯 Challenging build - good luck!")

            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Character creation declined - returning to menu"
        case CancelledElicitation():
            response = "Character creation cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://game-character"), text=response
            )
        ]
    )


@mcp.tool()
async def roll_new_character(campaign_name: str = "Adventure") -> str:
    """
    Roll a new character for your campaign.

    Args:
        campaign_name: The name of the campaign

    Returns:
        Character details or status message
    """

    class GameCharacter(BaseModel):
        character_name: str = Field(description="Name your character", min_length=2, max_length=30)
        character_class: str = Field(
            description="Choose your class",
            json_schema_extra={
                "enum": ["warrior", "mage", "rogue", "ranger", "paladin", "bard"],
                "enumNames": [
                    "⚔️ Warrior",
                    "🔮 Mage",
                    "🗡️ Rogue",
                    "🏹 Ranger",
                    "🛡️ Paladin",
                    "🎵 Bard",
                ],
            },
        )
        strength: int = Field(description="Strength (3-18)", ge=3, le=18, default=10)
        intelligence: int = Field(description="Intelligence (3-18)", ge=3, le=18, default=10)
        dexterity: int = Field(description="Dexterity (3-18)", ge=3, le=18, default=10)
        charisma: int = Field(description="Charisma (3-18)", ge=3, le=18, default=10)
        lucky_dice: bool = Field(False, description="Roll for a lucky bonus?")

    result = await mcp.get_context().elicit(
        f"🎮 Create Character for {campaign_name}!", schema=GameCharacter
    )

    match result:
        case AcceptedElicitation(data=data):
            response = f"🎭 {data.character_name} the {data.character_class.title()} joins {campaign_name}!\n"
            response += f"Stats: STR:{data.strength} INT:{data.intelligence} DEX:{data.dexterity} CHA:{data.charisma}"

            if data.lucky_dice:
                dice_roll = random.randint(1, 20)
                if dice_roll >= 15:
                    response += f"\n🎲 Lucky roll ({dice_roll})! Starting with bonus equipment!"
                else:
                    response += f"\n🎲 Rolled {dice_roll} - Standard starting gear."

            return response
        case DeclinedElicitation():
            return f"Character creation for {campaign_name} was declined"
        case CancelledElicitation():
            return f"Character creation for {campaign_name} was cancelled"


if __name__ == "__main__":
    logger.info("Starting game character creation server...")
    mcp.run()

```

--- END OF FILE docs/elicitations/elicitation_game_server.py ---

--- START OF FILE docs/elicitations/fastagent.config.yaml ---

```yaml
# Model string takes format:
#   <provider>.<model_string>.<reasoning_effort?> (e.g. anthropic.claude-3-5-sonnet-20241022 or openai.o3-mini.low)
#
# Can be overriden with a command line switch --model=<model>, or within the Agent decorator.
# Check here for current details: https://fast-agent.ai/models/
default_model: "passthrough"

# Logging and Console Configuration
logger:
  level: "error"
  type: "console"

# MCP Server Configuration
mcp:
  servers:
    # Forms demo server - interactive form examples
    elicitation_forms_server:
      command: "uv"
      args: ["run", "elicitation_forms_server.py"]
      elicitation:
        mode: "forms" # Shows forms to users (default)

    # Account creation server - for CALL_TOOL demos
    elicitation_account_server:
      command: "uv"
      args: ["run", "elicitation_account_server.py"]
      elicitation:
        mode: "forms"

    # Game character server - for custom handler demos
    elicitation_game_server:
      command: "uv"
      args: ["run", "elicitation_game_server.py"]
      elicitation:
        mode: "forms"

```

--- END OF FILE docs/elicitations/fastagent.config.yaml ---

--- START OF FILE docs/elicitations/fastagent.secrets.yaml.example ---

```example
# Secrets configuration for elicitation examples
#
# Rename this file to fastagent.secrets.yaml and add your API keys
# to use the account_creation.py example with real LLMs

# OpenAI
openai_api_key: "sk-..."

# Anthropic
anthropic_api_key: "sk-ant-..."

# Google (Gemini)
google_api_key: "..."

# Other providers - see documentation for full list
# groq_api_key: "..."
# mistral_api_key: "..."
```

--- END OF FILE docs/elicitations/fastagent.secrets.yaml.example ---

--- START OF FILE docs/elicitations/forms_demo.py ---

```py
"""
Quick Start: Elicitation Forms Demo

This example demonstrates the elicitation forms feature of fast-agent.

When Read Resource requests are sent to the MCP Server, it generates an Elicitation
which creates a form for the user to fill out.
The results are returned to the demo program which prints out the results in a rich format.
"""

import asyncio

from rich.console import Console
from rich.panel import Panel

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.helpers.content_helpers import get_resource_text

fast = FastAgent("Elicitation Forms Demo", quiet=True)
console = Console()


@fast.agent(
    "forms-demo",
    servers=[
        "elicitation_forms_server",
    ],
)
async def main():
    """Run the improved forms demo showcasing all elicitation features."""
    async with fast.run() as agent:
        console.print("\n[bold cyan]Welcome to the Elicitation Forms Demo![/bold cyan]\n")
        console.print("This demo shows how to collect structured data using MCP Elicitations.")
        console.print("We'll present several forms and display the results collected for each.\n")

        # Example 1: Event Registration
        console.print("[bold yellow]Example 1: Event Registration Form[/bold yellow]")
        console.print(
            "[dim]Demonstrates: string validation, email format, URL format, date format[/dim]"
        )
        result = await agent.get_resource("elicitation://event-registration")

        if result_text := get_resource_text(result):
            panel = Panel(
                result_text,
                title="🎫 Registration Confirmation",
                border_style="green",
                expand=False,
            )
            console.print(panel)
        else:
            console.print("[red]No registration data received[/red]")

        console.print("\n" + "─" * 50 + "\n")

        # Example 2: Product Review
        console.print("[bold yellow]Example 2: Product Review Form[/bold yellow]")
        console.print(
            "[dim]Demonstrates: number validation (range), radio selection, multiline text[/dim]"
        )
        result = await agent.get_resource("elicitation://product-review")

        if result_text := get_resource_text(result):
            review_panel = Panel(
                result_text, title="🛍️ Product Review", border_style="cyan", expand=False
            )
            console.print(review_panel)

        console.print("\n" + "─" * 50 + "\n")

        # Example 3: Account Settings
        console.print("[bold yellow]Example 3: Account Settings Form[/bold yellow]")
        console.print(
            "[dim]Demonstrates: boolean selections, radio selection, number validation[/dim]"
        )
        result = await agent.get_resource("elicitation://account-settings")

        if result_text := get_resource_text(result):
            settings_panel = Panel(
                result_text, title="⚙️ Account Settings", border_style="blue", expand=False
            )
            console.print(settings_panel)

        console.print("\n" + "─" * 50 + "\n")

        # Example 4: Service Appointment
        console.print("[bold yellow]Example 4: Service Appointment Booking[/bold yellow]")
        console.print(
            "[dim]Demonstrates: string validation, radio selection, boolean, datetime format[/dim]"
        )
        result = await agent.get_resource("elicitation://service-appointment")

        if result_text := get_resource_text(result):
            appointment_panel = Panel(
                result_text, title="🔧 Appointment Confirmed", border_style="magenta", expand=False
            )
            console.print(appointment_panel)

        console.print("\n[bold green]✅ Demo Complete![/bold green]")
        console.print("\n[bold cyan]Features Demonstrated:[/bold cyan]")
        console.print("• [green]String validation[/green] (min/max length)")
        console.print("• [green]Number validation[/green] (range constraints)")
        console.print("• [green]Radio selections[/green] (enum dropdowns)")
        console.print("• [green]Boolean selections[/green] (checkboxes)")
        console.print("• [green]Format validation[/green] (email, URL, date, datetime)")
        console.print("• [green]Multiline text[/green] (expandable text areas)")
        console.print("\nThese forms demonstrate natural, user-friendly data collection patterns!")


if __name__ == "__main__":
    asyncio.run(main())

```

--- END OF FILE docs/elicitations/forms_demo.py ---

--- START OF FILE docs/elicitations/game_character.py ---

```py
#!/usr/bin/env python3
"""
Demonstration of Custom Elicitation Handler

This example demonstrates a custom elicitation handler that creates
an interactive game character creation experience with dice rolls,
visual gauges, and fun interactions.
"""

import asyncio

# Import our custom handler from the separate module
from game_character_handler import game_character_elicitation_handler
from rich.console import Console
from rich.panel import Panel

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.helpers.content_helpers import get_resource_text

fast = FastAgent("Game Character Creator", quiet=True)
console = Console()


@fast.agent(
    "character-creator",
    servers=["elicitation_game_server"],
    # Register our handler from game_character_handler.py
    elicitation_handler=game_character_elicitation_handler,
)
async def main():
    """Run the game character creator with custom elicitation handler."""
    async with fast.run() as agent:
        console.print(
            Panel(
                "[bold cyan]Welcome to the Character Creation Studio![/bold cyan]\n\n"
                "Create your hero with our magical character generator.\n"
                "Watch as the cosmic dice determine your fate!",
                title="🎮 Game Time 🎮",
                border_style="magenta",
            )
        )

        # Trigger the character creation
        result = await agent.get_resource("elicitation://game-character")

        if result_text := get_resource_text(result):
            character_panel = Panel(
                result_text, title="📜 Your Character 📜", border_style="green", expand=False
            )
            console.print(character_panel)

            console.print("\n[italic]Your character is ready for adventure![/italic]")
            console.print("[dim]The tavern door opens, and your journey begins...[/dim]\n")

            # Fun ending based on character
            if "Powerful character" in result_text:
                console.print("⚔️  [bold]The realm trembles at your might![/bold]")
            elif "Challenging build" in result_text:
                console.print("🎯 [bold]True heroes are forged through adversity![/bold]")
            else:
                console.print("🗡️  [bold]Your legend begins now![/bold]")


if __name__ == "__main__":
    asyncio.run(main())

```

--- END OF FILE docs/elicitations/game_character.py ---

--- START OF FILE docs/elicitations/game_character_handler.py ---

```py
"""
Custom Elicitation Handler for Game Character Creation

This module provides a whimsical custom elicitation handler that creates
an interactive game character creation experience with dice rolls,
visual gauges, and animated effects.
"""

import asyncio
import random
from typing import TYPE_CHECKING, Any, Dict

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)
console = Console()


async def game_character_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Custom handler that creates an interactive character creation experience."""
    logger.info(f"Game character elicitation handler called: {params.message}")

    if params.requestedSchema:
        properties = params.requestedSchema.get("properties", {})
        content: Dict[str, Any] = {}

        console.print("\n[bold magenta]🎮 Character Creation Studio 🎮[/bold magenta]\n")

        # Character name with typewriter effect
        if "character_name" in properties:
            console.print("[cyan]✨ Generating your character's name...[/cyan] ", end="")
            name_prefixes = ["Hero", "Legend", "Epic", "Mighty", "Brave", "Noble"]
            name_suffixes = ["blade", "heart", "storm", "fire", "shadow", "star"]

            name = f"{random.choice(name_prefixes)}{random.choice(name_suffixes)}{random.randint(1, 999)}"

            for char in name:
                console.print(char, end="", style="bold green")
                await asyncio.sleep(0.03)
            console.print("\n")
            content["character_name"] = name

        # Class selection with visual menu and fate dice
        if "character_class" in properties:
            class_enum = properties["character_class"].get("enum", [])
            class_names = properties["character_class"].get("enumNames", class_enum)

            table = Table(title="🎯 Choose Your Destiny", show_header=False, box=None)
            table.add_column("Option", style="cyan", width=8)
            table.add_column("Class", style="yellow", width=20)
            table.add_column("Description", style="dim", width=30)

            descriptions = [
                "Master of sword and shield",
                "Wielder of arcane mysteries",
                "Silent shadow striker",
                "Nature's deadly archer",
                "Holy warrior of light",
                "Inspiring magical performer",
            ]

            for i, (cls, name, desc) in enumerate(zip(class_enum, class_names, descriptions)):
                table.add_row(f"[{i + 1}]", name, desc)

            console.print(table)

            # Dramatic fate dice roll
            console.print("\n[bold yellow]🎲 The Fates decide your path...[/bold yellow]")
            for _ in range(8):
                dice_face = random.choice(["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"])
                console.print(f"\r  Rolling... {dice_face}", end="")
                await asyncio.sleep(0.2)

            fate_roll = random.randint(1, 6)
            selected_idx = (fate_roll - 1) % len(class_enum)
            console.print(f"\n  🎲 Fate dice: [bold red]{fate_roll}[/bold red]!")
            console.print(
                f"✨ Destiny has chosen: [bold yellow]{class_names[selected_idx]}[/bold yellow]!\n"
            )
            content["character_class"] = class_enum[selected_idx]

        # Stats rolling with animated progress bars and cosmic effects
        stat_names = ["strength", "intelligence", "dexterity", "charisma"]
        stats_info = {
            "strength": {"emoji": "💪", "desc": "Physical power"},
            "intelligence": {"emoji": "🧠", "desc": "Mental acuity"},
            "dexterity": {"emoji": "🏃", "desc": "Agility & speed"},
            "charisma": {"emoji": "✨", "desc": "Personal magnetism"},
        }

        console.print("[bold]🌟 Rolling cosmic dice for your abilities...[/bold]\n")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25, style="cyan", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            for stat in stat_names:
                if stat in properties:
                    # Roll 3d6 for classic D&D feel with bonus potential
                    rolls = [random.randint(1, 6) for _ in range(3)]
                    total = sum(rolls)

                    # Add cosmic bonus chance
                    if random.random() < 0.15:  # 15% chance for cosmic boost
                        cosmic_bonus = random.randint(1, 3)
                        total = min(18, total + cosmic_bonus)
                        cosmic_text = f" ✨+{cosmic_bonus} COSMIC✨"
                    else:
                        cosmic_text = ""

                    stat_info = stats_info.get(stat, {"emoji": "📊", "desc": stat.title()})
                    task = progress.add_task(
                        f"{stat_info['emoji']} {stat.capitalize()}: {stat_info['desc']}", total=18
                    )

                    # Animate the progress bar with suspense
                    for i in range(total + 1):
                        progress.update(task, completed=i)
                        await asyncio.sleep(0.04)

                    content[stat] = total
                    console.print(
                        f"   🎲 Rolled: {rolls} = [bold green]{total}[/bold green]{cosmic_text}"
                    )

        # Lucky dice legendary challenge
        if "lucky_dice" in properties:
            console.print("\n" + "=" * 60)
            console.print("[bold yellow]🎰 LEGENDARY CHALLENGE: Lucky Dice! 🎰[/bold yellow]")
            console.print("The ancient dice of fortune whisper your name...")
            console.print("Do you dare tempt fate for legendary power?")
            console.print("=" * 60)

            # Epic dice rolling sequence
            console.print("\n[cyan]🌟 Rolling the Dice of Destiny...[/cyan]")

            for i in range(15):
                dice_faces = ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"]
                d20_faces = ["🎲"] * 19 + ["💎"]  # Special diamond for 20

                if i < 10:
                    face = random.choice(dice_faces)
                else:
                    face = random.choice(d20_faces)

                console.print(f"\r  [bold]{face}[/bold] Rolling...", end="")
                await asyncio.sleep(0.15)

            final_roll = random.randint(1, 20)

            if final_roll == 20:
                console.print("\r  [bold red]💎 NATURAL 20! 💎[/bold red]")
                console.print("  [bold green]🌟 LEGENDARY SUCCESS! 🌟[/bold green]")
                console.print("  [gold1]You have been blessed by the gods themselves![/gold1]")
                bonus_text = "🏆 Divine Champion status unlocked!"
            elif final_roll >= 18:
                console.print(f"\r  [bold yellow]⭐ {final_roll} - EPIC ROLL! ⭐[/bold yellow]")
                bonus_text = "🎁 Epic treasure discovered!"
            elif final_roll >= 15:
                console.print(f"\r  [green]🎲 {final_roll} - Great success![/green]")
                bonus_text = "🌟 Rare magical item found!"
            elif final_roll >= 10:
                console.print(f"\r  [yellow]🎲 {final_roll} - Good fortune.[/yellow]")
                bonus_text = "🗡️ Modest blessing received."
            elif final_roll == 1:
                console.print("\r  [bold red]💀 CRITICAL FUMBLE! 💀[/bold red]")
                bonus_text = "😅 Learning experience gained... try again!"
            else:
                console.print(f"\r  [dim]🎲 {final_roll} - The dice are silent.[/dim]")
                bonus_text = "🎯 Your destiny remains unwritten."

            console.print(f"  [italic]{bonus_text}[/italic]")
            content["lucky_dice"] = final_roll >= 10

        # Epic character summary with theatrical flair
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]📜 Your Character Has Been Rolled! 📜[/bold cyan]")
        console.print("=" * 70)

        # Show character summary
        total_stats = sum(content.get(stat, 10) for stat in stat_names if stat in content)

        # Create a simple table
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Label", style="cyan", width=15)
        stats_table.add_column("Value", style="bold white")

        if "character_name" in content:
            stats_table.add_row("Name:", content["character_name"])
        if "character_class" in content:
            class_idx = class_enum.index(content["character_class"])
            stats_table.add_row("Class:", class_names[class_idx])

        stats_table.add_row("", "")  # Empty row for spacing

        # Add stats
        for stat in stat_names:
            if stat in content:
                stat_label = f"{stat.capitalize()}:"
                stats_table.add_row(stat_label, str(content[stat]))

        stats_table.add_row("", "")
        stats_table.add_row("Total Power:", str(total_stats))

        console.print(stats_table)

        # Power message
        if total_stats > 60:
            console.print("✨ [bold gold1]The realm trembles before your might![/bold gold1] ✨")
        elif total_stats > 50:
            console.print("⚔️ [bold green]A formidable hero rises![/bold green] ⚔️")
        elif total_stats < 35:
            console.print("🎯 [bold blue]The underdog's tale begins![/bold blue] 🎯")
        else:
            console.print("🗡️ [bold white]Adventure awaits the worthy![/bold white] 🗡️")

        # Ask for confirmation
        console.print("\n[bold yellow]Do you accept this character?[/bold yellow]")
        console.print("[dim]Press Enter to accept, 'n' to decline, or Ctrl+C to cancel[/dim]\n")

        try:
            accepted = Confirm.ask("Accept character?", default=True)

            if accepted:
                console.print(
                    "\n[bold green]✅ Character accepted! Your adventure begins![/bold green]"
                )
                return ElicitResult(action="accept", content=content)
            else:
                console.print(
                    "\n[yellow]❌ Character declined. The fates will roll again...[/yellow]"
                )
                return ElicitResult(action="decline")
        except KeyboardInterrupt:
            console.print("\n[red]❌ Character creation cancelled![/red]")
            return ElicitResult(action="cancel")

    else:
        # No schema, return a fun message
        content = {"response": "⚔️ Ready for adventure! ⚔️"}
        return ElicitResult(action="accept", content=content)

```

--- END OF FILE docs/elicitations/game_character_handler.py ---

--- START OF FILE docs/elicitations/tool_call.py ---

```py
import asyncio

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("fast-agent example")


# Define the agent
@fast.agent(
    instruction="You are a helpful AI Agent",
    servers=["elicitation_account_server"],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.send('***CALL_TOOL create_user_account {"service_name": "fast-agent"}')


if __name__ == "__main__":
    asyncio.run(main())

```

--- END OF FILE docs/elicitations/tool_call.py ---

--- START OF FILE docs/README.md ---

```md
# Agent Dashboard

A terminal client for the `fast-agent` framework.

This project started as a way to have a more stable and transparent interface for agent development. The core is a Model-View-Controller (MVC) architecture, separating the application's state from its terminal UI and logic.

## Technical Details

The client is built with a few key ideas in mind:

*   **Context Management.** Following the philosophy of the Model Context Protocol, the controller assembles the conversational history and other data to form the precise context sent to the agent on each turn. This allows for more deliberate, developer-driven context strategies.

*   **Asynchronous Core.** The application uses `asyncio` and a non-blocking prompt, which keeps the UI responsive. It's designed to support more complex operations, like parallel agent interactions, and could be adapted for a GUI dashboard later.

*   **Stateful History.** While the terminal shows a clean chat log, a comprehensive history is maintained in the background. This history can be saved automatically or manually, providing a useful artifact for debugging or resuming sessions.

*   **Resilient Operation.** LLM or MCP server errors are handled by the controller, which rolls back the conversational state to its last valid point. The application also shuts down cleanly to avoid resource errors.

*   **Comprehensive Testing.** The application includes a complete testing suite with unit tests, integration tests, and retry mechanisms to ensure reliability and maintainability.

## Testing

The project includes a comprehensive testing suite to ensure reliability and maintainability:

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py test_model.py

# Run with verbose output
python run_tests.py -v
```

### Test Structure

- **`test_model.py`**: Unit tests for the Model class, covering state management, conversation history, and file operations
- **`test_controller.py`**: Unit tests for the Controller class, including command parsing and agent interaction with retry logic
- **`test_integration.py`**: Integration tests that verify the interaction between Model and Controller components

### Test Features

- **Retry Logic**: The controller now includes exponential backoff retry logic for agent calls, making the application more resilient to temporary network or API issues
- **Mock Testing**: All tests use mocks to avoid external dependencies while thoroughly testing the application logic
- **Async Support**: Full async/await support for testing the asynchronous nature of the application

## Project Journey

This client evolved through several stages:

1.  Began with simple `fast-agent` scripts run from the command line.
2.  Integrated a few powerful MCP servers (`filesystem`, `memory`, `fetch`), which revealed the potential of the protocol.
3.  Shifted focus from thinking of `fast-agent` as a script runner to using it as a library within a client/server model.
4.  Adopted the MVC pattern to cleanly separate concerns.
5.  The result is this application—a stable tool for further agent development.

```

--- END OF FILE docs/README.md ---

--- START OF FILE fastagent.config.yaml ---

```yaml
# fastagent.config.yaml

# --- Model Configuration ---
# Set the default model for all agents.
# You can override this per-agent in the decorator or with the --model CLI flag.
# Format: <provider>.<model_name> (e.g., openai.gpt-4o, anthropic.claude-3-5-sonnet-latest)
# Aliases like 'sonnet' or 'haiku' are also supported.
default_model: google.gemini-2.5-flash

# --- Logger Configuration ---
# This setup gives your client script full control over what is displayed.
logger:
  # Hide the default progress bar for a cleaner terminal experience.
  progress_display: false
  # We will print messages from our client script, so disable the default chat log.
  show_chat: false
  # We will handle tool display in our client script, so disable this too.
  show_tools: false

# --- MCP Server Configuration ---
# Defines the external tools and services available to your agents.
mcp:
  servers:
    # Fetch server for web scraping and data retrieval
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
    
    # Filesystem server for reading/writing local files
    filesystem:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-filesystem"
        - "G:/My Drive/AI Resources/Open collection"

    # Secure filesystem server for read-only access to specific directories
    secure-filesystem:
      command: "uv"
      args: ["run", "secure_filesystem_server.py", "G:/My Drive/AI Resources/Open collection"]

    # Memory server for persistent knowledge graph memory
    memory:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-memory"

    # Sequential Thinking server for dynamic and reflective problem-solving
    sequential-thinking:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-sequential-thinking"
```

--- END OF FILE fastagent.config.yaml ---

--- START OF FILE main.py ---

```py
# main.py
import asyncio
import sys
import argparse

from model import Model
from view import View
from controller import Controller, ExitCommand, SwitchAgentCommand
from agent_definitions import get_agent, list_available_agents

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

def parse_arguments():
    """Parse command line arguments for agent selection."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default="minimal",
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

async def run_agent_session(agent_name: str):
    """
    Run a session with a specific agent.
    
    Args:
        agent_name: The name of the agent to run
        
    Returns:
        The new agent name if switching, None if exiting
    """
    try:
        # Get the selected agent from the registry
        selected_agent = get_agent(agent_name)
        print(f"Starting {agent_name} agent...")
        
        # Run the selected agent
        async with selected_agent.run() as agent_app:
            # Initialize MVC components
            model = Model()
            controller = Controller(model, agent_app)
            view = View(model, controller)
            
            # Run the main loop until exit or switch
            await view.run_main_loop()
            return None  # Normal exit
            
    except SwitchAgentCommand as e:
        return e.agent_name  # Switch to new agent
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available agents: {', '.join(list_available_agents())}")
        return None

async def main():
    """
    The main entry point for the application.
    """
    # Parse command line arguments
    args = parse_arguments()
    current_agent = args.agent
    
    # Main agent loop - handles switching between agents
    while current_agent is not None:
        current_agent = await run_agent_session(current_agent)
        if current_agent:
            print(f"\nSwitching to {current_agent} agent...")
            # Small delay to show the switch message
            await asyncio.sleep(0.5)

    # This delay happens AFTER all agents have closed, giving background
    # tasks time to finalize their shutdown before the script terminates.
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # We no longer catch SystemExit here, but keep it for robustness.
        pass
    finally:
        # The final message is printed after everything has shut down.
        print_shutdown_message()
```

--- END OF FILE main.py ---

--- START OF FILE model.py ---

```py
# model.py
import asyncio
import json
import os
from datetime import datetime
from enum import Enum, auto
from typing import Callable, List, Optional

# Core types from the fast-agent framework.
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

class AppState(Enum):
    """Defines the possible states of the client application."""
    IDLE = auto()
    AGENT_IS_THINKING = auto()
    WAITING_FOR_USER_INPUT = auto()
    ERROR = auto()

class Model:
    """
    The Model represents the single source of truth for the application's state.
    It holds all data and notifies listeners when its state changes. It contains
    no business logic and is entirely passive.
    """
    def __init__(self):
        # --- State Data ---
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conversation_history: List[PromptMessageMultipart] = []
        self.application_state: AppState = AppState.IDLE
        self.last_error_message: Optional[str] = None
        self.last_success_message: Optional[str] = None
        
        # Corrected initialization sequence:
        # 1. Initialize the dictionary with static keys first.
        self.user_preferences: dict = {
            "auto_save_enabled": True,
            "context_dir": "_context",
        }
        # 2. Now that self.user_preferences exists, we can safely use its
        #    values to construct and add the dynamic key.
        self.user_preferences["auto_save_filename"] = f"{self._get_context_dir()}/{self.session_id}.json"

        # --- Notification System ---
        self._listeners: List[Callable] = []

    def _get_context_dir(self) -> str:
        """Helper to access the context directory from preferences."""
        return self.user_preferences.get("context_dir", "_context")

    async def _notify_listeners(self):
        """Asynchronously notify all registered listeners of a state change."""
        for listener in self._listeners:
            await listener()

    def register_listener(self, listener: Callable):
        """
        Allows other components (like the View) to register a callback
        to be notified of state changes.
        """
        self._listeners.append(listener)

    # --- Methods to Mutate State (Instructed by the Controller) ---

    async def add_message(self, message: PromptMessageMultipart):
        """Appends a new message to the conversation history."""
        self.conversation_history.append(message)
        await self._notify_listeners()

    async def pop_last_message(self) -> Optional[PromptMessageMultipart]:
        """
        Removes and returns the last message from the history.
        Crucial for rolling back state on agent failure.
        """
        if not self.conversation_history:
            return None
        last_message = self.conversation_history.pop()
        await self._notify_listeners()
        return last_message

    async def clear_history(self):
        """Clears the entire conversation history."""
        self.conversation_history = []
        await self._notify_listeners()

    async def set_state(self, new_state: AppState, error_message: Optional[str] = None, success_message: Optional[str] = None):
        """Updates the application's current state and notifies listeners."""
        self.application_state = new_state
        if new_state == AppState.ERROR:
            self.last_error_message = error_message
            self.last_success_message = None
        else:
            self.last_error_message = None # Clear error on non-error states.
            self.last_success_message = success_message
        await self._notify_listeners()

    async def load_history_from_file(self, filepath: str) -> bool:
        """
        Loads conversation history from a JSON file, replacing the current history.
        Returns True on success, False on failure.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)
            # Re-create the rich PromptMessageMultipart objects from the raw dicts.
            self.conversation_history = [
                PromptMessageMultipart(**data) for data in raw_history
            ]
            await self._notify_listeners()
            return True
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            # We don't change state on failure, just report it.
            await self.set_state(AppState.ERROR, f"Failed to load history: {e}")
            return False

    # --- Methods for Actions (Instructed by the Controller) ---

    async def save_history_to_file(self, filepath: Optional[str] = None) -> bool:
        """
        Saves the current conversation history to a specified JSON file.
        This method does not mutate the model's state.
        Returns True on success, False on failure.
        """
        target_filepath = filepath or self.user_preferences["auto_save_filename"]
        context_dir = self._get_context_dir()
        os.makedirs(context_dir, exist_ok=True)

        try:
            serializable_history = [
                message.model_dump(mode='json') for message in self.conversation_history
            ]
            with open(target_filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            # If saving fails, we set an error state to inform the user.
            await self.set_state(AppState.ERROR, f"Could not write to file {target_filepath}")
            return False
```

--- END OF FILE model.py ---

--- START OF FILE pyproject.toml ---

```toml
[project]
name = "agent-dashboard"
version = "0.1.0"
description = "A terminal-based agent dashboard for MCP agents"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.53.0",
    "mcp[cli]>=1.9.3",
    "python-dotenv>=1.1.0",
    "rich>=14.0.0",
    "prompt_toolkit>=3.0.0",
    "fast-agent-mcp>=0.2.40",
    "multidict>=6.5.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

```

--- END OF FILE pyproject.toml ---

--- START OF FILE secure_filesystem_server.py ---

```py
# secure_filesystem_server.py
# Needs to be validated; not sure this is the correct implementation.
import os
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP
import typer

# Initialize the FastMCP server
mcp = FastMCP("secure-filesystem")

def is_path_safe(base_dirs: List[Path], target_path: Path) -> bool:
    """Ensure the target path is within one of the allowed base directories."""
    resolved_path = target_path.resolve()
    for base in base_dirs:
        try:
            resolved_path.relative_to(base.resolve())
            return True
        except ValueError:
            continue
    return False

@mcp.tool()
def read_file(path: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Reads the complete contents of a single file."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_file():
        return f"Error: Path is not a file: {path}"
    return target_path.read_text(encoding="utf-8")

@mcp.tool()
def list_directory(path: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Lists the contents of a directory."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"
    
    contents = []
    for item in target_path.iterdir():
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        contents.append(f"{prefix} {item.name}")
    return "\n".join(contents)

@mcp.tool()
def search_files(path: str, pattern: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Recursively searches for files matching a pattern in a directory."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"

    matches = [str(p) for p in target_path.rglob(pattern)]
    return "\n".join(matches) if matches else "No matching files found."


@mcp.tool()
def list_allowed_directories(allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Lists all directories the server is allowed to access."""
    return "This server has read-only access to the following directories:\n" + "\n".join([str(d.resolve()) for d in allowed_dirs])


# The `run` function provided by FastMCP will automatically handle CLI arguments.
# Any arguments defined here (like `allowed_dirs`) that are not tools themselves
# will be passed to all tool functions that require them.
@mcp.run(transport="stdio")
def main(allowed_dirs: List[Path] = typer.Argument(..., help="List of directories to allow read access to.")):
    """
    A read-only filesystem MCP server.
    This server will run until the client disconnects.
    """
    # The typer decorator and mcp.run handle the server lifecycle.
    pass
```

--- END OF FILE secure_filesystem_server.py ---

--- START OF FILE tests/run_tests.py ---

```py
#!/usr/bin/env python3
"""
Simple test runner for the agent-dashboard project.
Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_model.py     # Run specific test file
    python run_tests.py -v                # Run with verbose output
"""

import sys
import subprocess
import os


def run_tests(test_file=None, verbose=False):
    """Run pytest with the specified options."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_file:
        cmd.append(test_file)
    else:
        # Run all test files
        cmd.extend(["test_model.py", "test_controller.py", "test_integration.py"])
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-asyncio")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for agent-dashboard")
    parser.add_argument("test_file", nargs="?", help="Specific test file to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🧪 Running tests for agent-dashboard...")
    success = run_tests(args.test_file, args.verbose)
    
    sys.exit(0 if success else 1) 
```

--- END OF FILE tests/run_tests.py ---

--- START OF FILE tests/test_agent_selection.py ---

```py
#!/usr/bin/env python3
"""
Test script for agent selection functionality.
"""

import asyncio
from agent_definitions import get_agent, list_available_agents, AGENT_REGISTRY

def test_agent_registry():
    """Test the agent registry functionality."""
    print("Testing Agent Registry...")
    
    # Test listing available agents
    available_agents = list_available_agents()
    print(f"Available agents: {available_agents}")
    assert len(available_agents) >= 2, "Should have at least 2 agents"
    
    # Test getting valid agents
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")
    print("✓ Successfully retrieved minimal and coding agents")
    
    # Test getting invalid agent
    try:
        get_agent("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✓ Correctly raised KeyError for invalid agent: {e}")
    
    print("All agent registry tests passed!")

def test_agent_characteristics():
    """Test that agents have different characteristics."""
    print("\nTesting Agent Characteristics...")
    
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")
    
    # Check that they're different instances
    assert minimal_agent != coding_agent, "Agents should be different instances"
    
    # Check that they have different names
    assert minimal_agent.name != coding_agent.name, "Agents should have different names"
    
    print("✓ Agents have different characteristics")
    print(f"  Minimal agent: {minimal_agent.name}")
    print(f"  Coding agent: {coding_agent.name}")
    
    print("All agent characteristics tests passed!")

if __name__ == "__main__":
    test_agent_registry()
    test_agent_characteristics()
    print("\n🎉 All tests passed! Agent selection system is working correctly.") 
```

--- END OF FILE tests/test_agent_selection.py ---

--- START OF FILE tests/test_controller.py ---

```py
import pytest
from unittest.mock import AsyncMock, MagicMock
from controller import Controller, ExitCommand
from model import Model, AppState
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_exit_command():
    """Test that the exit command raises ExitCommand exception."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    with pytest.raises(ExitCommand):
        await controller.process_user_input("/exit")

    with pytest.raises(ExitCommand):
        await controller.process_user_input("/quit")


@pytest.mark.asyncio
async def test_save_command():
    """Test the save command functionality."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    # Test save with default filename
    await controller.process_user_input("/save")
    mock_model.save_history_to_file.assert_called_once_with(None)

    # Test save with custom filename
    await controller.process_user_input("/save test_file.json")
    mock_model.save_history_to_file.assert_called_with("test_file.json")


@pytest.mark.asyncio
async def test_load_command():
    """Test the load command functionality."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    # Test load with filename
    await controller.process_user_input("/load test_file.json")
    mock_model.load_history_from_file.assert_called_once_with("test_file.json")


@pytest.mark.asyncio
async def test_clear_command():
    """Test the clear command functionality."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("/clear")
    mock_model.clear_history.assert_called_once()
    mock_model.set_state.assert_called_with(AppState.IDLE, success_message="Conversation history cleared.")


@pytest.mark.asyncio
async def test_unknown_command():
    """Test handling of unknown commands."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("/unknown")
    mock_model.set_state.assert_called_with(AppState.ERROR, error_message="Unknown command: /unknown")


@pytest.mark.asyncio
async def test_empty_input():
    """Test that empty input is handled gracefully."""
    mock_model = AsyncMock()
    mock_agent_app = MagicMock()
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("")
    await controller.process_user_input("   ")
    
    # Should not call any agent methods
    mock_agent_app.agent.generate.assert_not_called()


@pytest.mark.asyncio
async def test_successful_agent_prompt():
    """Test successful agent prompt handling."""
    mock_model = AsyncMock()
    mock_model.conversation_history = []
    mock_model.user_preferences = {"auto_save_enabled": False}
    
    mock_agent = AsyncMock()
    mock_response = MagicMock()
    mock_response.role = 'assistant'
    mock_response.content = [{'type': 'text', 'text': 'Mocked response'}]
    mock_agent.generate.return_value = mock_response
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    # Verify the flow
    mock_model.set_state.assert_called_with(AppState.AGENT_IS_THINKING)
    mock_model.add_message.assert_called()
    mock_agent.generate.assert_called_once()
    mock_model.set_state.assert_called_with(AppState.IDLE)


@pytest.mark.asyncio
async def test_agent_prompt_with_retry():
    """Test agent prompt handling with retry logic."""
    mock_model = AsyncMock()
    mock_model.conversation_history = []
    mock_model.user_preferences = {"auto_save_enabled": False}
    
    mock_agent = AsyncMock()
    # First call fails, second call succeeds
    mock_agent.generate.side_effect = [Exception("Network error"), MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Success'}])]
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    # Should have been called twice (retry)
    assert mock_agent.generate.call_count == 2
    # Should have set error state during retry
    mock_model.set_state.assert_any_call(AppState.ERROR, error_message=pytest.approx("Agent Error (attempt 1/3): Network error. Retrying in", rel=0.1))


@pytest.mark.asyncio
async def test_agent_prompt_final_failure():
    """Test agent prompt handling when all retries fail."""
    mock_model = AsyncMock()
    mock_model.conversation_history = []
    mock_model.user_preferences = {"auto_save_enabled": False}
    
    mock_agent = AsyncMock()
    # All calls fail
    mock_agent.generate.side_effect = Exception("Persistent error")
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(mock_model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    # Should have been called 3 times (max retries)
    assert mock_agent.generate.call_count == 3
    # Should have rolled back the user message
    mock_model.pop_last_message.assert_called_once()
    # Should have set final error state
    mock_model.set_state.assert_any_call(AppState.ERROR, error_message="Agent Error after 3 attempts: Persistent error") 
```

--- END OF FILE tests/test_controller.py ---

--- START OF FILE tests/test_integration.py ---

```py
import pytest
from unittest.mock import AsyncMock, MagicMock
from model import Model, AppState
from controller import Controller
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_prompt_handling_integration():
    """Test the full integration between Model and Controller for prompt handling."""
    model = Model()
    
    # Mock the agent_app and the agent's generate method
    mock_agent = AsyncMock()
    mock_response = MagicMock()
    mock_response.role = 'assistant'
    mock_response.content = [{'type': 'text', 'text': 'Mocked response'}]
    mock_agent.generate.return_value = mock_response
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    await controller.process_user_input("Hello, agent!")

    assert len(model.conversation_history) == 2
    assert model.conversation_history[0].role == 'user'
    assert model.conversation_history[1].role == 'assistant'
    assert model.conversation_history[1].last_text() == 'Mocked response'


@pytest.mark.asyncio
async def test_command_integration():
    """Test the integration of command handling with the Model."""
    model = Model()
    mock_agent_app = MagicMock()
    controller = Controller(model, mock_agent_app)

    # Test save command integration
    await controller.process_user_input("/save test_integration.json")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message == "History saved successfully."

    # Test clear command integration
    await controller.process_user_input("/clear")
    assert len(model.conversation_history) == 0
    assert model.last_success_message == "Conversation history cleared."


@pytest.mark.asyncio
async def test_error_handling_integration():
    """Test error handling integration between Model and Controller."""
    model = Model()
    
    # Mock agent that always fails
    mock_agent = AsyncMock()
    mock_agent.generate.side_effect = Exception("Test error")
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    # Add a message first to test rollback
    await model.add_message(Prompt.user("Previous message"))
    initial_history_length = len(model.conversation_history)

    await controller.process_user_input("This will fail")

    # Should have rolled back the user message
    assert len(model.conversation_history) == initial_history_length
    assert model.application_state == AppState.ERROR
    assert "Test error" in model.last_error_message


@pytest.mark.asyncio
async def test_state_management_integration():
    """Test that state management works correctly across the integration."""
    model = Model()
    mock_agent_app = MagicMock()
    controller = Controller(model, mock_agent_app)

    # Test that state changes are properly managed
    assert model.application_state == AppState.IDLE
    
    # Simulate a command that changes state
    await controller.process_user_input("/clear")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message is not None


@pytest.mark.asyncio
async def test_conversation_flow_integration():
    """Test a complete conversation flow with multiple turns."""
    model = Model()
    
    # Mock agent that returns different responses
    mock_agent = AsyncMock()
    responses = [
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'First response'}]),
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Second response'}]),
        MagicMock(role='assistant', content=[{'type': 'text', 'text': 'Third response'}])
    ]
    mock_agent.generate.side_effect = responses
    
    mock_agent_app = MagicMock()
    mock_agent_app.agent = mock_agent
    
    controller = Controller(model, mock_agent_app)

    # Simulate a conversation
    await controller.process_user_input("First message")
    await controller.process_user_input("Second message")
    await controller.process_user_input("Third message")

    assert len(model.conversation_history) == 6  # 3 user + 3 assistant messages
    assert model.conversation_history[0].role == 'user'
    assert model.conversation_history[1].role == 'assistant'
    assert model.conversation_history[2].role == 'user'
    assert model.conversation_history[3].role == 'assistant'
    assert model.conversation_history[4].role == 'user'
    assert model.conversation_history[5].role == 'assistant' 
```

--- END OF FILE tests/test_integration.py ---

--- START OF FILE tests/test_model.py ---

```py
import pytest
import tempfile
import os
from model import Model, AppState
from mcp_agent.core.prompt import Prompt


@pytest.mark.asyncio
async def test_model_initial_state():
    """Test that the model starts in the correct initial state."""
    model = Model()
    assert model.application_state == AppState.IDLE
    assert len(model.conversation_history) == 0
    assert model.last_error_message is None
    assert model.last_success_message is None


@pytest.mark.asyncio
async def test_model_state_change():
    """Test that state changes work correctly."""
    model = Model()
    assert model.application_state == AppState.IDLE
    
    await model.set_state(AppState.ERROR, "Test Error")
    assert model.application_state == AppState.ERROR
    assert model.last_error_message == "Test Error"
    
    await model.set_state(AppState.IDLE, "Test Success")
    assert model.application_state == AppState.IDLE
    assert model.last_success_message == "Test Success"


@pytest.mark.asyncio
async def test_add_message():
    """Test adding messages to conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    assert len(model.conversation_history) == 1
    assert model.conversation_history[0].role == 'user'
    
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    assert model.conversation_history[1].role == 'assistant'


@pytest.mark.asyncio
async def test_pop_last_message():
    """Test removing the last message from conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    
    await model.pop_last_message()
    assert len(model.conversation_history) == 1
    assert model.conversation_history[0].role == 'user'


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing the conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    assert len(model.conversation_history) == 2
    
    await model.clear_history()
    assert len(model.conversation_history) == 0


@pytest.mark.asyncio
async def test_save_and_load_history():
    """Test saving and loading conversation history."""
    model = Model()
    user_message = Prompt.user("Hello")
    assistant_message = Prompt.assistant("Hi there!")
    
    await model.add_message(user_message)
    await model.add_message(assistant_message)
    
    # Test saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_filename = f.name
    
    try:
        success = await model.save_history_to_file(temp_filename)
        assert success is True
        
        # Test loading
        new_model = Model()
        success = await new_model.load_history_from_file(temp_filename)
        assert success is True
        assert len(new_model.conversation_history) == 2
        assert new_model.conversation_history[0].role == 'user'
        assert new_model.conversation_history[1].role == 'assistant'
        
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@pytest.mark.asyncio
async def test_user_preferences():
    """Test user preferences functionality."""
    model = Model()
    
    # Test default preferences
    assert model.user_preferences.get("auto_save_enabled") is True
    
    # Test setting preferences
    model.user_preferences["auto_save_enabled"] = False
    assert model.user_preferences.get("auto_save_enabled") is False
    
    model.user_preferences["test_setting"] = "test_value"
    assert model.user_preferences.get("test_setting") == "test_value" 
```

--- END OF FILE tests/test_model.py ---

--- START OF FILE view.py ---

```py
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
```

--- END OF FILE view.py ---

