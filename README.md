# Agent Dashboard

A terminal client for the `fast-agent` framework with a modern Textual-based UI.

This project started as a way to have a more stable and transparent interface for agent development. The core is a Model-View-Controller (MVC) architecture, separating the application's state from its terminal UI and logic.

## Features

- **Multiple Agent Support**: Switch between different specialized agents (minimal, coding, interpreter)
- **Modern Textual UI**: Clean, responsive terminal interface with command support
- **Context Management**: Comprehensive conversation history and state management
- **Resilient Operation**: Error handling with retry logic and clean shutdown
- **Command System**: Built-in commands for switching agents, saving/loading history, and more

## Available Agents

- **minimal**: General-purpose assistant with filesystem, fetch, and sequential-thinking capabilities
- **coding**: Specialized coding assistant with enhanced debugging and code review features
- **interpreter**: Structured data interpreter for JSON schema extraction

## Project Structure

```md
agent-dashboard/
├── src/                           # Main application code
│   ├── main.py                   # Application entry point
│   ├── controller.py              # Business logic controller
│   ├── model.py                   # Data model and state management
│   ├── textual_view.py            # Textual-based UI
│   ├── agent_registry.py          # Agent definitions and registry
│   ├── commands.py                # Command implementations
│   ├── secure_filesystem_server.py # MCP filesystem server
│   └── fastagent.config.yaml     # FastAgent configuration
├── tests/                         # Test suite
├── paperwork/                     # Documentation and project files
│   ├── AGENT_SELECTION.md         # Agent selection guide
│   └── CHANGELOG.md              # Version history
└── _context/                      # Session history and context files
```

## Technical Details

The client is built with a few key ideas in mind:

- **Context Management.** Following the philosophy of the Model Context Protocol, the controller assembles the conversational history and other data to form the precise context sent to the agent on each turn. This allows for more deliberate, developer-driven context strategies.

- **Asynchronous Core.** The application uses `asyncio` and a non-blocking prompt, which keeps the UI responsive. It's designed to support more complex operations, like parallel agent interactions, and could be adapted for a GUI dashboard later.

- **Stateful History.** While the terminal shows a clean chat log, a comprehensive history is maintained in the background. This history can be saved automatically or manually, providing a useful artifact for debugging or resuming sessions.

- **Resilient Operation.** LLM or MCP server errors are handled by the controller, which rolls back the conversational state to its last valid point. The application also shuts down cleanly to avoid resource errors.

- **Comprehensive Testing.** The application includes a complete testing suite with unit tests, integration tests, and retry mechanisms to ensure reliability and maintainability.

## Running the Application

```bash
# From the project root
uv run python src/main.py

# With specific agent
uv run python src/main.py --agent coding

# List available agents
uv run python src/main.py --help
```

## Using the Application

Once the application starts, you'll see a clean terminal UI with:

- **Chat Interface**: Type your messages and press Enter to send
- **Command System**: Use commands starting with `/`:
  - `/switch <agent>` - Switch to a different agent
  - `/agents` - List available agents
  - `/save [filename]` - Save conversation history
  - `/load <filename>` - Load conversation history
  - `/clear` - Clear current conversation
  - `/exit` or `/quit` - Exit the application
