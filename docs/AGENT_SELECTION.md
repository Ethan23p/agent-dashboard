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