# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.
#
# NOTE: All agents should use use_history=False since we manage conversation
# history ourselves in the Model class and pass it explicitly to the agent.

# A list of dictionaries, where each dictionary defines an agent.
# This is flexible – only include the keys you need for each agent.
AGENT_DEFINITIONS = [
    {
        "name": "minimal",
        "description": "A helpful assistant for general operations.",
        "instruction": """
        You are a helpful assistant that can perform various operations.
        You can read files, write files, and list directory contents.
        Always be helpful and provide clear responses to user requests.
        """,
        "servers": ["filesystem", "fetch", "sequential-thinking"],
        "max_tokens": 2048,
    },
    {
        "name": "coding",
        "description": "A specialized coding assistant.",
        "instruction": """
        You are a specialized coding assistant. You excel at:
        - Code review and suggestions
        - Debugging and problem-solving
        - Explaining complex technical concepts
        - Providing code examples and best practices
        
        Always provide clear, well-documented code examples when relevant.
        """,
        "servers": ["filesystem"],
        "max_tokens": 4096,
    },
    {
        "name": "interpreter",
        "description": "A structured data interpreter.",
        "instruction": """
        You are a highly efficient data parsing engine.
        Given a user's natural language text and a target JSON schema,
        your sole purpose is to extract the relevant information and respond
        ONLY with the JSON object that conforms to the schema.
        """,
        "use_history": False,
    },
]

def _create_agent_from_definition(definition: dict) -> FastAgent:
    """Factory function to build a FastAgent instance from a dictionary."""
    
    # Use .get() to provide defaults for optional keys
    agent_name = definition.get("name", "minimal")
    description = definition.get("description", "A fast-agent.")
    instruction = definition.get("instruction", "You are a helpful assistant.")
    servers = definition.get("servers", [])
    max_tokens = definition.get("max_tokens", 2048)

    agent_instance = FastAgent(description, config_path="src/fastagent.config.yaml")

    # The decorator needs a function to decorate, even a placeholder
    @agent_instance.agent(
        name=agent_name,
        instruction=instruction,
        servers=servers,
        request_params=RequestParams(maxTokens=max_tokens),
        use_history=False
    )
    async def placeholder_func(): pass
    
    return agent_instance

# The registry is now BUILT dynamically from the definitions list.
AGENT_REGISTRY = {}

# Default agent (first one in the list)
DEFAULT_AGENT = AGENT_DEFINITIONS[0]["name"] if AGENT_DEFINITIONS else "minimal"

# Populate the registry
for definition in AGENT_DEFINITIONS:
    agent_name = definition.get("name")
    if agent_name:
        AGENT_REGISTRY[agent_name] = _create_agent_from_definition(definition)

def get_agent(agent_name: str = None):
    """
    Get an agent by name from the registry.
    
    Args:
        agent_name: The name of the agent to retrieve. If None, uses DEFAULT_AGENT
        
    Returns:
        The FastAgent instance for the requested agent
        
    Raises:
        KeyError: If the agent name is not found in the registry
    """
    
    if agent_name is None:
        agent_name = DEFAULT_AGENT

    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """Return a list of available agent names."""
    return list(AGENT_REGISTRY.keys())
