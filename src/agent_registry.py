from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from typing import Optional

# This module acts as a centralized catalog for agent definitions, making it
# easy to manage and access different agent configurations.
#
# NOTE: All agents should use `use_history=False`, as the application's Model
# manages conversation history explicitly.

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
    """Factory function to build a FastAgent from a definition dictionary."""
    agent_name = definition.get("name", "minimal")
    description = definition.get("description", "A fast-agent.")
    instruction = definition.get("instruction", "You are a helpful assistant.")
    servers = definition.get("servers", [])
    max_tokens = definition.get("max_tokens", 2048)

    agent_instance = FastAgent(description, config_path="src/fastagent.config.yaml")

    # The decorator requires a function to decorate, even if it's a placeholder.
    @agent_instance.agent(
        name=agent_name,
        instruction=instruction,
        servers=servers,
        request_params=RequestParams(maxTokens=max_tokens),
        use_history=False
    )
    async def placeholder_func(): pass
    
    return agent_instance

# The registry is built dynamically from the definitions list.
AGENT_REGISTRY = {
    definition["name"]: _create_agent_from_definition(definition)
    for definition in AGENT_DEFINITIONS if "name" in definition
}

DEFAULT_AGENT = AGENT_DEFINITIONS[0]["name"] if AGENT_DEFINITIONS else "minimal"

def get_agent(agent_name: Optional[str] = None):
    """
    Retrieves an agent from the registry by name.
        
    Raises:
        KeyError: If the agent name is not found.
    """
    agent_name = agent_name or DEFAULT_AGENT

    if agent_name not in AGENT_REGISTRY:
        available_agents = ", ".join(AGENT_REGISTRY.keys())
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {available_agents}")
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """Returns a list of available agent names."""
    return list(AGENT_REGISTRY.keys())