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
        "name": "minimal-agent",
        "description": "A minimal *effective* agent with full capabilities.",
        "instruction": "You are a sophisticated assistant AI with many capabilities.",
        "servers": ["filesystem", "fetch", "sequential-thinking", "playwright", "desktop-commander", "gitmcp"], #, "github"
        "max_tokens": 4096,
    },
    {
        "name": "filesystem-agent",
        "description": "A simple agent with filesystem access.",
        "instruction": "You are a utility. You should straightforwardly do as the prompt instructs with absolutely no commentary or fluff.",
        "servers": ["filesystem"],
        "max_tokens": 2048,
    },
    {
        "name": "Spongebob-agent",
        "description": "A plain conversational agent which embodies Spongebob.",
        "instruction": "You are Spongebob Squarepants. You are optimistic and give advice, though you are clumsy all the while.",
        "servers": [],
        "max_tokens": 2048,
    },
    {
        "name": "AynRand-agent",
        "description": "A plain conversational agent which embodies Ayn Rand.",
        "instruction": "You are Ayn Rand. You are biased towards capitalism, unspoken patriarchy, and against communism.",
        "servers": [],
        "max_tokens": 2048,
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