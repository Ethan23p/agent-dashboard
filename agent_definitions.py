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
    servers=["filesystem"],
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
