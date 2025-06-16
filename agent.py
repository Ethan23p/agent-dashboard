# agent.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# --- AGENT DEFINITION ---
# This file now only defines the agent and its capabilities.
# The client logic that runs it is in cli.py.

# Instantiate the main fast-agent application object.
fast = FastAgent("Minimal Controllable Agent")

# Decorator to define our agent.
@fast.agent(
    name="base_agent",
    instruction="You are a helpful and concise assistant. You have access to a filesystem.",
    servers=["filesystem"],
    use_history=False,
    request_params=RequestParams(max_tokens=2048)
)
async def define_agents():
    """
    This function is now just a placeholder for the decorator.
    The `fast.run()` context manager in cli.py will discover any agents
    defined in this file.
    """
    # We don't need any logic here anymore.
    pass