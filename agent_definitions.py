# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.

fast = FastAgent("Minimal Controllable Agent")

@fast.agent(
    name="base_agent",
    instruction="You are a helpful and concise assistant. You have access to a filesystem.",
    servers=["filesystem"],
    use_history=False,
    request_params=RequestParams(max_tokens=2048),
)
async def define_agents():
    """
    This function is a placeholder for the decorator. The `fast.run()`
    context manager will discover any agents defined in this file.
    defined in this file.
    """
    pass
