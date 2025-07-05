# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.

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
    request_params=RequestParams(maxTokens=2048)
)

async def agent():
    """ This function is a placeholder for the decorator. """
    pass
