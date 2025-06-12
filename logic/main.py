# main.py
import asyncio
from mcp_agent.core.fastagent import FastAgent

# --- Configuration for our simple agent ---
# Note: We are not using any MCP servers in this first step.
AGENT_CONFIG = {
    "name": "basic_agent",
    "model": "google.gemini-1.5-flash-latest", # Or any other model you have configured
    "instruction": "You are a helpful and friendly assistant.",
}

# --- fast-agent Application Setup ---
fast = FastAgent("basic-agent-app")

@fast.agent(
    name=AGENT_CONFIG["name"],
    instruction=AGENT_CONFIG["instruction"],
    model=AGENT_CONFIG["model"],
)
async def main():
    """
    Defines the agent and starts a simple, interactive chat session.
    """
    print(f"Starting agent '{AGENT_CONFIG['name']}'...")

    # The 'async with' block handles connecting to the LLM provider.
    async with fast.run() as agent:
        # The .interactive() method starts the command-line chat loop.
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())