
# main.py
import asyncio
import argparse

from model import Model
from textual_view import AgentDashboardApp  # <-- Import the new Textual view
from controller import Controller, SwitchAgentCommand
from agent_definitions import get_agent, list_available_agents

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

def parse_arguments():
    """Parse command line arguments for agent selection."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default="minimal",
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

async def run_agent_session(agent_name: str) -> str | None:
    """
    Run a session with a specific agent using the Textual UI.
    
    Args:
        agent_name: The name of the agent to run
        
    Returns:
        The new agent name if switching, None if exiting.
    """
    try:
        selected_agent = get_agent(agent_name)
        print(f"Starting {agent_name} agent...")
        
        async with selected_agent.run() as agent_app:
            model = Model()
            controller = Controller(model, agent_app)
            
            # Instantiate and run the Textual app
            tui_app = AgentDashboardApp(model, controller)
            
            # The `run` method is blocking. It will return a result when
            # the app calls `self.exit(result=...)`.
            switch_to_agent = await tui_app.run_async()
            
            # If the app exited with a result, it's the name of the new agent.
            return switch_to_agent

    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available agents: {', '.join(list_available_agents())}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

async def main():
    """
    The main entry point for the application. It manages the agent-switching loop.
    """
    args = parse_arguments()
    current_agent = args.agent
    
    while current_agent is not None:
        # run_agent_session will block until the Textual app exits.
        # It returns the name of the next agent, or None to quit.
        next_agent = await run_agent_session(current_agent)
        
        if next_agent:
            print(f"\nSwitching to {next_agent} agent...")
            await asyncio.sleep(0.1) # Brief pause for visual feedback
            current_agent = next_agent
        else:
            # No next agent, so we exit the loop.
            current_agent = None

    # This delay happens AFTER all agents have closed, giving background
    # tasks time to finalize their shutdown before the script terminates.
    await asyncio.sleep(0.1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print_shutdown_message()
