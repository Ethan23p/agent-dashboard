# main.py
import asyncio
import sys
import argparse

from model import Model
from view import View
from controller import Controller, ExitCommand, SwitchAgentCommand
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

async def run_agent_session(agent_name: str):
    """
    Run a session with a specific agent.
    
    Args:
        agent_name: The name of the agent to run
        
    Returns:
        The new agent name if switching, None if exiting
    """
    try:
        # Get the selected agent from the registry
        selected_agent = get_agent(agent_name)
        print(f"Starting {agent_name} agent...")
        
        # Run the selected agent
        async with selected_agent.run() as agent_app:
            # Initialize MVC components
            model = Model()
            controller = Controller(model, agent_app)
            view = View(model, controller)
            
            # Run the main loop until exit or switch
            await view.run_main_loop()
            return None  # Normal exit
            
    except SwitchAgentCommand as e:
        return e.agent_name  # Switch to new agent
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Available agents: {', '.join(list_available_agents())}")
        return None

async def main():
    """
    The main entry point for the application.
    """
    # Parse command line arguments
    args = parse_arguments()
    current_agent = args.agent
    
    # Main agent loop - handles switching between agents
    while current_agent is not None:
        current_agent = await run_agent_session(current_agent)
        if current_agent:
            print(f"\nSwitching to {current_agent} agent...")
            # Small delay to show the switch message
            await asyncio.sleep(0.5)

    # This delay happens AFTER all agents have closed, giving background
    # tasks time to finalize their shutdown before the script terminates.
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # We no longer catch SystemExit here, but keep it for robustness.
        pass
    finally:
        # The final message is printed after everything has shut down.
        print_shutdown_message()