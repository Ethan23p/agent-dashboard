# main.py
import asyncio
import sys
import argparse
from typing import Optional

from model import Model
from textual_view import AgentDashboardApp
from controller import Controller
from agent_registry import list_available_agents, DEFAULT_AGENT

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

def parse_arguments():
    """Parse command line arguments for agent selection."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=DEFAULT_AGENT,
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

class Application:
    """
    The main application class that orchestrates the Model, View, and Controller.
    """
    def __init__(self, initial_agent_name: str):
        # The TUI now creates the Model and Controller
        self.initial_agent_name = initial_agent_name

    async def run(self):
        """
        Initializes and runs the Textual user interface. The TUI now drives
        the application by sending user input to the controller.
        """
        tui_app = AgentDashboardApp(
            agent_name=self.initial_agent_name
        )
        await tui_app.run_async()

async def main():
    """
    The main entry point for the application.
    """
    args = parse_arguments()
    app = Application(initial_agent_name=args.agent)
    await app.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print_shutdown_message()
