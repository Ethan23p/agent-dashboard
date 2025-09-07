import asyncio
import sys
import argparse
import logging
from typing import Optional

from textual_view import AgentDashboardApp
from agent_registry import list_available_agents, DEFAULT_AGENT

def setup_logging():
    """Initializes logging to a file and stderr for critical errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="agent-dashboard.log",
        filemode="a"
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(console_handler)
    logging.info("--- Application session started ---")

def print_shutdown_message():
    print("\nClient session ended.")
    logging.info("--- Application session ended ---")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Agent Dashboard")
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default=DEFAULT_AGENT,
        help=f"Select agent to use. Available: {', '.join(list_available_agents())}"
    )
    return parser.parse_args()

class Application:
    """Orchestrates the main application components."""
    def __init__(self, initial_agent_name: str):
        self.initial_agent_name = initial_agent_name

    async def run(self):
        tui_app = AgentDashboardApp(
            agent_name=self.initial_agent_name
        )
        await tui_app.run_async()

async def main():
    """Application entry point."""
    setup_logging()
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
