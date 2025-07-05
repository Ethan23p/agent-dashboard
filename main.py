# main.py
import asyncio

from model import Model
from view import View
from controller import Controller, ExitCommand
from agent_definitions import fast_planner

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

async def main():
    """
    The main entry point for the application.
    """
    # Run the planning workflow
    async with fast_planner.run() as agent:
        print("Starting planning workflow...")
        await agent.prompt("approve_and_execute_workflow")

    # This delay happens AFTER fast_planner.run() has closed, giving background
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