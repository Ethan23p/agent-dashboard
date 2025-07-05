# main.py
import asyncio

from model import Model
from view import View
from controller import Controller, ExitCommand
from agent_definitions import minimal_agent

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

async def main():
    """
    The main entry point for the application.
    """
    # Run the minimal agent
    async with minimal_agent.run() as agent_app:
        print("Starting minimal agent...")
        
        # Initialize MVC components
        model = Model()
        controller = Controller(model, agent_app)
        view = View(model, controller)
        
        # Run the main loop until exit
        await view.run_main_loop()

    # This delay happens AFTER minimal_agent.run() has closed, giving background
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