# main.py
import asyncio

from model import Model
from view import View
from controller import Controller, ExitCommand
from agent_definitions import fast

def print_shutdown_message():
    """Prints a consistent shutdown message."""
    print("\nClient session ended.")

async def main():
    """
    The main entry point for the application.
    """
    # This context manager handles startup and shutdown of MCP servers.
    async with fast.run() as agent_app:
        model = Model()
        controller = Controller(model, agent_app)
        view = View(model, controller)

        try:
            # The view's main loop now runs until an ExitCommand is raised.
            await view.run_main_loop()
        except ExitCommand:
            # This is our clean exit path.
            pass

    # This delay happens AFTER fast.run() has closed, giving background
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