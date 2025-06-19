# cli.py
import asyncio
from typing import List

from prompt_toolkit import PromptSession

from agent import fast
from history_manager import save_conversation_to_file

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

ENABLE_AUTO_SAVE = True
AUTO_SAVE_FILENAME = "_context/autosave_history.json"


async def main():
    """
    The main client application loop.
    """
    async with fast.run() as agent_app:
        conversation_history: List[PromptMessageMultipart] = []
        prompt_session = PromptSession()

        print(
            "Agent is ready. Type '/save [filename]' to save history, or '/exit' to quit."
        )
        if ENABLE_AUTO_SAVE:
            print(
                f"Auto-saving is ON. History will be saved to '{AUTO_SAVE_FILENAME}' after each turn."
            )

        while True:
            print("\n" + "---" * 20 + "\n")
            print("You:")
            try:
                user_input = await prompt_session.prompt_async("")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if user_input.strip().lower() in ["/exit", "/quit", "exit", "quit"]:
                print("Session ended.")
                break

            if user_input.strip().lower().startswith("/save"):
                from history_manager import handle_save_command

                await handle_save_command(user_input, conversation_history)
                continue

            user_message = Prompt.user(user_input)
            conversation_history.append(user_message)

            try:
                response_message = await agent_app.base_agent.generate(
                    conversation_history
                )
                conversation_history.append(response_message)

                final_text = response_message.last_text()

                print("\nAgent:")
                if final_text:
                    indented_text = "\n".join(
                        ["    " + line for line in final_text.splitlines()]
                    )
                    print(indented_text)

                if ENABLE_AUTO_SAVE:
                    await save_conversation_to_file(
                        conversation_history, AUTO_SAVE_FILENAME
                    )

            except Exception as e:
                print(f"\n[ERROR] An error occurred: {e}")

    # A small delay to prevent shutdown race conditions.
    # This should be removed/resolved gracefully if/when we transition to a frontend UI.
    await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient interrupted.")
