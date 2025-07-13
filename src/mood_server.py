import sys
import logging
from mcp.server.fastmcp import FastMCP
from mcp.server.elicitation import AcceptedElicitation, DeclinedElicitation, CancelledElicitation
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mood_server")

mcp = FastMCP("Mood Elicitation Server")

class MoodElicitationForm(BaseModel):
    mood: str = Field(
        description="In a few words, how are you feeling right now?",
        max_length=100
    )

@mcp.tool()
async def elicit_mood() -> str:
    """Elicits user mood through interactive form and returns structured response."""
    logger.info("Tool 'elicit_mood' called. Requesting free-form mood from user.")
    
    result = await mcp.get_context().elicit(
        "Please share how you're feeling.",
        schema=MoodElicitationForm
    )

    match result:
        case AcceptedElicitation(data=data):
            logger.info(f"User entered mood: '{data.mood}'")
            return f"The user described their mood as: '{data.mood}'"
        case DeclinedElicitation():
            logger.info("User declined the mood elicitation.")
            return "The user chose not to share their mood."
        case CancelledElicitation():
            logger.info("User cancelled the mood elicitation.")
            return "The user cancelled the request."

if __name__ == "__main__":
    mcp.run()