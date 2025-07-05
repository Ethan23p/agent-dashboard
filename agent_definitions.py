# agent_definitions.py
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# This module's sole purpose is to define the agents for the application.
# It acts as a catalog that can be imported by any client or runner.

# FastAgent instance for planning functionality
fast_planner = FastAgent("Human-Approved Planning Agent")

# --- Planning Agent Definitions ---

# 1. The Pre-Processor Agent: Lists tools for the planner.
@fast_planner.agent(
    name="tool_lister",
    instruction="""
    Based on the tools available to you, generate a concise summary of your capabilities.
    This summary will be given to a planning agent.
    Example: 'The executor can write files, read files, and list directories.'
    """,
    # This agent sees the same tools as the executor.
    servers=["filesystem"],
    request_params=RequestParams(maxTokens=2048)
)

# 2. The Planner Agent: Now receives the tool list and is explicitly instructed on efficiency.
@fast_planner.agent(
    name="planner",
    instruction="""
    You are a master planning agent. You will be given a user's goal and a summary of the tools available for execution.
    When creating the plan, you MUST prioritize efficiency. If a tool exists to perform a bulk action
    (e.g., read_multiple_files), your plan should use that single tool call instead of multiple individual calls.

    If you are also given feedback on a previous plan, you MUST use it to generate a new, improved plan.
    Your output must be a step-by-step plan in markdown format.
    """,
    # This agent gets the new read-only filesystem server
    servers=["readonly_fs"],
    use_history=True,
    request_params=RequestParams(maxTokens=2048)
)

# 3. The Human Feedback Agent: Unchanged, but essential.
@fast_planner.agent(
    name="human_feedback_evaluator",
    instruction="""
    You have been given a plan to show to a human user.
    Present the plan and ask for their feedback or approval using the Human Input tool.
    If the user approves the plan (e.g., says 'yes', 'looks good', 'proceed'), respond with the single word 'EXCELLENT'.
    If the user provides any suggestions, changes, or critique, your output MUST be their verbatim feedback.
    """,
    human_input=True,
    request_params=RequestParams(maxTokens=2048)
)

# 4. The Executor Agent: Unchanged.
@fast_planner.agent(
    name="executor",
    instruction="Execute the given step-by-step plan precisely using your available tools.",
    servers=["filesystem"], # The full-access server
    request_params=RequestParams(maxTokens=2048)
)

# 5. The Planning Chain: Now includes the tool_lister pre-processor.
# We set cumulative=True so the original prompt is passed along with the tool list.
@fast_planner.chain(
    name="plan_generation_chain",
    sequence=["tool_lister", "planner"],
    cumulative=True
)

# 6. The Feedback Loop: Now uses the new planning chain as its generator.
@fast_planner.evaluator_optimizer(
    name="interactive_planner",
    generator="plan_generation_chain", # Use the new chain
    evaluator="human_feedback_evaluator",
    min_rating="EXCELLENT",
    max_refinements=5
)

# 7. The Final Workflow: The main entry point.
@fast_planner.chain(
    name="approve_and_execute_workflow",
    sequence=["interactive_planner", "executor"],
    instruction="Generates a plan with iterative human feedback, then executes the final approved plan."
)
async def approve_and_execute_workflow():
    """ This function is a placeholder for the decorator. """
    pass
