from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from .config import get_llm_async
from .state import AgentState
from .web_search_agent import create_web_search_agent
from .data_team import create_data_team_graph
from src.config.logger import logger

# Supervisor Prompt
SUPERVISOR_PROMPT = """
You are a supervisor managing a team of agents. Your team consists of:
- web_search_agent: Performs web searches for general knowledge and current events.
- data_analysis_team: Handles tasks involving database schema lookup, SQL query generation, validation, and execution.

**First, analyze the user's request:**
Based on the user's request, determine which agent is best suited to handle the task. 
If the query is about data, databases, or requires accessing specific table information, route to data_analysis_team. 
If the query is about general knowledge, current events, or requires searching the internet, route to web_search_agent.

Route the user query to the appropriate agent to begin processing. Only route to one agent at a time. 
Do not attempt to answer the query yourself initially.

**Second, when control returns to you after an agent has finished:**
Review the history. The last message should contain the result or response from the agent who just finished their work. 
Your final response to the user should be based *directly* on this last message from the agent. Present the information clearly. Do not just state which agent provided the information; present the actual information itself. If the agent indicated an error, report the error.
"""

# Make function async
async def create_supervisor_agent() -> CompiledStateGraph:
    """Creates and compiles the main supervisor agent."""
    logger.info("Creating supervisor agent...")
    
    # --- Fetch LLM client first --- 
    llm_service = await get_llm_async()
    llm = llm_service.client
    # -----------------------------

    # Initialize the agents/teams (await the async creators)
    web_agent = await create_web_search_agent()
    # Pass llm client to data team graph creator
    data_team = create_data_team_graph(llm_client=llm)
    
    # List of agents the supervisor manages
    # The names must match the 'name' attribute of the compiled graphs/agents
    members = [web_agent.name, data_team.name]
    
    # Ensure AgentState is used
    options = AgentState
    
    # LLM was fetched above
    # llm_service = await get_llm_async()
    # llm = llm_service.client
    
    # Create the supervisor workflow
    # Note: The agents list expects compiled graphs/executors
    supervisor_workflow = create_supervisor(
        agents=[web_agent, data_team],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        # State definition is implicitly taken from the agents if they share one,
        # but explicitly providing it can be clearer if needed.
        # state_schema=AgentState 
    )
    
    # Compile the supervisor graph
    # You could add memory/checkpointer here if needed
    # e.g., app = supervisor_workflow.compile(checkpointer=InMemorySaver())
    app = supervisor_workflow.compile()
    logger.info("Supervisor agent created and compiled.")
    return app 