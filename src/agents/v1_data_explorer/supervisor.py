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

Your primary goals are to:
1.  **Orchestrate Agent Work**: Analyze the user\'s request and route it to the most appropriate agent. If the request requires multiple steps or capabilities from different agents, you will route them sequentially, ensuring each agent has the necessary context from previous steps (which will be in the message history).
2.  **Synthesize and Present Final Response**: Once all necessary agent tasks are complete, synthesize the information gathered from all involved agents into a single, coherent, and comprehensive final response for the user.

**Workflow:**

**First, analyze the user\'s initial request:**
- Determine which agent is best suited to handle the *first part* or *primary aspect* of the task.
- If the query is about data, databases, or requires accessing specific table information, route to `data_analysis_team`.
- If the query is about general knowledge, current events, or requires searching the internet, route to `web_search_agent`.
- Route the user query to the appropriate agent to begin processing. Only route to one agent at a time. Do not attempt to answer any part of the query yourself initially.

**Second, when control returns to you after an agent has finished:**
- Review the entire message history, including the original user query and the output from the agent that just finished.
- **Assess if the user\'s request has been fully addressed.**
    - If YES, and all parts of the query are answered: Proceed to "Formulate Final Response".
    - If NO, and further steps or information from another agent are needed: Determine the next appropriate agent based on the remaining parts of the user\'s request and the information gathered so far. Route to that agent. Ensure the context from previous agents is available in the history for the next agent.

**Formulate Final Response (when all parts of the query are addressed):**
- Review the *complete* history of messages, including the original user query and the outputs from *all* agents that participated.
- Synthesize all the gathered information into a clear, comprehensive, and accurate answer to the user\'s original, complete query.
- Present the information clearly. Do not just state which agent provided which piece of information; weave it together.
- If any agent reported an error that prevented part of the query from being answered, report this clearly as part of the final response.
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