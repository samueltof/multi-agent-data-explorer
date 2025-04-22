from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from .config import get_llm_async
from .state import AgentState
from .web_search_agent import create_web_search_agent
from .data_team import data_team_graph
from src.config.logger import logger
from typing import List, Sequence, Any, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.config.llm_config import LLMSettings
from .tools import web_tools

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

# Rename function and make it the main export for getting the graph
async def get_supervisor_graph() -> CompiledStateGraph:
    """Creates and compiles the main supervisor agent graph."""
    logger.info("Creating supervisor graph...")
    
    # --- Fetch LLM client first ---
    # This assumes get_llm_async provides the necessary client
    # If ChatOpenAI is needed directly, we might need LLMSettings here too
    # llm_service = await get_llm_async()
    # llm = llm_service.client
    # Use the direct instantiation for the supervisor's LLM
    # Map settings fields correctly to ChatOpenAI parameters
    openai_settings = LLMSettings().openai
    supervisor_llm = ChatOpenAI(
        model=openai_settings.default_model,  # Map default_model to model
        api_key=openai_settings.api_key,
        temperature=openai_settings.temperature,
        max_tokens=openai_settings.max_tokens,
        # Add other relevant ChatOpenAI params if needed from openai_settings
    )
    # -----------------------------

    # Initialize the agents/teams (await the async creators)
    # Ensure agents have .name attribute set correctly
    web_search_agent_graph = await create_web_search_agent()
    # data_team_graph is imported directly
    data_team_analysis_graph = data_team_graph 
    
    # Verify agent names match the prompt expectations
    # web_search_agent_graph.name should be 'web_search_agent'
    # data_team_analysis_graph.name should be 'data_analysis_team'
    logger.info(f"Supervisor using agents: Web Search ('{web_search_agent_graph.name}'), Data Team ('{data_team_analysis_graph.name}')")

    # List of compiled agent graphs the supervisor manages
    agents = [web_search_agent_graph, data_team_analysis_graph]
    
    # Create the supervisor workflow using the library function
    supervisor_workflow = create_supervisor(
        agents=agents,
        model=supervisor_llm, # Use the LLM fetched/created for the supervisor
        prompt=SUPERVISOR_PROMPT,
        # state_schema=AgentState # Optional: Define state if needed explicitly
    )
    
    # Compile the supervisor graph
    # Consider adding memory/checkpointer here if needed
    app = supervisor_workflow.compile()
    logger.info("Supervisor graph created and compiled.")
    return app 