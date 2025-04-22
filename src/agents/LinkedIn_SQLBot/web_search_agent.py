from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from .config import get_llm_async
from .tools import web_tools

# Define the prompt for the web search agent
WEB_SEARCH_PROMPT = """
You are a web search expert. Your primary role is to use the web_search tool 
to find current, relevant information online based on the user's query. 
Focus on providing concise answers based on the search results. 
Do not engage in conversation beyond providing the search findings.
"""

# Make the function async
async def create_web_search_agent() -> CompiledStateGraph:
    """Creates and compiles the web search ReAct agent."""
    # Await the async getter and access the client
    llm_service = await get_llm_async()
    llm = llm_service.client
    
    agent_executor = create_react_agent(
        model=llm,
        tools=web_tools,
        name="web_search_agent",
        prompt=WEB_SEARCH_PROMPT
    )
    return agent_executor 