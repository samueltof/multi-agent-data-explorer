from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from .config import get_llm
from .tools import web_tools

# Define the prompt for the web search agent
WEB_SEARCH_PROMPT = """
You are a web search expert. Your primary role is to use the web_search tool 
to find current, relevant information online based on the user's query. 
Focus on providing concise answers based on the search results. 
Do not engage in conversation beyond providing the search findings.
"""


def create_web_search_agent() -> CompiledStateGraph:
    """Creates and compiles the web search ReAct agent."""
    llm = get_llm().client  # Get the underlying LangChain LLM client
    
    agent_executor = create_react_agent(
        model=llm,
        tools=web_tools,
        name="web_search_agent",
        prompt=WEB_SEARCH_PROMPT
    )
    return agent_executor 