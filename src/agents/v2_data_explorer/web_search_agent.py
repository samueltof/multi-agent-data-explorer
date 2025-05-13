import os
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

# Added for ChatPromptTemplate and SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
# AgentScratchpadMessagesPlaceholder for react agent
from langchain.agents.format_scratchpad.openai_tools import ( # Or other appropriate scratchpad
    format_to_openai_tool_messages,
)
from langchain_core.messages import AIMessage, HumanMessage

from .config import get_llm_async

# Define the system prompt for the web search agent
WEB_SEARCH_SYSTEM_PROMPT = """
You are an expert web researcher. Your primary role is to use the available Tavily tools 
to find current, relevant information online based on the user's query. 
Focus on providing concise answers based on the search results. 
Do not engage in conversation beyond providing the search findings.
When you have the answer, respond to the user directly.
"""

# Make the function async and accept ClientSession
async def create_web_search_agent(tavily_mcp_session: ClientSession) -> CompiledStateGraph:
    """Creates and compiles the web search ReAct agent using tools from a Tavily MCP session."""
    # Await the async getter and access the client
    llm_service = await get_llm_async()
    llm = llm_service.client
    
    # Load tools from the provided MCP session
    tavily_tools = await load_mcp_tools(tavily_mcp_session)
    
    # Construct the prompt for create_react_agent
    # Based on typical react agent setups
    # Note: The exact placeholder names might vary based on the react agent's internal implementation
    # We might need MessagesPlaceholder(variable_name="agent_scratchpad")
    # or ensure the llm and tools are compatible with format_to_openai_tool_messages if used.
    
    # Let's try a more standard prompt structure for create_react_agent
    # which typically expects 'input' and 'agent_scratchpad' or similar.
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(WEB_SEARCH_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"), # For history/input
            MessagesPlaceholder(variable_name="agent_scratchpad"), # For react agent's thoughts/tool calls
        ]
    )

    agent_executor = create_react_agent(
        model=llm,
        tools=tavily_tools,
        prompt=prompt # Use the prompt argument
    )
    return agent_executor 