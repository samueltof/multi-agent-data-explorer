import os
from dotenv import load_dotenv
from mcp import ClientSession # StdioServerParameters, stdio_client are not used here directly
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from .config import get_llm_async
# Remove direct import of web_tools as it will be replaced by MCP tools
# from .tools import web_tools 

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Used for a check, actual key used by MCP server process

# Define the prompt for the web search agent
WEB_SEARCH_PROMPT = """
You are a web search expert. Your primary role is to use the Tavily Search tool 
to find current, relevant information online based on the user's query. 
Focus on providing concise answers based on the search results. 
Do not engage in conversation beyond providing the search findings.
You have access to Tavily search tools.
"""

# This function will now also handle the MCP client setup.
# It might be better to manage the MCP client session lifecycle outside
# if the agent is long-lived and used multiple times.
# For now, let's assume the session is created with the agent.
# We'll need to decide if this function returns the agent AND the session,
# or if the session is managed internally by the tools after loading.

# A major consideration: `create_react_agent` is not async, but `load_mcp_tools` is.
# The `stdio_client` and `ClientSession` are async context managers.
# This means `create_web_search_agent` itself doesn't create the agent directly,
# but rather provides the components or a factory.
#
# Let's rethink. The supervisor will need to manage the MCP connection.
# So, `create_web_search_agent` should perhaps take a list of MCP tools
# that have already been loaded.

# Alternative: The function returns an async function that, when called,
# enters the context, creates the agent, and runs it. This seems complex.

# Let's assume `create_web_search_agent` will be responsible for returning an
# agent that is already configured with tools from an active MCP session.
# This implies the session management might need to be handled by the caller,
# or the tools themselves handle the session after `load_mcp_tools`.

# Based on `simple_mcp_client.py`, `load_mcp_tools(session)` returns tools.
# The agent is then created with these tools. The session needs to be active.

# Let's make `create_web_search_agent` a function that takes an active MCP session
# and returns an agent configured to use tools from that session.
# The MCP server setup and session management will then be moved to a higher level,
# likely where the supervisor agent is created or run.

async def create_web_search_agent(mcp_session: ClientSession) -> CompiledStateGraph:
    """
    Creates and compiles the web search ReAct agent using tools from an active Tavily MCP session.
    The MCP session must be initialized (session.initialize()) and managed by the caller.
    """
    if not TAVILY_API_KEY:
        # This is a local check; the MCP server process independently uses the key from its env.
        print("Warning: TAVILY_API_KEY not found in local environment variables. "
              "Ensure it's available to the Tavily MCP server process.")

    llm_service = await get_llm_async()
    llm = llm_service.client
    
    # Load tools from the provided MCP session
    # The caller is responsible for `await mcp_session.initialize()` before passing it here.
    mcp_tools = await load_mcp_tools(mcp_session)
    
    if not mcp_tools:
        # Log this or raise a more specific error if possible
        print("Error: Failed to load any tools from the Tavily MCP session.")
        raise RuntimeError("Failed to load any tools from the Tavily MCP session. Ensure the MCP server is running and configured correctly.")

    # Optional: Log the names of loaded tools for debugging
    # print(f"Loaded tools from Tavily MCP: {[tool.name for tool in mcp_tools]}")

    agent_executor = create_react_agent(
        model=llm,
        tools=mcp_tools,
        name="web_search_mcp_agent", # Renamed to clearly indicate MCP usage
        prompt=WEB_SEARCH_PROMPT
    )
    return agent_executor

# Example of how this might be used (conceptual, actual use in supervisor):
# async def setup_and_get_web_agent():
#     if not TAVILY_API_KEY:
#         raise ValueError("TAVILY_API_KEY not found in environment variables.")
#
#     server_params = StdioServerParameters(
#         command="npx",
#         args=["-y", "tavily-mcp@latest"],
#         env={"TAVILY_API_KEY": TAVILY_API_KEY, "NODE_NO_WARNINGS": "1"},
#     )
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             web_agent = await create_web_search_agent(session)
#             # Now web_agent can be used as long as the session is active.
#             # This implies the agent needs to be used within this context,
#             # or the tools/agent handle the session state internally.
#             return web_agent 
#
# This structure indicates that the agent's lifecycle is tied to the MCP session's lifecycle.
# The supervisor will need to manage this. 