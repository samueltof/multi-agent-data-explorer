import sys
import os
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
import argparse
import asyncio

# MCP imports for Tavily
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure the src directory is in the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, AIMessage
# AgentState is not directly used in main.py but is central to the agent's operation.
# from .state import AgentState 
from .supervisor import create_supervisor_agent
# Tools are used by agents, not directly in main.py orchestration usually.
# from .tools import data_tools, web_tools 
from src.config.logger import logger

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Combine the tool lists - This might not be needed here if tools are managed by agents
# all_tools = data_tools + web_tools

async def run_agent_session(query: str):
    """Runs the supervisor agent for a single query and handles its lifecycle, including Tavily MCP setup."""
    logger.info(f"Starting agent execution with query: '{query}'")
    
    app = None
    final_state_snapshot = None

    if not TAVILY_API_KEY:
        logger.error("TAVILY_API_KEY not found in environment variables. Cannot start Tavily MCP server.")
        print("Agent Error: TAVILY_API_KEY is not set. Web search capabilities will be unavailable.")
        # Optionally, decide if you want to proceed without web search or halt
        # For now, we'll let it try and potentially fail at supervisor creation if web agent is critical
        # or if create_supervisor_agent handles the missing session gracefully (it currently expects one).
        # To be robust, create_supervisor_agent might need to handle a None session if we allow this.
        # However, current design expects an active session for the web_search_agent.
        return # Halt if Tavily key is missing, as web_search_agent depends on it.

    # Setup Tavily MCP Server Parameters
    tavily_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@latest"], # Use a specific version if preferred, e.g., tavily-mcp@0.2.0
        env={"TAVILY_API_KEY": TAVILY_API_KEY, "NODE_NO_WARNINGS": "1"},
        # Ensure logs from MCP server can be captured or suppressed if too verbose
    )

    try:
        # Manage Tavily MCP client and session lifecycle
        async with stdio_client(tavily_server_params) as (read, write):
            logger.info("Tavily MCP stdio_client started.")
            async with ClientSession(read, write) as tavily_session:
                logger.info("Tavily MCP ClientSession started. Initializing...")
                await tavily_session.initialize()
                logger.info("Tavily MCP ClientSession initialized.")

                # Create the supervisor agent application, passing the active Tavily session
                app = await create_supervisor_agent(tavily_mcp_session=tavily_session)
                
                inputs = {"messages": [HumanMessage(content=query)]}
                
                logger.info("Streaming agent execution steps...")
                async for output in app.astream(inputs, stream_mode="values"):
                    final_state_snapshot = output 

                if final_state_snapshot and "messages" in final_state_snapshot:
                    final_messages = final_state_snapshot["messages"]
                    if final_messages:
                        last_message = final_messages[-1]
                        if (isinstance(last_message, AIMessage) and
                           not last_message.tool_calls and
                           not getattr(last_message.response_metadata, '__is_handoff_back', False)):
                            logger.info(f"ALL Response: {final_messages}") # Potentially very verbose
                            print("\nAgent Final Response:")
                            print(last_message.content)
                            print("-" * 30)
                        else:
                            logger.info("Last message was not a displayable AIMessage or was a handoff.")
                    else:
                        logger.warning("Final state snapshot contained no messages.")
                else:
                     logger.warning("No final state snapshot captured or messages key missing.")

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        print(f"\nAgent encountered an error: {e}\n")
    finally:
        # app and tavily_session/client will be cleaned up by async with context exit
        logger.debug("run_agent_session finished.")
        # Explicit deletion can help hint GC but context managers handle resource release.
        del app
        del final_state_snapshot


def main_chat_loop():
    """Main loop for the chat interface."""
    print("Starting Data Explorer Agent Chat Interface...")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not user_query.strip():
                continue
                
            print("Agent: Processing...")
            asyncio.run(run_agent_session(user_query))
            
        except KeyboardInterrupt:
            print("\nExiting due to KeyboardInterrupt...")
            break
        except Exception as e:
            logger.error(f"An error occurred in the chat loop: {e}", exc_info=True)
            print("An unexpected error occurred. Please check logs or try again.")
        finally:
            pass
    
    logger.info("Chat interface stopped.")

if __name__ == "__main__":
    main_chat_loop() 