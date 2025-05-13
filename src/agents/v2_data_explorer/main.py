import sys
import os
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
import argparse
import asyncio # Import asyncio

# Ensure the src directory is in the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, AIMessage # Ensure AIMessage is imported
# AgentState is not directly used in main.py but is central to the agent's operation.
# from .state import AgentState 
from .supervisor import create_supervisor_agent
# Tools are used by agents, not directly in main.py orchestration usually.
# from .tools import data_tools, web_tools 
from src.config.logger import logger
from .tavily_mcp import tavily_mcp_client_session # Import Tavily session manager
from .config import get_llm_async # To get the LLM service
from langgraph.graph.state import CompiledStateGraph # For type hinting app

load_dotenv()

# Combine the tool lists - This might not be needed here if tools are managed by agents
# all_tools = data_tools + web_tools

# Modified to accept the pre-initialized app
async def run_agent_session(query: str, app: CompiledStateGraph):
    """Runs the pre-initialized supervisor agent for a single query."""
    logger.info(f"Executing agent session with query: '{query}'")
    
    final_state_snapshot = None
    try:
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
                    logger.info(f"ALL Response: {final_messages}")
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
        # Log the error within this session, but allow main_chat_loop to continue
        logger.error(f"Error during agent session for query '{query}': {e}", exc_info=True)
        print(f"\nAgent encountered an error processing your query: {e}\n")
    # Removed finally block that deleted app, as app is now managed by main_chat_loop

# Modified to be async and manage resource lifecycles
async def main_chat_loop():
    """Main loop for the chat interface, managing supervisor and resource lifecycles."""
    print("Initializing Data Explorer Agent Chat Interface...")
    
    llm_service = None # Initialize to None for finally block
    app = None # Initialize to None for finally block

    try:
        async with tavily_mcp_client_session() as actual_tavily_session:
            logger.info("Tavily MCP session active.")
            
            # Get LLM service. Note: get_llm_async() itself is not an async context manager.
            # We will manage its client's potential aclose() if available.
            llm_service = await get_llm_async()
            llm_client = llm_service.client # This is the actual Langchain chat model instance
            logger.info("LLM service initialized.")

            app = await create_supervisor_agent(actual_tavily_session, llm_client)
            logger.info("Supervisor agent initialized and ready.")
            print("Agent is ready. Type 'exit' or 'quit' to end.")
            
            while True:
                try:
                    user_query = await asyncio.to_thread(input, "You: ") # Use async input
                    if user_query.lower() in ["exit", "quit"]:
                        print("Exiting...")
                        break
                    if not user_query.strip():
                        continue
                        
                    print("Agent: Processing...")
                    await run_agent_session(user_query, app) # Pass the initialized app
                    
                except KeyboardInterrupt:
                    print("\nExiting due to KeyboardInterrupt...")
                    break # Exit while loop
                except Exception as e:
                    logger.error(f"An error occurred in the inner chat query loop: {e}", exc_info=True)
                    print("An unexpected error occurred with your query. Please try again.")
        
    except Exception as e:
        logger.error(f"A critical error occurred during agent setup or main loop: {e}", exc_info=True)
        print(f"A critical error occurred: {e}. Exiting.")
    finally:
        logger.info("Shutting down chat interface...")
        # Attempt to clean up the LLM client if it has an aclose method
        if llm_service and hasattr(llm_service.client, 'aclose'):
            try:
                logger.info("Attempting to close LLM client...")
                await llm_service.client.aclose()
                logger.info("LLM client closed.")
            except Exception as e_close:
                logger.error(f"Error closing LLM client: {e_close}", exc_info=True)
        
        # Tavily MCP session is managed by its own asynccontextmanager and will close automatically.
        # No explicit cleanup for 'app' needed here as its components are managed by their creators
        # or the context managers (Tavily) or attempted cleanup (LLM).
        logger.info("Chat interface stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main_chat_loop())
    except KeyboardInterrupt:
        logger.info("Application terminated by KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True) 