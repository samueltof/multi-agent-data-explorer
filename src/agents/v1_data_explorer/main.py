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

load_dotenv()

# Combine the tool lists - This might not be needed here if tools are managed by agents
# all_tools = data_tools + web_tools

async def run_agent_session(query: str):
    """Runs the supervisor agent for a single query and handles its lifecycle."""
    logger.info(f"Starting agent execution with query: '{query}'")
    
    app = None  # Initialize app to None for the finally block
    final_state_snapshot = None
    
    try:
        # Create the supervisor agent application. 
        # All async resources (LLM client, agents) should be created within this scope.
        app = await create_supervisor_agent()
        
        inputs = {"messages": [HumanMessage(content=query)]}
        
        logger.info("Streaming agent execution steps...")
        async for output in app.astream(inputs, stream_mode="values"):
            # pprint(output) # Keep this for detailed debugging if needed
            # print("---")
            final_state_snapshot = output 

        if final_state_snapshot and "messages" in final_state_snapshot:
            final_messages = final_state_snapshot["messages"]
            if final_messages:
                last_message = final_messages[-1]
                if (isinstance(last_message, AIMessage) and
                   not last_message.tool_calls and
                   not getattr(last_message.response_metadata, '__is_handoff_back', False)):
                    # Simplified logging for final response
                    logger.info(f"ALL Response: {final_messages}")
                    print("\nAgent Final Response:")
                    print(last_message.content)
                    print("-" * 30)
                else:
                    logger.info("Last message was not a displayable AIMessage or was a handoff.")
            else:
                logger.warning("Final state snapshot contained no messages.")
        else:
             logger.warning("No final state snapshot captured from the stream or messages key missing.")

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        print(f"\nAgent encountered an error: {e}\n")
    finally:
        # Attempt to hint at cleanup, though Python's GC should handle 'app' going out of scope.
        # If 'app' or underlying components (like an LLM client) need explicit async cleanup,
        # they should implement an __aexit__ or an async close() method.
        if app is not None:
            # If LangGraph's compiled app or the LLM client had an async close method:
            # if hasattr(app, 'aclose'):
            #     await app.aclose()
            # elif hasattr(app, 'llm_service_to_close') and hasattr(app.llm_service_to_close, 'aclose'):
            # await app.llm_service_to_close.aclose()
            logger.debug("run_agent_session finished, 'app' going out of scope.")
        del app # Explicitly delete to hint to GC
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
            # Each call to asyncio.run creates a new event loop, runs the coro, 
            # and closes the loop. This is generally good for isolating sessions.
            asyncio.run(run_agent_session(user_query))
            
        except KeyboardInterrupt:
            print("\nExiting due to KeyboardInterrupt...")
            break
        except Exception as e:
            logger.error(f"An error occurred in the chat loop: {e}", exc_info=True)
            # Avoid printing overly verbose errors directly to user in chat loop
            print("An unexpected error occurred. Please check logs or try again.")
        finally:
            # Any cleanup that needs to happen after each query, if not handled by asyncio.run scope
            pass
    
    logger.info("Chat interface stopped.")

if __name__ == "__main__":
    # Renamed main function to avoid confusion if we want to import run_agent_session elsewhere
    main_chat_loop() 