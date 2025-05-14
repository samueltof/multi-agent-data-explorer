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
    """
    Runs the supervisor agent for a single query and handles its lifecycle.
    Returns a dictionary containing the agent's final response ('response_content')
    and any generated SQL ('generated_sql').
    """
    logger.info(f"Starting agent execution with query: '{query}'")
    
    app = None
    final_state_snapshot = None
    response_content = "Agent did not produce a final response." # Default response
    generated_sql_output = None

    try:
        app = await create_supervisor_agent()
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
                    logger.info(f"Agent Final Response: {last_message.content}")
                    response_content = last_message.content
                else:
                    logger.info("Last message was not a displayable AIMessage or was a handoff.")
                    response_content = "Agent finished processing, but no displayable message was returned."
            else:
                logger.warning("Final state snapshot contained no messages.")
                response_content = "Agent processed the query but did not return any messages."
        else:
             logger.warning("No final state snapshot captured from the stream or messages key missing.")
             response_content = "Agent stream did not yield a final state with messages."
        
        # ---- ADDING DETAILED LOGGING HERE ----
        logger.info(f"run_agent_session - final_state_snapshot received by run_agent_session: {final_state_snapshot}")
        # ---- END OF ADDED LOGGING ----

        if final_state_snapshot:
            generated_sql_output = final_state_snapshot.get("generated_sql")
            if generated_sql_output:
                logger.info(f"Extracted SQL: {generated_sql_output}")
        
        return_value = {
            "response_content": response_content,
            "generated_sql": generated_sql_output
        }
        logger.info(f"run_agent_session returning (try): {type(return_value)} - {return_value}")
        return return_value

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return_value = {
            "response_content": f"Agent encountered an error: {e}",
            "generated_sql": None
        }
        logger.info(f"run_agent_session returning (except): {type(return_value)} - {return_value}")
        return return_value
    finally:
        if app is not None:
            logger.debug("run_agent_session finished, 'app' going out of scope.")
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
                
            print("Agent: Processing...") # Keep this for console, Streamlit will have its own
            # Each call to asyncio.run creates a new event loop, runs the coro, 
            # and closes the loop. This is generally good for isolating sessions.
            response_data = asyncio.run(run_agent_session(user_query))
            # Print the returned response - adapt for dict
            print(f"Agent Response: {response_data.get('response_content')}")
            if response_data.get('generated_sql'):
                print(f"Generated SQL: {response_data.get('generated_sql')}")
            
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