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

from langchain_core.messages import HumanMessage
from .state import AgentState
from .supervisor import create_supervisor_agent
from .tools import data_tools, web_tools  # Import the tool lists
from src.config.logger import logger

load_dotenv()

# Combine the tool lists
all_tools = data_tools + web_tools

# Make the function async
async def run_agent(query: str):
    """Runs the supervisor agent with the given query."""
    logger.info(f"Starting agent execution with query: '{query}'")
    
    try:
        # Create the supervisor agent application (await the async function)
        app = await create_supervisor_agent()
        
        # Format the input for the LangGraph application
        inputs = {"messages": [HumanMessage(content=query)]}
        
        # Invoke the agent
        # Stream events to see the flow (optional, can use .invoke for final result)
        # final_state = app.invoke(inputs)
        
        logger.info("Streaming agent execution steps...")
        for output in app.stream(inputs, stream_mode="values"):
            # output is the current state of the graph
            # you can inspect the state here if needed
            # logger.debug(f"Current State: {output}")
            pprint(output)
            print("---")
            
        # Get the final state after streaming is complete
        # Note: The stream itself consumes the iterator, 
        #       so invoke again or capture the last element if final state needed separately.
        # For simplicity, we'll just print the stream for now.
        # final_state = app.invoke(inputs)
        # logger.info("Final State:")
        # pprint(final_state)

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)

def main():
    # Example Usage:
    # Provide a sample query
    # example_query = "What were the total sales for product 'X' in the last quarter?"
    # example_query = "How many tables are in the database?"
    # example_query = "What is the latest news about large language models?"
    
    print("Starting Data Explorer Agent Chat Interface...")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not user_query:
                continue
                
            print("Agent: Processing...") # Indicate that the agent is working
            # Use asyncio.run to execute the async function for each query
            asyncio.run(run_agent(user_query))
            print("-" * 20) # Separator for clarity
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred in the chat loop: {e}", exc_info=True)
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    main() 