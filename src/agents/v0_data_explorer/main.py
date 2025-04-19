import sys
import os
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
import argparse
from operator import add

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

def run_agent(query: str):
    """Runs the supervisor agent with the given query."""
    logger.info(f"Starting agent execution with query: '{query}'")
    
    try:
        # Create the supervisor agent application
        app = create_supervisor_agent()
        
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
    example_query = "How many tables are in the database?"
    # example_query = "What is the latest news about large language models?"
    
    # Or take query from command line arguments
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = example_query
        logger.info(f"No query provided, using default: '{user_query}'")

    run_agent(user_query)

if __name__ == "__main__":
    main() 