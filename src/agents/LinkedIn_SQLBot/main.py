import sys
import os
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
import argparse
import asyncio # Import asyncio
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from src.config.logger import logger

from src.config.settings import Settings # Keep for config loading

# New imports for refactored structure
from .state import AgentState
# Import the graph getter function instead of the Supervisor class
from .supervisor import get_supervisor_graph # UPDATED

load_dotenv()

# Removed old run_agent function
# async def run_agent(query: str): ...

async def main():
    """Main asynchronous function to run the multi-agent system."""
    settings = Settings()
    logger.info("Starting Multi-Agent Data Explorer v1...")

    # Get the compiled supervisor graph (needs await)
    supervisor_graph = await get_supervisor_graph() # UPDATED

    # Main interaction loop
    while True:
        try:
            user_input = input("Enter your query (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                logger.info("Exiting application.")
                break

            # Prepare initial state for the graph
            initial_state = {"messages": [HumanMessage(content=user_input)]}
            
            # Invoke the supervisor graph
            logger.info(f"--- Invoking Supervisor Graph for query: '{user_input}' ---")
            # Add configuration like recursion limit
            final_state = supervisor_graph.invoke(initial_state, config={"recursion_limit": 25})
            logger.info(f"--- Supervisor Graph Invocation Complete ---")

            # Extract and print the final response
            if final_state.get("messages"):
                final_response_message = final_state["messages"][-1]
                if hasattr(final_response_message, 'content'):
                    print(f"\nFinal Answer:\n{final_response_message.content}")
                else:
                    # Handle cases where the last message might not have content (e.g., ToolMessage)
                    print(f"\nFinal State Message (no content): {final_response_message}")
            else:
                print("\nWorkflow finished, but no messages found in the final state.")


        except EOFError:
             logger.info("Exiting application due to EOF.")
             break
        except KeyboardInterrupt:
            logger.info("Exiting application due to KeyboardInterrupt.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}. Please check logs.")
            # break

if __name__ == "__main__":
    # Load dotenv before running main, typically done at the top level
    # Imports moved inside __main__ block to avoid potential global namespace issues
    # if they were needed elsewhere, they should be at the top.
    from dotenv import load_dotenv
    import asyncio # Added asyncio import here as it's used below
    load_dotenv()

    asyncio.run(main())