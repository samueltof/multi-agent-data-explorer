"""
Interactive demo runner for the multi-agent network.

This script allows the user to interact with the multi-agent travel advisory system.
"""

import sys
import os

# Add the parent directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demos.travel_multi_agent_network.main import create_multi_agent_graph, pretty_print_messages, get_api_key, load_env_vars


def main():
    """Run the interactive demo."""
    # Load environment variables from .env file before anything else
    load_env_vars()
    
    print("\n" + "="*60)
    print("Multi-Agent Travel Advisory System".center(60))
    print("="*60)
    print("\nThis system uses two agents:")
    print("1. Travel Advisor: Recommends travel destinations")
    print("2. Hotel Advisor: Recommends hotels for specific destinations")
    print("\nThe agents can hand off to each other when needed.")
    print("\nType 'exit' to quit the demo.")
    print("="*60)

    # Get API key and create the multi-agent graph
    api_key = get_api_key()
    graph = create_multi_agent_graph(api_key)
    
    while True:
        user_query = input("\n\nWhat would you like to know about travel? ")
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("\nThank you for using the Multi-Agent Travel Advisory System!")
            break
        
        print("\nProcessing your query...\n")
        
        # Run the graph with the user query
        try:
            result = graph.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_query,
                        }
                    ]
                }
            )
            
            # Print the final result
            print("\nResult:")
            pretty_print_messages(result)
            
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 