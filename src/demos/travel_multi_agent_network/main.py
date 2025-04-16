"""
Multi-agent network with React agents.

This demo implements a travel assistant system with two agents:
1. Travel Advisor: Recommends travel destinations
2. Hotel Advisor: Recommends hotels for specific destinations

The agents can hand off to each other when needed.
"""

import os
import random
import getpass
from typing import Annotated, Literal
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

load_dotenv()

def get_api_key():
    """Get API key from environment or prompt user."""
    # First load from .env if possible
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nANTHROPIC_API_KEY environment variable not found.")
        api_key = getpass.getpass("Please enter your Anthropic API key: ")
        if not api_key:
            raise ValueError("An Anthropic API key is required to run this demo.")
        # Set it for this session
        os.environ["ANTHROPIC_API_KEY"] = api_key
    return api_key


# Tools for the agents
@tool
def get_travel_recommendations():
    """Get recommendation for travel destinations"""
    return random.choice(["aruba", "turks and caicos"])


@tool
def get_hotel_recommendations(location: Literal["aruba", "turks and caicos"]):
    """Get hotel recommendations for a given destination."""
    return {
        "aruba": [
            "The Ritz-Carlton, Aruba (Palm Beach)",
            "Bucuti & Tara Beach Resort (Eagle Beach)"
        ],
        "turks and caicos": ["Grace Bay Club", "COMO Parrot Cay"],
    }[location]


# Helper function to create handoff tools
def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            # navigate to another agent node in the PARENT graph
            goto=agent_name,
            graph=Command.PARENT,
            # This is the state update that the agent `agent_name` will see when it is invoked.
            # We're passing agent's FULL internal message history AND adding a tool message to make sure
            # the resulting chat history is valid.
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent


def create_multi_agent_graph(api_key=None):
    """Create and return the multi-agent graph.
    
    Args:
        api_key (str, optional): The Anthropic API key. If None, will try to get from environment.
    """
    # Get API key if not provided
    if not api_key:
        api_key = get_api_key()
    
    try:
        # Initialize the model with the API key - trying claude-3-sonnet-20240229
        model = ChatAnthropic(model="claude-3-sonnet-20240229", anthropic_api_key=api_key)
        print("Successfully initialized ChatAnthropic with the provided API key")
    except Exception as e:
        print(f"Error initializing ChatAnthropic: {e}")
        print("\nTry getting a new API key from https://console.anthropic.com/keys")
        raise

    # Define travel advisor ReAct agent
    travel_advisor_tools = [
        get_travel_recommendations,
        make_handoff_tool(agent_name="hotel_advisor"),
    ]
    travel_advisor = create_react_agent(
        model,
        travel_advisor_tools,
        prompt=(
            "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
            "If you need hotel recommendations, ask 'hotel_advisor' for help. "
            "You MUST include human-readable response before transferring to another agent."
        ),
    )

    def call_travel_advisor(
        state: MessagesState,
    ) -> Command[Literal["hotel_advisor", "__end__"]]:
        # Invoke the ReAct agent with the full history of messages in the state
        return travel_advisor.invoke(state)

    # Define hotel advisor ReAct agent
    hotel_advisor_tools = [
        get_hotel_recommendations,
        make_handoff_tool(agent_name="travel_advisor"),
    ]
    hotel_advisor = create_react_agent(
        model,
        hotel_advisor_tools,
        prompt=(
            "You are a hotel expert that can provide hotel recommendations for a given destination. "
            "If you need help picking travel destinations, ask 'travel_advisor' for help. "
            "You MUST include human-readable response before transferring to another agent."
        ),
    )

    def call_hotel_advisor(
        state: MessagesState,
    ) -> Command[Literal["travel_advisor", "__end__"]]:
        return hotel_advisor.invoke(state)

    # Create the graph
    builder = StateGraph(MessagesState)
    builder.add_node("travel_advisor", call_travel_advisor)
    builder.add_node("hotel_advisor", call_hotel_advisor)
    # Always start with the travel advisor
    builder.add_edge(START, "travel_advisor")
    # Add END edge for both agents
    builder.add_edge("travel_advisor", END)
    builder.add_edge("hotel_advisor", END)

    return builder.compile()


def pretty_print_messages(chunk):
    """Helper function to print messages in a readable format."""
    if "messages" not in chunk:
        print("No messages in chunk")
        return
    
    for message in chunk["messages"]:
        # Check if message is a dictionary or LangChain message object
        if isinstance(message, dict):
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            name = message.get("name", "")
        elif hasattr(message, 'type') and hasattr(message, 'content'):
            role = message.type.upper()
            content = message.content
            name = getattr(message, 'name', None)
        else:
            print(f"Warning: Unknown message format: {type(message)}")
            continue
        
        print(f"\n{'='*20} {role} {'='*20}")
        if name:
            print(f"Name: {name}\n")
        print(content)
        print(f"{'='*50}")

api_key = get_api_key()
graph = create_multi_agent_graph(api_key)

# if __name__ == "__main__":
#     # Get API key and create the multi-agent graph
#     api_key = get_api_key()
#     graph = create_multi_agent_graph(api_key)
    
#     # Example user query
#     user_query = "I want to go somewhere warm in the Caribbean. Pick one destination and give me hotel recommendations."
    
#     print("\n\nStarting multi-agent conversation (Streaming Mode)...\n")
#     print(f"User query: {user_query}\n")
    
#     # Use stream to see the step-by-step execution
#     # The input format for stream/invoke can be a dictionary or a list of tuples/messages
#     # Using the list format here for variety
#     inputs = {"messages": [( "user", user_query)]}
    
#     for chunk in graph.stream(inputs, stream_mode="values"):
#         # Print the content of the chunk (agent output)
#         # We use pretty_print_messages to format it nicely
#         pretty_print_messages(chunk)
    
#     print("\n\nConversation completed!") 