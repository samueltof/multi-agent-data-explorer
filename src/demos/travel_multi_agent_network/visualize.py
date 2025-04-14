"""
Visualization demo for the multi-agent network.

This script generates a visualization of the agent graph structure
and saves it as a PNG file for viewing.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demos.travel_multi_agent_network.main import create_multi_agent_graph, get_api_key, load_env_vars


def visualize_graph():
    """Generate and save a visualization of the agent graph."""
    # Load environment variables from .env file
    load_env_vars()
    
    # Get API key and create the multi-agent graph
    api_key = get_api_key()
    graph = create_multi_agent_graph(api_key)
    
    # Create the output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate the graph visualization as PNG
    png_data = graph.get_graph().draw_mermaid_png()
    
    # Save the PNG file
    output_path = output_dir / "multi_agent_graph.png"
    with open(output_path, "wb") as f:
        f.write(png_data)
    
    print(f"\nGraph visualization saved to: {output_path.absolute()}")
    print("You can open this file to see the structure of the multi-agent network.")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_env_vars()
    
    print("\n" + "="*60)
    print("Multi-Agent Graph Visualization".center(60))
    print("="*60)
    print("\nGenerating visualization of the multi-agent network...")
    
    visualize_graph() 