from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4o")

async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["/Users/samueltorres/Documents/Repos/research/multi-agent-data-explorer/src/demos/mcp_langgraph/mcp_servers/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        print(f"Math response: {math_response}")
        weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
        print(f"Weather response: {weather_response}")

if __name__ == "__main__":
    asyncio.run(main())