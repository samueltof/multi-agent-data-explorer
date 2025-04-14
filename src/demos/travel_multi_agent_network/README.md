# Multi-Agent Network Demo

This demo implements a travel advisory system using LangGraph's multi-agent architecture. The system consists of two cooperating agents:

1. **Travel Advisor**: Recommends travel destinations
2. **Hotel Advisor**: Recommends hotels for specific destinations

The agents can hand off to each other when needed, forming a fully connected network where each agent can communicate with any other agent.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your API keys:

   You need a valid Anthropic API key that starts with one of these prefixes: `sk-ant-`, `sk-`, or `gsk-`.
   
   You can get an API key from [Anthropic's Console](https://console.anthropic.com/keys).

   You have two options:

   a. Set the API key as an environment variable (recommended):
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

   b. Create a `.env` file in the project root with the following content:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

   c. The program will prompt you for the API key if it's not found using either method above.

## Troubleshooting

If you encounter authentication errors like `invalid x-api-key`, make sure:
1. You're using a valid API key from [Anthropic's Console](https://console.anthropic.com/keys)
2. Your key has the correct format (starting with `sk-ant-`, `sk-`, or `gsk-`)
3. Your account has sufficient credit/quota

## Running the Demo

There are several ways to run the demo:

### 1. Streaming Demo (Default)

Run the main demo script, which processes a query and shows the step-by-step streaming output:

```bash
python -m src.demos.multi_agent_network.main
```

### 2. Interactive Demo

Run the interactive demo that allows you to have a conversation with the agents:

```bash
python -m src.demos.multi_agent_network.run
```

### 3. Graph Visualization

Generate a visualization of the agent graph structure:

```bash
python -m src.demos.multi_agent_network.visualize
```

This will save a PNG image of the graph structure in the `output` directory.

## How It Works

The multi-agent system uses LangGraph's ReAct agents and handoff mechanisms. Each agent is implemented as a node in the graph, and they can communicate with each other using handoff tools.

### Key Components:

1. **Agent Nodes**: Each agent is a node in the graph.
2. **Handoff Tools**: Custom tools that allow agents to transfer control to other agents.
3. **ReAct Implementation**: Agents use ReAct (Reasoning and Acting) to decide when to use tools or hand off to other agents.

### Multi-Agent Communication Flow:

1. The user query is sent to the Travel Advisor agent.
2. If the Travel Advisor needs hotel information, it hands off to the Hotel Advisor.
3. The Hotel Advisor processes the request and may hand back to the Travel Advisor if needed.
4. The final answer is returned to the user.

## Extending the System

You can extend the system by:

1. Adding more specialized agents
2. Enhancing the tools available to each agent
3. Modifying the conversation flow and handoff rules

## Reference

This implementation is based on the LangGraph multi-agent network tutorial:
https://langchain-ai.github.io/langgraph/how-tos/multi-agent-network/ 