# Multi-Agent Data Explorer

A collection of multi-agent system implementations for exploring and analyzing data.

## Demos

### Multi-Agent Network

A travel advisory system using LangGraph's multi-agent architecture. Two cooperating agents (Travel Advisor and Hotel Advisor) can hand off to each other when needed. The demo shows the step-by-step streaming output.

- [Go to Multi-Agent Network Demo](./src/demos/multi_agent_network/)

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your API keys as environment variables or in a `.env` file (see individual demo READMEs for details):

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Running the Demos

See the individual demo READMEs for specific instructions on how to run each demo.

## Features

- Multi-agent architectures with specialized agents
- Agent communication via handoffs
- ReAct agents for decision-making 
- Graph-based agent coordination

## References

- [LangGraph Multi-Agent Network Tutorial](https://langchain-ai.github.io/langgraph/how-tos/multi-agent-network/)