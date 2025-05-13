# Data Explorer Agent (v1)

This agent acts as a multi-agent system designed to answer user queries by orchestrating tasks between a web search agent and a data analysis team that interacts with a configured SQLite database.

## Architecture

The system employs a supervisor-worker pattern built using [LangGraph](https://python.langchain.com/docs/langgraph/):

1.  **Supervisor Agent (`supervisor.py`)**:
    *   Receives the initial user query.
    *   Analyzes the query to determine the appropriate worker agent or sequence of agents.
        *   `web_search_agent`: For general knowledge, current events, or web lookups.
        *   `data_analysis_team`: For questions requiring database interaction (schema lookup, SQL query generation, execution).
    *   **Orchestration**: If a query requires multiple steps (e.g., fetch data then search web based on results), the supervisor routes tasks sequentially, ensuring context is passed via message history.
    *   **Synthesis**: Receives results from worker(s) and synthesizes them into a final, coherent response for the user. The prompt for the supervisor guides it to combine information from all participating agents if multiple were involved.

2.  **Web Search Agent (`web_search_agent.py`)**:
    *   A ReAct-based agent responsible for performing web searches using available tools (e.g., Tavily Search).
    *   Receives tasks from the supervisor and returns search findings.

3.  **Data Analysis Team (`data_team.py`)**:
    *   A specialized LangGraph graph responsible for database interactions:
        *   **Schema or SQL Focus**: The team can either provide the database schema directly or generate and execute SQL queries.
        *   **`extract_schema_or_sql_node`**: A key node that determines if the agent's LLM output is schema text or an SQL query.
        *   **Schema Path**: If schema text is identified, it bypasses SQL validation/execution and is returned directly.
        *   **SQL Path**: If SQL is identified:
            *   **Generate SQL**: Uses an LLM (ReAct agent) to generate a SQLite query based on the user's natural language question and the database schema.
            *   **Validate SQL**: Uses an LLM to validate the generated SQL. Includes a retry mechanism (`MAX_SQL_RETRIES = 2`).
            *   **Execute SQL**: Executes the validated SQL query.
        *   **Format Response**: Prepares the schema, query results, or error messages for the supervisor.

## Key Components

*   **`main.py`**: Entry point for running the agent. Provides an interactive chat loop.
*   **`supervisor.py`**: Defines the main supervisor agent logic, multi-step orchestration, and response synthesis.
*   **`data_team.py`**: Defines the LangGraph workflow for the data analysis sub-agent, including schema/SQL differentiation.
*   **`web_search_agent.py`**: Defines the web search ReAct sub-agent.
*   **`state.py`**: Defines the shared `AgentState` TypedDict, including fields like `provided_schema_text`.
*   **`tools.py`**: Contains tool definitions (e.g., `get_database_schema`, `execute_sql_query`, web search tools).
*   **`config.py`**: Handles configuration, particularly asynchronous LLM client setup.

## Setup and Configuration

1.  **Environment Variables**: Uses a `.env` file in the project root.
    ```dotenv
    # Example .env content
    OPENAI_API_KEY="sk-YourOpenAIKey"
    TAVILY_API_KEY="tvly-YourTavilyKey" # For web search tool
    # Database configuration (used by tools.py and config.py)
    DB_TYPE="sqlite" # or other types if DatabaseManager supports them
    DB_NAME="./databases/dummy_db_clinical_trials/clinical_trials.db" # Path to SQLite file
    DB_SCHEMA_PATH="./databases/dummy_db_clinical_trials/schema_description.yaml" # Path to YAML schema description
    ```
2.  **Database**: The `data_analysis_team` uses a SQLite database. Paths are configured via environment variables.
3.  **Dependencies**: Install required Python packages (see `requirements.txt` if available, otherwise standard LangChain/LangGraph, OpenAI, Tavily, etc.).

## How to Run

Execute the agent from the project root directory (`multi-agent-data-explorer`):

```bash
python -m src.agents.v1_data_explorer.main
```

This will start an interactive chat session. You can then type your queries.

Example queries:

```
# Get database schema, then weather (multi-step)
Tell me what is the database schema using the data agent, then tell me what is the weather in montreal using the web search agent

# Get data from DB, then search web based on results (multi-step)
What are the mechanisms of action and targets from the database and what can we find about this in the web?

# Query the database (single agent step for data team)
How many patients are in the database?

# Ask a general question (single agent step for web search)
What is the latest news about AI?
```

The agent streams its internal logging to the console, and the final response is printed.

## Notes

*   The system is fully asynchronous (`asyncio`).
*   Error handling is implemented within agent graphs.
*   Logging is configured via `src.config.logger`.
*   The supervisor prompt has been enhanced to explicitly guide multi-step orchestration and final response synthesis.
*   The data team can now distinguish between requests for schema information and requests requiring SQL query generation, handling each appropriately. 