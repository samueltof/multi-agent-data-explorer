# Data Explorer Agent (v0)

This agent acts as a multi-agent system designed to answer user queries by either searching the web or interacting with a configured SQLite database.

## Architecture

The system employs a supervisor-worker pattern built using [LangGraph](https://python.langchain.com/docs/langgraph/):

1.  **Supervisor Agent (`supervisor.py`)**:
    *   Receives the initial user query.
    *   Analyzes the query to determine the appropriate worker agent:
        *   `web_search_agent`: For general knowledge, current events, or web lookups.
        *   `data_analysis_team`: For questions requiring database interaction (schema lookup, SQL query generation, execution).
    *   Routes the query to the selected worker.
    *   Receives the result from the worker and presents the final answer to the user.

2.  **Web Search Agent (`web_search_agent.py`)**:
    *   A dedicated agent (likely using LangChain agent tools) that performs web searches to answer queries. (Details are within its own implementation, managed by the supervisor).

3.  **Data Analysis Team (`data_team.py`)**:
    *   A specialized LangGraph graph responsible for database interactions:
        *   **Fetch Schema**: Retrieves the schema of the connected SQLite database.
        *   **Generate SQL**: Uses an LLM to generate a SQLite query based on the user's natural language question and the database schema.
        *   **Validate SQL**: Uses an LLM to validate the generated SQL for correctness and appropriateness. Includes a retry mechanism (`MAX_SQL_RETRIES = 2`) if validation fails.
        *   **Execute SQL**: Executes the validated SQL query against the database.
        *   **Format Response**: Prepares the query results or error messages for the supervisor.

## Key Components

*   **`main.py`**: Entry point for running the agent. Parses command-line arguments for the query.
*   **`supervisor.py`**: Defines the main supervisor agent logic and routing.
*   **`data_team.py`**: Defines the LangGraph workflow for the data analysis sub-agent.
*   **`web_search_agent.py`**: Defines the web search sub-agent.
*   **`state.py`**: Defines the shared `AgentState` TypedDict used for passing information within the LangGraph workflows.
*   **`tools.py`**: Contains the definitions or wrappers for tools used by the agents (e.g., `get_database_schema`, `execute_sql_query`, web search tools).
*   **`config.py`**: Handles configuration, particularly LLM client setup (async).

## Setup and Configuration

1.  **Environment Variables**: The agent relies on environment variables loaded via a `.env` file in the project root. Ensure this file contains necessary API keys (e.g., for the LLM provider like OpenAI or Anthropic) and potentially database connection details if not hardcoded in the tools.
    ```dotenv
    # Example .env content
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="..." # If using Tavily for web search
    # Add other necessary variables (e.g., database path if needed by tools)
    SQLITE_DB_PATH="path/to/your/database.db"
    ```
2.  **Database**: The `data_analysis_team` is configured to interact with a SQLite database. The specific database file path needs to be accessible and correctly configured, likely within the `tools.py` implementation or via an environment variable used by the tools.
3.  **Dependencies**: Install required Python packages. (A `requirements.txt` file would be beneficial here, but assuming standard LangChain/LangGraph dependencies).

## How to Run

Execute the agent from the project root directory (`multi-agent-data-explorer`):

```bash
python src/agents/v0_data_explorer/main.py "Your query about data or the web"
```

For example:

```bash
# Query the database
python src/agents/v0_data_explorer/main.py "How many users are in the database?"

# Ask a general question
python src/agents/v0_data_explorer/main.py "What is the latest news about AI?"
```

The agent will stream its internal state and steps to the console during execution.

## Notes

*   The agent uses asynchronous operations (`asyncio`) for potentially improved performance, especially around LLM calls and potentially tool usage.
*   Error handling is implemented within the graph nodes to manage issues during schema fetching, SQL generation/validation, and execution.
*   Logging is configured via `src.config.logger`. 