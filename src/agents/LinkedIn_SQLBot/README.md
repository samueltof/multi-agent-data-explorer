# LinkedIn SQL Bot Agent attempt

This agent acts as a multi-agent system designed to answer user queries by either searching the web or interacting with a configured database, inspired by architectures like LinkedIn's SQL Bot.

## Architecture (LinkedIn-Inspired)

The system employs a supervisor-worker pattern built using [LangGraph](https://python.langchain.com/docs/langgraph/):

1.  **Supervisor (`supervisor.py`)**: 
    *   Receives the initial user query.
    *   Analyzes the query to determine the appropriate worker agent:
        *   `web_search_agent`: For general knowledge, current events, or web lookups.
        *   `data_team_graph`: For questions requiring database interaction.
    *   Routes the query to the selected worker/graph.
    *   Receives the final state from the worker/graph and presents the result (or error) to the user.

2.  **Web Search Agent (`web_search_agent.py`)**:
    *   A dedicated agent (likely using LangChain agent tools) that performs web searches to answer queries. (Managed by the supervisor).

3.  **Data Team Graph (`data_team.py`)**:
    *   A specialized LangGraph workflow implementing a multi-step process for database interactions, inspired by LinkedIn's approach:
        *   **Retrieve Candidates (`retrieve_node`)**: (Placeholder) Retrieves candidate table/column contexts potentially relevant to the user query. *Future: Implement using vector stores (e.g., FAISS) and schema embeddings.*
        *   **Rank Tables (`rank_tables_node`)**: (Placeholder) Ranks the candidate tables based on relevance using an LLM. *Future: Implement LLM call with context.*
        *   **Select Fields (`select_fields_node`)**: (Placeholder) Selects relevant fields from the ranked tables using an LLM. *Future: Implement LLM call with context.*
        *   **Plan Query (`plan_query_node`)**: (Placeholder) Generates a SQL query plan and the SQL query itself using an LLM, based on selected tables/fields. *Future: Implement LLM call for planning and generation.*
        *   **Validate Query (`validate_node`)**: (Placeholder) Validates the generated SQL query (e.g., syntax check, `EXPLAIN`). *Future: Implement robust validation, potentially using `EXPLAIN` via `execute_sql` tool.*
        *   **Correction Loop**: If validation fails, the graph attempts to replan/regenerate the query up to `MAX_ITERATIONS` times.
        *   **Execute Query (`execute_query_node`)**: Executes the validated SQL query against the database using the `execute_sql` tool.
        *   **Format Response (`format_final_response_node`)**: Prepares the query results or error messages for the supervisor.

## Key Components

*   **`main.py`**: Entry point for running the agent. Sets up logging and runs the main interaction loop.
*   **`supervisor.py`**: Defines the `Supervisor` class responsible for routing and invoking agents/graphs.
*   **`data_team.py`**: Defines the LangGraph workflow (`data_team_graph`) for the data analysis process.
*   **`web_search_agent.py`**: Defines the web search agent.
*   **`state.py`**: Defines the shared `AgentState` TypedDict used for passing information within the LangGraph workflows.
*   **`tools.py`**: Contains the definitions for tools used by the agents (e.g., `execute_sql`, `get_database_schema`, web search tools, and the placeholder functions for the data team workflow).
*   **`config.py`**: Handles configuration, particularly LLM client setup.

## Setup and Configuration

1.  **Environment Variables**: The agent relies on environment variables loaded via a `.env` file in the project root. Ensure this file contains necessary API keys (e.g., `OPENAI_API_KEY`, `TAVILY_API_KEY`).
    ```dotenv
    # Example .env content
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="..."
    # Database path used by default in tools
    # SQLITE_DB_PATH="database/Chinook.db"
    ```
2.  **Database**: The `data_team_graph` interacts with a database (defaulting to `database/Chinook.db` in the placeholder tools/nodes). Ensure this path is correct or configure the `db_connection_string` in the state if needed.
3.  **Dependencies**: Install required Python packages. A `requirements.txt` file should include `langgraph`, `langchain`, `langchain-openai`, `loguru`, `python-dotenv`, `sqlite3` (usually built-in), `pandas` (if used by tools), `tavily-python`.

## How to Run

Execute the agent from the project root directory (`multi-agent-data-explorer`):

```bash
python -m src.agents.v1_data_explorer.main 
```

The application will start an interactive loop where you can enter queries.

Example queries:

```text
Enter your query (or type 'exit' to quit): Show me the first 5 customers by email.
Enter your query (or type 'exit' to quit): What is LangGraph?
Enter your query (or type 'exit' to quit): exit
```

## Notes

*   This version (v1) refactors the data interaction logic into a multi-step graph inspired by production systems like LinkedIn's SQL Bot.
*   **Placeholders**: Key steps in the `data_team_graph` (retrieval, ranking, planning, validation) currently use simplified placeholder logic. Future work involves implementing these with vector stores and LLM calls.
*   The agent uses asynchronous operations (`asyncio`) in `main.py`.
*   Error handling is implemented within the graph nodes and conditional edges.
*   Logging is configured via `src.config.logger`. 