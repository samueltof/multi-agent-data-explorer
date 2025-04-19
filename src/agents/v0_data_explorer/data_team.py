from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers.json import JsonOutputParser
from typing import List, Dict, Any, Optional
from functools import partial

from .state import AgentState
from .tools import get_database_schema, execute_sql_query
from src.config.logger import logger

# Constants
MAX_SQL_RETRIES = 2

# --- Node to prepare input for the data team ---
def prepare_data_query_node(state: AgentState) -> AgentState:
    """Extracts the latest user query and resets data team state."""
    logger.info("DATA_TEAM: Preparing query...")
    
    # Find the latest HumanMessage from the history
    last_human_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    if last_human_message is None:
        # Handle the case where no HumanMessage is found (shouldn't happen in normal flow)
        error_msg = "No HumanMessage found in the state history."
        logger.error(f"DATA_TEAM: {error_msg}")
        return {"error_message": error_msg}

    # Log the found query
    logger.info(f"DATA_TEAM: Found user query: '{last_human_message.content}'")

    # Reset specific fields for a new run within the data team
    return {
        "natural_language_query": last_human_message.content,
        "schema": None,
        "generated_sql": None,
        "validation_status": None,
        "validation_feedback": None,
        "execution_result": None,
        "error_message": None,
        "sql_generation_retries": 0
    }

# --- Prompts ---
SQL_GENERATOR_PROMPT = PromptTemplate(
    template="""
You are interacting with a **SQLite** database.

Given the following database schema:
{schema}

And the user query:
{query}

Generate a valid **SQLite** SQL query that addresses the user's request. 

- Use `sqlite_master` or `sqlite_schema` for metadata queries if needed.
- Focus ONLY on generating the SQL.
{retry_feedback}

Respond ONLY with the raw SQL query, without any markdown formatting (like ```sql) or explanations.
SQL Query:""",
    input_variables=["schema", "query", "retry_feedback"]
)

# Using JSON output parser for structured validation feedback
SQL_VALIDATOR_PROMPT = PromptTemplate(
    template="""
You are validating a query for a **SQLite** database.

Database Schema:
{schema}

User Query:
{query}

Generated SQL Query:
{sql}

Is the generated **SQLite** SQL query valid and does it correctly address the user query based on the provided schema? 

Consider:
- **SQLite** SQL syntax correctness (e.g., usage of `sqlite_master` is valid for metadata).
- Table and column existence in the schema.
- Appropriateness of the query for the user's request.

Respond with JSON containing:
1. 'status': 'valid' or 'invalid'.
2. 'feedback': If invalid, provide specific reasons and suggestions for correction. If valid, provide a brief confirmation.

{format_instructions}
""",
    input_variables=["schema", "query", "sql"],
    partial_variables={"format_instructions": JsonOutputParser().get_format_instructions()},
)

# --- Graph Nodes ---

def fetch_schema_node(state: AgentState) -> AgentState:
    """Fetches the database schema using the tool."""
    logger.info("DATA_TEAM: Fetching schema...")
    try:
        schema_description = get_database_schema.invoke({})
        if "Error" in schema_description:
             raise ValueError(f"Schema fetching failed: {schema_description}")
        return {"schema": schema_description}
    except Exception as e:
        logger.error(f"DATA_TEAM: Error in fetch_schema_node: {e}")
        return {"error_message": f"Failed to fetch database schema: {str(e)}"}

def sql_generator_node(state: AgentState, llm_client: Any) -> AgentState:
    """Generates SQL based on the schema and user query."""
    logger.info("DATA_TEAM: Generating SQL...")
    if state.get("error_message"):
         return {}

    # Use passed llm_client
    # llm_service = await get_llm_async()
    llm = llm_client 
    user_query = state.get("natural_language_query")
    schema = state.get("schema")
    retry_feedback = state.get("validation_feedback", "") # Get feedback from previous validation if any
    
    prompt = SQL_GENERATOR_PROMPT.format(
        schema=schema,
        query=user_query,
        retry_feedback=f"\nPrevious attempt feedback: {retry_feedback}" if retry_feedback else ""
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_sql = response.content.strip()
        
        # Clean potential markdown formatting
        if raw_sql.startswith("```sql"):
            cleaned_sql = raw_sql[len("```sql"):].strip()
            if cleaned_sql.endswith("```"):
                cleaned_sql = cleaned_sql[:-len("```")].strip()
        elif raw_sql.startswith("```"):
             cleaned_sql = raw_sql[len("```"):].strip()
             if cleaned_sql.endswith("```"):
                 cleaned_sql = cleaned_sql[:-len("```")].strip()
        else:
            cleaned_sql = raw_sql
            
        # Basic check if LLM returned something that looks like SQL
        if not cleaned_sql or not any(kw in cleaned_sql.upper() for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]):
            raise ValueError(f"LLM did not return a valid SQL query structure after cleaning. Original: '{raw_sql}'")
        
        logger.info(f"DATA_TEAM: Generated SQL (Cleaned): {cleaned_sql}")
        return {"generated_sql": cleaned_sql, "validation_feedback": None} # Clear old feedback
    except Exception as e:
        logger.error(f"DATA_TEAM: Error generating SQL: {e}")
        return {"error_message": f"Failed to generate SQL: {str(e)}"}

def sql_validator_node(state: AgentState, llm_client: Any) -> AgentState:
    """Validates the generated SQL using an LLM."""
    logger.info("DATA_TEAM: Validating SQL...")
    if state.get("error_message") or not state.get("generated_sql"):
        return {}

    # Use passed llm_client
    # llm_service = await get_llm_async()
    llm = llm_client 
    parser = JsonOutputParser()
    
    prompt = SQL_VALIDATOR_PROMPT.format(
        schema=state["schema"],
        query=state["natural_language_query"],
        sql=state["generated_sql"]
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        validation_result = parser.parse(response.content)
        logger.info(f"DATA_TEAM: Validation result: {validation_result}")
        
        status = validation_result.get("status", "error").lower()
        feedback = validation_result.get("feedback", "No feedback provided.")

        return {"validation_status": status, "validation_feedback": feedback}
    except Exception as e:
        logger.error(f"DATA_TEAM: Error validating SQL: {e}")
        return {"validation_status": "error", "validation_feedback": f"Failed to parse validation response: {str(e)}"}

def sql_executor_node(state: AgentState) -> AgentState:
    """Executes the validated SQL query using the tool."""
    logger.info("DATA_TEAM: Executing SQL...")
    if state.get("error_message") or not state.get("generated_sql") or state.get("validation_status") != "valid":
        return {}
        
    try:
        execution_result = execute_sql_query.invoke({"query": state["generated_sql"]})
        logger.info(f"DATA_TEAM: Execution result: {execution_result}")
        return {"execution_result": execution_result}
    except Exception as e:
        logger.error(f"DATA_TEAM: Error executing SQL: {e}")
        return {"error_message": f"Failed during SQL execution: {str(e)}"}

def handle_error_node(state: AgentState) -> AgentState:
    """Handles errors encountered during the process."""
    logger.warning(f"DATA_TEAM: Handling error: {state.get('error_message')}")
    # Simple error handling: just pass the error message along
    # More sophisticated error handling could be added here
    return {"error_message": state.get("error_message", "An unspecified error occurred.")}
    
def format_final_response_node(state: AgentState) -> AgentState:
    """Formats the final response message for the supervisor."""
    logger.info("DATA_TEAM: Formatting final response...")
    error_message = state.get("error_message")
    execution_result = state.get("execution_result")
    query = state.get("natural_language_query", "the user's query") # Get original query for context

    if error_message:
        # Keep error messages direct
        final_message_content = f"I encountered an error trying to answer '{query}': {error_message}"
    elif execution_result:
        # Make success message more conversational and include the result directly
        final_message_content = f"Okay, I looked into '{query}'. Here's the result:\n\n{execution_result}"
    else:
        # Should not happen in normal flow if validation passes
        final_message_content = f"I finished processing '{query}', but no execution result was found or an error occurred before execution."

    # Append the final message to the main message list for the supervisor
    final_message = AIMessage(content=final_message_content, name="data_team_final_response")
    return {"messages": [final_message]} # Return as list to be added by operator.add

# --- Conditional Edges ---

def decide_after_validation(state: AgentState) -> str:
    """Decides the next step after SQL validation."""
    if state.get("error_message"): # Prioritize errors from previous steps
        return "handle_error"
    
    validation_status = state.get("validation_status")
    retries = state.get("sql_generation_retries", 0)

    if validation_status == "valid":
        logger.info("DATA_TEAM: SQL Valid. Proceeding to execution.")
        return "execute_sql"
    elif retries < MAX_SQL_RETRIES:
        logger.warning(f"DATA_TEAM: SQL Invalid. Retrying generation (Attempt {retries + 1}). Feedback: {state.get('validation_feedback')}")
        # Important: Increment retry count in the state update dictionary
        return "retry_sql_generation"
    else:
        logger.error(f"DATA_TEAM: SQL Invalid after {MAX_SQL_RETRIES} retries. Handling error.")
        # Set error message explicitly for the error handler
        state["error_message"] = f"SQL validation failed after {MAX_SQL_RETRIES} attempts. Last feedback: {state.get('validation_feedback')}"
        return "handle_error"

def check_for_errors(state: AgentState) -> str:
    """Checks if an error occurred in the preceding node."""
    if state.get("error_message"):
        return "handle_error"
    return "continue" # Default transition name

# --- Graph Definition ---

def create_data_team_graph(llm_client: Any):
    """Creates and compiles the LangGraph StateGraph for the data team."""
    workflow = StateGraph(AgentState)

    # Use partial to bind the llm_client to the node functions
    generate_sql_with_llm = partial(sql_generator_node, llm_client=llm_client)
    validate_sql_with_llm = partial(sql_validator_node, llm_client=llm_client)

    # Add nodes
    workflow.add_node("prepare_query", prepare_data_query_node) # New entry node
    workflow.add_node("fetch_schema", fetch_schema_node)
    workflow.add_node("generate_sql", generate_sql_with_llm) # Use partial function
    workflow.add_node("validate_sql", validate_sql_with_llm) # Use partial function
    workflow.add_node("execute_sql", sql_executor_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("format_response", format_final_response_node)

    # Set entry point
    workflow.set_entry_point("prepare_query") # Changed entry point

    # Add edges
    # Edge from prepare_query to fetch_schema
    workflow.add_conditional_edges(
        "prepare_query",
        check_for_errors, # Use the same error check
        {
            "handle_error": "handle_error",
            "continue": "fetch_schema"
        }
    )

    # Conditional edge after schema fetching to check for errors
    workflow.add_conditional_edges(
        "fetch_schema",
        check_for_errors,
        {
            "handle_error": "handle_error",
            "continue": "generate_sql" 
        }
    )
    
    # Edge from generation to validation
    # Add retry increment logic within the node or a separate modifying node if needed
    # Simple approach: handle retry increment within the conditional logic's target state update
    workflow.add_node("increment_retry", lambda state: {"sql_generation_retries": state.get("sql_generation_retries", 0) + 1})
    workflow.add_edge("increment_retry", "generate_sql") # Loop back to generate

    workflow.add_conditional_edges(
        "generate_sql",
        check_for_errors,
        {
            "handle_error": "handle_error",
            "continue": "validate_sql"
        }
    )

    # Conditional edge after validation
    workflow.add_conditional_edges(
        "validate_sql",
        decide_after_validation,
        {
            "execute_sql": "execute_sql",
            "retry_sql_generation": "increment_retry", # Go to increment node before regenerating
            "handle_error": "handle_error",
        },
    )

    # Edge after execution
    workflow.add_conditional_edges(
        "execute_sql",
        check_for_errors,
        {
            "handle_error": "handle_error",
            "continue": "format_response"
        }
    )
    
    # Edges to END
    workflow.add_edge("handle_error", "format_response") # Format error response before ending
    workflow.add_edge("format_response", END)

    # Compile the graph
    data_team_app = workflow.compile()
    # Add a name attribute for supervisor identification
    data_team_app.name = "data_analysis_team"
    return data_team_app 