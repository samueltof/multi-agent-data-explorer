from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Dict, Any, Optional, Literal
from src.config.logger import logger

from .state import AgentState
from .tools import (
    retrieve_candidates,
    rank_tables,
    select_fields,
    plan_query,
    validate_query,
    execute_sql,
    get_database_schema,
)

# Constants
MAX_ITERATIONS = 3

# --- Node Functions ---

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Running Retrieve Candidates Node ---")
    # Initialize state variables locally, don't modify input state directly here
    updates: Dict[str, Any] = {
        "current_iteration": 0,
        "error": None,
        "sql_results": None,
        "sql_query": None,
        "plan_steps": None,
        "ranked_fields": None,
        "ranked_tables": None,
        "candidate_contexts": [],
        "validation_errors": []
    }
    
    # Propagate the user query from messages if not set directly
    user_query = state.get("user_query")
    if not user_query and state.get("messages"):
        last_human_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
        if last_human_message:
            user_query = last_human_message.content
            updates["user_query"] = user_query # Store it in updates dictionary
        else:
            logger.error("Retrieve Node: Could not find user query in messages.")
            updates["error"] = "Could not identify user query."
            return updates # Return updates
    elif user_query:
        updates["user_query"] = user_query # Ensure it's in updates if passed in state

    current_user_query = updates.get("user_query")
    if not current_user_query:
        logger.error("Retrieve Node: user_query is missing.")
        updates["error"] = "User query is missing."
        return updates # Return updates

    # Call the refactored tool with only the required input
    tool_result = retrieve_candidates.invoke({"user_query": current_user_query}) 

    # Update state based on the tool's return value
    if tool_result.get("error"):
        updates["error"] = tool_result["error"]
    updates["candidate_contexts"] = tool_result.get("candidate_contexts", [])
    
    # Return only the dictionary of updates
    return updates

def rank_tables_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Running Rank Tables Node ---")
    if state.get("error"): return {"error": state.get("error")} # Propagate error
    # Extract required inputs for the tool
    user_query = state.get("user_query")
    candidate_contexts = state.get("candidate_contexts", [])
    if user_query is None:
        logger.error("Rank Tables Node: user_query is missing in state.")
        return {"error": "User query missing for table ranking."} # Return error update
    
    # Call the refactored tool with specific arguments
    tool_result = rank_tables.invoke({
        "user_query": user_query,
        "candidate_contexts": candidate_contexts
    })
    
    # Return only the dictionary of updates
    return {"ranked_tables": tool_result}

def select_fields_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Running Select Fields Node ---")
    if state.get("error"): return {"error": state.get("error")} # Propagate error
    # Extract required inputs for the tool
    user_query = state.get("user_query")
    candidate_contexts = state.get("candidate_contexts", [])
    ranked_tables = state.get("ranked_tables", [])
    if user_query is None:
        logger.error("Select Fields Node: user_query is missing in state.")
        return {"error": "User query missing for field selection."} # Return error update
    
    # Call the refactored tool with specific arguments
    tool_result = select_fields.invoke({
        "user_query": user_query,
        "candidate_contexts": candidate_contexts,
        "ranked_tables": ranked_tables
    })
    
    # Return only the dictionary of updates
    return {"ranked_fields": tool_result}

def plan_query_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Running Plan Query Node ---")
    if state.get("error"): return {"error": state.get("error")} # Propagate error
    # Extract required inputs for the tool
    user_query = state.get("user_query", "")
    ranked_tables = state.get("ranked_tables", [])
    ranked_fields = state.get("ranked_fields", [])
    candidate_contexts = state.get("candidate_contexts", [])

    # Add checks for potentially problematic None values passed to tools expecting lists
    if ranked_tables is None:
        logger.warning("plan_query_node: ranked_tables is None, defaulting to empty list.")
        ranked_tables = []
    if ranked_fields is None:
        logger.warning("plan_query_node: ranked_fields is None, defaulting to empty list.")
        ranked_fields = []
    if candidate_contexts is None: # Also check candidate_contexts
        logger.warning("plan_query_node: candidate_contexts is None, defaulting to empty list.")
        candidate_contexts = []
    if not user_query:
        logger.error("plan_query_node: user_query is missing or empty in state.")
        # Return error update if query is missing
        return {"error": "User query missing for query planning."}
    
    # Call the refactored tool with specific arguments, including candidate_contexts
    tool_result = plan_query.invoke({
        "user_query": user_query,
        "ranked_tables": ranked_tables,
        "ranked_fields": ranked_fields,
        "candidate_contexts": candidate_contexts # Pass candidate_contexts
    })
    
    # Prepare updates dictionary
    updates: Dict[str, Any] = {
        "plan_steps": tool_result.get("plan_steps", []),
        "sql_query": tool_result.get("sql_query", None),
        "validation_errors": [], # Reset errors before validation
        "current_iteration": state.get("current_iteration", 0) + 1 # Increment iteration
    }
    return updates # Return updates dictionary

def validate_query_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Running Validate Query Node ---")
    if state.get("error"): return {"error": state.get("error")} # Propagate error
    # Extract required input for the tool
    sql_query = state.get("sql_query", None)
    user_query = state.get("user_query", "") # Extract user_query
    candidate_contexts = state.get("candidate_contexts", []) # Extract candidate_contexts

    # Add checks similar to plan_query_node
    if candidate_contexts is None:
        logger.warning("validate_query_node: candidate_contexts is None, defaulting to empty list.")
        candidate_contexts = []
    if not user_query:
        logger.error("validate_query_node: user_query is missing or empty in state.")
        # Return error update if query is missing
        return {"error": "User query missing for validation."}

    # Call the refactored tool with specific arguments, including user_query and candidate_contexts
    tool_result = validate_query.invoke({
        "sql_query": sql_query, 
        "user_query": user_query, 
        "candidate_contexts": candidate_contexts
    })
    
    # Return only the dictionary of updates
    return {"validation_errors": tool_result}

def execute_query_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- Running Execute Query Node ---")
    updates: Dict[str, Any] = {"sql_results": None, "error": None} # Initialize updates
    if state.get("error"): 
        updates["error"] = state.get("error")
        return updates # Propagate error
        
    sql_query = state.get("sql_query")

    if sql_query and "-- Could not determine" not in sql_query and "-- Error:" not in sql_query:
        db_path = state.get("db_connection_string", "database/Chinook.db")
        try:
            result = execute_sql.invoke({"query": sql_query, "db_path": db_path})
            updates["sql_results"] = result # Store result in updates
            if "Error executing SQL" in result:
                logger.warning(f"SQL Execution Returned Error: {result}")
            else:
                logger.info(f"SQL Execution Successful. Results stored.")
        except Exception as e:
             error_msg = f"Exception during SQL execution node: {e}"
             updates["error"] = error_msg # Store error in updates
             logger.error(error_msg)
    elif state.get("sql_query") is None:
        error_msg = "No SQL query was generated in the planning step."
        updates["error"] = error_msg # Store error in updates
        logger.error(error_msg)
    else:
        error_msg = "SQL query generation failed, cannot execute."
        updates["error"] = error_msg # Store error in updates
        logger.error(error_msg)

    return updates # Return updates dictionary

def format_final_response_node(state: AgentState) -> Dict[str, Any]:
    """ Formats the final response message based on graph outcome."""
    logger.info("--- Formatting Final Response Node ---")
    error = state.get("error")
    sql_results = state.get("sql_results")
    sql_query = state.get("sql_query")
    user_query = state.get("user_query", "your request")

    generated_sql_str = f"Generated SQL:\n```sql\n{sql_query}\n```\n" if sql_query else ""
    # Corrected f-string termination and content for results
    results_str = f"Execution Results:\n```\n{sql_results}\n```" if sql_results else ""

    if error:
        final_message_content = f"I encountered an issue while processing '{user_query}'. Error: {error}"
        logger.error(f"Final Response: Error - {error}")
    elif sql_results and "Error executing SQL" in sql_results:
        final_message_content = (
            f"I tried executing the query for '{user_query}', but the database returned an error.\n"
            f"{generated_sql_str}"
            f"Database Error:\n```\n{sql_results}\n```"
        )
        logger.warning(f"Final Response: Execution Error - {sql_results}")
    elif sql_results:
        final_message_content = (
            f"Okay, I processed '{user_query}'.\n"
            f"{generated_sql_str}"
            f"{results_str}"
        )
        logger.info(f"Final Response: Success - Results: {str(sql_results)[:100]}...")
    else:
        final_message_content = f"I finished processing '{user_query}', but couldn't retrieve the results. Please check the query or logs.\n{generated_sql_str}"
        logger.warning("Final Response: No results or error recorded.")

    final_message = AIMessage(content=final_message_content, name="data_team_response")
    current_messages = state.get("messages", [])
    # state["messages"] = current_messages + [final_message]
    # Return the message update for the operator.add reducer
    return {"messages": [final_message]}

# --- Conditional Edge Logic ---

def should_correct_query(state: AgentState) -> str:
    """Determines whether to attempt correction or end due to errors/max iterations."""
    logger.info("--- Checking Query Validity and Iteration Count ---")
    if state.get("error"):
        logger.warning("Upstream error detected, proceeding to format response.")
        return "format_response"

    errors = state.get("validation_errors", [])
    current_iter = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", MAX_ITERATIONS)

    if errors:
        logger.warning(f"Validation Errors Found: {errors}")
        if current_iter < max_iter:
            logger.info(f"Attempting correction (Iteration {current_iter + 1}/{max_iter}).")
            return "plan_query"
        else:
            logger.error(f"Max correction attempts ({max_iter}) reached. Aborting SQL generation.")
            state["error"] = f"Failed to generate valid SQL after {max_iter} attempts. Last errors: {errors}"
            return "format_response"
    else:
        logger.info("Query validation successful.")
        return "execute_query"

# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rank_tables", rank_tables_node)
workflow.add_node("select_fields", select_fields_node)
workflow.add_node("plan_query", plan_query_node)
workflow.add_node("validate", validate_query_node)
workflow.add_node("execute_query", execute_query_node)
workflow.add_node("format_response", format_final_response_node)

# Define edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rank_tables")
workflow.add_edge("rank_tables", "select_fields")
workflow.add_edge("select_fields", "plan_query")
workflow.add_edge("plan_query", "validate")
workflow.add_edge("execute_query", "format_response")

# Conditional edge from validation
workflow.add_conditional_edges(
    "validate",
    should_correct_query,
    {
        "plan_query": "plan_query",
        "execute_query": "execute_query",
        "format_response": "format_response",
    },
)

# Define finish point
workflow.add_edge("format_response", END)

# Compile the graph
data_team_graph = workflow.compile()
# Assign a name for the supervisor
data_team_graph.name = "data_analysis_team"

logger.info("Data Team LangGraph workflow compiled.") 