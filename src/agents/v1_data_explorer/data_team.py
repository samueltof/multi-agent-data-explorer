from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent # Import the prebuilt agent creator
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage # Keep BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import List, Dict, Any, Optional, Literal
from functools import partial
import re # Import regex for SQL extraction
from pydantic import BaseModel, Field
from src.config.logger import logger
from .state import AgentState
from .tools import get_database_schema, execute_sql_query

# Constants
MAX_SQL_RETRIES = 2
# Initial prompt message for the ReAct agent
REACT_AGENT_INITIAL_PROMPT = """
You are a data analyst expert tasked with generating SQLite SQL queries based on user requests.
You have access to a tool to fetch the database schema (`get_database_schema`).

Your goal is to generate a valid SQLite SQL query to answer the user's question.

1.  Analyze the user's query: {query}
2.  If you need the database schema to understand the tables and columns, use the `get_database_schema` tool.
3.  Once you have the necessary information (schema, if required), generate the SQLite SQL query.
4.  Respond ONLY with the final SQL query itself, without any introductory text, explanations, or markdown formatting like ```sql.

{retry_feedback}
"""

# --- Node to prepare input for the data team ---
def prepare_data_query_node(state: AgentState) -> Dict[str, Any]:
    """Extracts the latest user query and prepares the initial message for the ReAct agent."""
    logger.info("👨‍💻 DATA TEAM: Preparing query for ReAct agent...")

    last_human_message_content = None
    # Find the last human message to get the core query
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_human_message_content = msg.content
            break

    if last_human_message_content is None:
        error_msg = "No HumanMessage found in the state history to process."
        logger.error(f"👨‍💻 DATA TEAM: {error_msg}")
        return {"error_message": error_msg}

    logger.info(f"👨‍💻 DATA TEAM: Found user query: '{last_human_message_content}'")

    # Check for retry feedback
    retry_feedback = state.get("validation_feedback", "")
    retry_prompt_addition = f"\nRetry Feedback: You previously generated SQL that failed validation with the following feedback: '{retry_feedback}'. Please analyze this feedback and generate a corrected query." if retry_feedback else ""

    # Format the initial prompt for the ReAct agent
    initial_prompt = REACT_AGENT_INITIAL_PROMPT.format(
        query=last_human_message_content,
        retry_feedback=retry_prompt_addition
    )

    # Reset state fields for this run, but keep the initial message for the agent
    return {
        "messages": [HumanMessage(content=initial_prompt)], # Start agent with this prompt
        "natural_language_query": last_human_message_content, # Keep original query for context
        "schema": state.get("schema"), # Persist schema if available
        "generated_sql": None,
        "validation_status": None,
        "validation_feedback": None, # Clear feedback before agent runs
        "execution_result": None,
        "error_message": None,
        "sql_generation_retries": state.get("sql_generation_retries", 0) # Keep retry count from previous attempt if applicable
    }


# --- Node to Extract SQL from ReAct Agent Output ---
def extract_sql_node(state: AgentState) -> Dict[str, Any]:
    """Extracts the SQL query from the last message of the ReAct agent."""
    logger.info("👨‍💻 DATA TEAM: Extracting SQL from ReAct agent output...")
    if state.get("error_message"): # Pass through errors
        return {}

    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, AIMessage) and last_message.content:
        content = last_message.content.strip()
        # Simple extraction: assume the entire content is the SQL query
        # More robust: Use regex to find SQL block if agent adds ```sql markers despite prompt
        sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            extracted_sql = sql_match.group(1).strip()
            logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (from markdown block): {extracted_sql}")
        else:
            # Assume raw content is SQL if no markdown block found
             extracted_sql = content
             logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (raw content): {extracted_sql}")

        # Basic sanity check
        if extracted_sql and any(kw in extracted_sql.upper() for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]):
            return {"generated_sql": extracted_sql, "validation_feedback": None} # Clear feedback after successful generation attempt
        else:
            logger.error(f"👨‍💻 DATA TEAM: Failed to extract valid SQL structure from agent response: {content}")
            return {"error_message": f"Agent did not produce a valid SQL query structure. Final response: {content}"}
    else:
        logger.error(f"👨‍💻 DATA TEAM: No valid AIMessage content found from ReAct agent. Last message: {last_message}")
        return {"error_message": "ReAct agent did not return a final message with content."}


# --- SQL Validation and Execution Nodes (Largely Unchanged) ---

# Pydantic model for structured validation feedback
class SQLValidationResult(BaseModel):
    status: Literal["valid", "invalid", "error"] = Field(description="The validation status of the SQL query.")
    feedback: str = Field(description="Detailed feedback on the validation, including reasons for invalidity or confirmation of validity.")

SQL_VALIDATOR_PROMPT = PromptTemplate(
    template="""\
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
- Table and column existence in the schema (if schema is provided). If schema is missing, validate syntax only.
- Appropriateness of the query for the user's request.

Respond with the required structure.
""""",
    input_variables=["schema", "query", "sql"],
)

def sql_validator_node(state: AgentState, llm_client: Any) -> AgentState:
    """Validates the generated SQL using an LLM with structured output.
       If schema is not in state, attempts to find it from message history."""
    logger.info("👨‍💻 DATA TEAM: Validating SQL...")
    if state.get("error_message") or not state.get("generated_sql"):
        logger.warning("👨‍💻 DATA TEAM: Skipping validation due to prior error or missing SQL.")
        if not state.get("generated_sql") and not state.get("error_message"):
             state["error_message"] = "Agent failed to produce an SQL query."
        return {}

    schema = state.get("schema")
    schema_source = "state" # Track where schema came from for logging

    # If schema not in state, try to find it in message history from the tool
    if not schema:
        logger.warning("👨‍💻 DATA TEAM: Schema not found in state. Checking message history for schema tool results...")
        fetched_schema = None
        schema_tool_call_id = None
        messages = state.get("messages", [])

        # Iterate backwards to find the latest schema tool call ID
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "get_database_schema":
                        schema_tool_call_id = tc.get("id")
                        logger.debug(f"Found schema tool call ID in AIMessage: {schema_tool_call_id}")
                        break # Found the latest call
                if schema_tool_call_id:
                    break # Stop searching messages once call ID is found
        
        # If a call ID was found, find the corresponding ToolMessage result
        if schema_tool_call_id:
            for msg in reversed(messages): # Search again from the end for the result
                if isinstance(msg, ToolMessage) and msg.tool_call_id == schema_tool_call_id:
                    if "Error" not in msg.content and msg.content: # Basic check for error and non-empty
                        fetched_schema = msg.content
                        schema_source = "ToolMessage" # Update schema source
                        logger.info(f"👨‍💻 DATA TEAM: Found schema in ToolMessage history.")
                        logger.debug(f"Schema snippet: {fetched_schema[:100]}...") 
                        break # Found the result
                    else:
                        logger.warning(f"👨‍💻 DATA TEAM: Found ToolMessage for schema call {schema_tool_call_id}, but it contained an error or was empty: {msg.content}")
                        break # Stop searching, tool failed or returned nothing useful

        if fetched_schema:
            schema = fetched_schema
            # Optionally, we could update the state here, but validator should ideally not modify state.
            # state["schema"] = fetched_schema # Avoid doing this here
        else:
            logger.warning("👨‍💻 DATA TEAM: Schema not found in ToolMessages either. Proceeding with validation without schema.")
            schema = "Schema not available for validation." # Fallback message for prompt
            schema_source = "unavailable"
    else:
        logger.info("👨‍💻 DATA TEAM: Using schema found previously in state for validation.")

    # Proceed with validation using the determined schema
    llm = llm_client
    prompt_str = SQL_VALIDATOR_PROMPT.format(
        schema=schema, # Use the schema variable (from state or messages or fallback)
        query=state["natural_language_query"],
        sql=state["generated_sql"]
    )

    try:
        structured_llm = llm.with_structured_output(SQLValidationResult)
        logger.info(f"Attempting SQL validation (schema source: {schema_source})...")
        validation_result = structured_llm.invoke([HumanMessage(content=prompt_str)])
        logger.info(f"👨‍💻 DATA TEAM: Validation result: {validation_result}")
        status = validation_result.status
        feedback = validation_result.feedback
        return {"validation_status": status, "validation_feedback": feedback}
    except Exception as e:
        logger.error(f"👨‍💻 DATA TEAM: Error validating SQL: {e}")
        return {"validation_status": "error", "validation_feedback": f"Failed during validation: {str(e)}"}


def sql_executor_node(state: AgentState) -> AgentState:
    """Executes the validated SQL query using the tool."""
    logger.info("👨‍💻 DATA TEAM: Executing SQL...")
    if state.get("error_message") or not state.get("generated_sql") or state.get("validation_status") != "valid":
        if state.get("validation_status") != "valid":
            logger.warning(f"👨‍💻 DATA TEAM: Skipping execution because SQL validation status is '{state.get('validation_status')}'.")
        return {}

    try:
        execution_result = execute_sql_query.invoke({"query": state["generated_sql"]})
        logger.info(f"👨‍💻 DATA TEAM: Execution result obtained.")
        logger.debug(f"👨‍💻 DATA TEAM: Execution result sample: {str(execution_result)[:200]}...")
        return {"execution_result": execution_result}
    except Exception as e:
        logger.error(f"👨‍💻 DATA TEAM: Error executing SQL: {e}")
        return {"error_message": f"Failed during SQL execution: {str(e)}"}


def handle_error_node(state: AgentState) -> AgentState:
    """Handles errors encountered during the process."""
    error_msg = state.get('error_message', "An unspecified error occurred.")
    logger.warning(f"👨‍💻 DATA TEAM: Handling error: {error_msg}")
    return {"error_message": error_msg}


def format_final_response_node(state: AgentState) -> AgentState:
    """Formats the final response message for the supervisor."""
    logger.info("👨‍💻 DATA TEAM: Formatting final response...")
    error_message = state.get("error_message")
    execution_result = state.get("execution_result")
    query = state.get("natural_language_query", "your query")

    if error_message:
        final_message_content = f"I encountered an error trying to answer '{query}': {error_message}"
    elif execution_result:
        final_message_content = f"Okay, I looked into '{query}'. Here's the result:\n\n{execution_result}"
    else:
        final_message_content = f"I tried processing '{query}', but couldn't successfully execute a query. The final validation feedback was: {state.get('validation_feedback', 'No specific feedback available.')}"

    final_message = AIMessage(content=final_message_content, name="data_team_final_response")
    return {"messages": [final_message]}


# --- Conditional Edges ---

def check_sql_extraction(state: AgentState) -> str:
    """Checks if SQL was successfully extracted after the ReAct agent."""
    if state.get("error_message"):
        logger.error(f"👨‍💻 DATA TEAM: Error detected after SQL extraction attempt: {state['error_message']}")
        return "handle_error"
    elif state.get("generated_sql"):
        logger.info("👨‍💻 DATA TEAM: SQL extracted successfully. Proceeding to validation.")
        return "validate_sql"
    else:
         # Should be caught by extract_sql_node setting error_message, but as a fallback
         logger.error("👨‍💻 DATA TEAM: SQL not found after extraction node, but no error message set. Handling as error.")
         state["error_message"] = "SQL extraction failed unexpectedly."
         return "handle_error"


def decide_after_validation(state: AgentState) -> str:
    """Decides the next step after SQL validation."""
    if state.get("error_message"):
        logger.error(f"👨‍💻 DATA TEAM: Error detected before deciding after validation: {state['error_message']}")
        return "handle_error"

    validation_status = state.get("validation_status")
    retries = state.get("sql_generation_retries", 0)

    if validation_status == "valid":
        logger.info("👨‍💻 DATA TEAM: SQL Valid. Proceeding to execution.")
        return "execute_sql"
    elif retries < MAX_SQL_RETRIES:
        logger.warning(f"👨‍💻 DATA TEAM: SQL Invalid. Retrying generation (Attempt {retries + 1}). Feedback: {state.get('validation_feedback')}")
        # Increment happens in the 'increment_retry' node now
        return "retry_sql_generation"
    else:
        logger.error(f"👨‍💻 DATA TEAM: SQL Invalid after {MAX_SQL_RETRIES} retries. Handling error.")
        state["error_message"] = f"SQL validation failed after {MAX_SQL_RETRIES} attempts. Last feedback: {state.get('validation_feedback')}"
        return "handle_error"


def check_for_errors(state: AgentState) -> str:
    """Generic check for errors before proceeding (used after prepare and execute)."""
    if state.get("error_message"):
        logger.error(f"👨‍💻 DATA TEAM: Error detected in check_for_errors: {state['error_message']}")
        return "handle_error"
    return "continue"


# --- Graph Definition ---

def create_data_team_graph(llm_client: Any):
    """Creates and compiles the LangGraph StateGraph for the data team using a prebuilt ReAct agent."""
    workflow = StateGraph(AgentState)

    # Instantiate the prebuilt ReAct agent for SQL generation + schema tool use
    # It expects input state to have a 'messages' key.
    sql_generating_agent = create_react_agent(llm_client, tools=[get_database_schema])

    # Bind the llm_client to the validation node
    validate_sql_with_llm = partial(sql_validator_node, llm_client=llm_client)

    # Add nodes
    workflow.add_node("prepare_query", prepare_data_query_node)
    workflow.add_node("sql_generating_agent", sql_generating_agent) # The prebuilt ReAct agent
    workflow.add_node("extract_sql", extract_sql_node) # New node to extract SQL
    workflow.add_node("validate_sql", validate_sql_with_llm)
    workflow.add_node("execute_sql", sql_executor_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("format_response", format_final_response_node)

    # Helper node to increment retry counter before looping back to agent
    workflow.add_node("increment_retry", lambda state: {
        "sql_generation_retries": state.get("sql_generation_retries", 0) + 1,
        "messages": [] # Clear messages before retry to avoid confusing agent? Or keep history? Let's clear for now.
        # Alternative: Keep history but add a specific retry instruction in prepare_query
    })


    # Set entry point
    workflow.set_entry_point("prepare_query")

    # Add edges
    workflow.add_conditional_edges(
        "prepare_query",
        check_for_errors, # Check for errors during preparation
        {
            "handle_error": "handle_error",
            "continue": "sql_generating_agent" # Start the ReAct agent
        }
    )

    # Edge from ReAct agent to SQL extraction node
    workflow.add_edge("sql_generating_agent", "extract_sql")

    # Conditional edge after SQL extraction
    workflow.add_conditional_edges(
        "extract_sql",
        check_sql_extraction, # Check if SQL was found
        {
            "validate_sql": "validate_sql",
            "handle_error": "handle_error"
        }
    )

    # Conditional edge after validation (retry logic)
    workflow.add_conditional_edges(
        "validate_sql",
        decide_after_validation,
        {
            "execute_sql": "execute_sql",
            "retry_sql_generation": "increment_retry", # Go to increment node before retrying
            "handle_error": "handle_error",
        },
    )

    # Edge from increment_retry back to prepare_query (to format the retry prompt correctly)
    workflow.add_edge("increment_retry", "prepare_query")

    # Edge after execution
    workflow.add_conditional_edges(
        "execute_sql",
        check_for_errors, # Check for execution errors
        {
            "handle_error": "handle_error",
            "continue": "format_response"
        }
    )

    # Edges to END
    workflow.add_edge("handle_error", "format_response")
    workflow.add_edge("format_response", END)

    # Compile the graph
    data_team_app = workflow.compile()
    data_team_app.name = "data_analysis_team"
    logger.info("👨‍💻 DATA TEAM: Compiled graph using prebuilt ReAct agent.")
    return data_team_app 