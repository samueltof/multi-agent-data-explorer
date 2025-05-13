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
You are a data analyst expert tasked with generating SQLite SQL queries based on user requests OR providing the database schema when asked.
You have access to a tool to fetch the database schema (`get_database_schema`).

Your goal is to:
1. Analyze the user's query: {query}
2. **If the user is asking for the database schema:** Use the `get_database_schema` tool. Once you receive the schema description from the tool, **respond ONLY with that exact schema description** provided by the tool, without any additional text or SQL.
3. **If the user is asking a question that requires data from the database:**
    a. Use the `get_database_schema` tool *if* you need it to understand the tables and columns.
    b. Once you have the necessary information (schema, if required), generate the appropriate SQLite SQL query.
    c. Respond ONLY with the final SQL query itself, without any introductory text, explanations, or markdown formatting like ```sql.

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

    # Check for retry feedback (only relevant if we are retrying SQL generation)
    retry_feedback = state.get("validation_feedback", "")
    # Only add retry feedback if we are actually in a SQL retry loop, not if the first try was a schema request
    # This check can be refined if we add a specific flag for "retrying_sql"
    is_retrying_sql = state.get("generated_sql") is not None and state.get("validation_status") == "invalid"
    retry_prompt_addition = ""
    if is_retrying_sql and retry_feedback:
        retry_prompt_addition = f"\nRetry Feedback: You previously generated SQL that failed validation with the following feedback: '{retry_feedback}'. Please analyze this feedback and generate a corrected query."
    
    initial_prompt = REACT_AGENT_INITIAL_PROMPT.format(
        query=last_human_message_content,
        retry_feedback=retry_prompt_addition
    )

    return {
        "messages": [HumanMessage(content=initial_prompt)],
        "natural_language_query": last_human_message_content,
        "schema": state.get("schema"), 
        "generated_sql": None, # Reset for this run
        "provided_schema_text": None, # Reset for this run
        "validation_status": None,
        "validation_feedback": None,
        "execution_result": None,
        "error_message": None,
        # Preserve retry count if we are coming from an increment_retry node
        "sql_generation_retries": state.get("sql_generation_retries", 0) if not is_retrying_sql else state["sql_generation_retries"]
    }


# --- Node to Extract Schema or SQL from ReAct Agent Output ---
def extract_schema_or_sql_node(state: AgentState) -> Dict[str, Any]:
    """Extracts schema text or SQL query from the last message of the ReAct agent."""
    logger.info("👨‍💻 DATA TEAM: Extracting schema or SQL from ReAct agent output...")
    if state.get("error_message"): # Pass through errors
        return {}

    last_message = state["messages"][-1] if state["messages"] else None

    if isinstance(last_message, AIMessage) and last_message.content:
        content = last_message.content.strip()
        
        # Heuristic to detect if the content is likely a schema description
        # This could be improved (e.g., checking for specific keywords like "Table:", "Columns:")
        # For now, we'll rely on the agent's prompt to return *only* schema if it's a schema.
        # If it's not clearly SQL, we'll assume it might be schema or an error message from the agent.
        
        is_likely_schema = "database schema:" in content.lower() or "table:" in content.lower() and "columns:" in content.lower()
        
        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "DROP ", "ALTER "]
        is_sql = any(kw in content.upper() for kw in sql_keywords)
        
        # Try to extract SQL if markdown is present, even if it looks like schema (agent might be wrong)
        sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        extracted_sql_from_markdown = None
        if sql_match:
            extracted_sql_from_markdown = sql_match.group(1).strip()
            logger.info(f"👨‍💻 DATA TEAM: Found SQL-like content in markdown block: {extracted_sql_from_markdown[:100]}...")
            # If markdown SQL is found, prioritize it as SQL
            if extracted_sql_from_markdown and any(kw in extracted_sql_from_markdown.upper() for kw in sql_keywords):
                logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (from markdown block): {extracted_sql_from_markdown}")
                return {"generated_sql": extracted_sql_from_markdown, "provided_schema_text": None, "validation_feedback": None}
            else: # Markdown content wasn't valid SQL structure
                logger.warning(f"👨‍💻 DATA TEAM: Markdown block found but content doesn't look like valid SQL: {extracted_sql_from_markdown}")


        # If no valid SQL from markdown, evaluate raw content
        if is_sql and not is_likely_schema: # It looks like SQL and not primarily schema
            logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (raw content, looks like SQL): {content}")
            return {"generated_sql": content, "provided_schema_text": None, "validation_feedback": None}
        
        elif is_likely_schema and not is_sql: # It looks like schema and not SQL
            logger.info(f"👨‍💻 DATA TEAM: Identified as schema description: {content[:150]}...")
            return {"provided_schema_text": content, "generated_sql": None}

        elif is_likely_schema and is_sql: # Ambiguous: contains schema-like terms AND SQL keywords
             logger.warning(f"👨‍💻 DATA TEAM: Ambiguous output, contains schema-like terms and SQL keywords. Prioritizing as SQL if valid: {content}")
             if any(kw in content.upper() for kw in sql_keywords): # Double check raw content as SQL
                logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (raw content, ambiguous but processing as SQL): {content}")
                return {"generated_sql": content, "provided_schema_text": None, "validation_feedback": None}
             else: # Doesn't pass SQL keyword check after all
                logger.info(f"👨‍💻 DATA TEAM: Ambiguous output, but not valid SQL structure. Treating as schema/text: {content[:150]}...")
                return {"provided_schema_text": content, "generated_sql": None}

        else: # Does not look like SQL and not clearly schema - could be an error message or other text from agent
            # If agent was supposed to return schema but didn't, or SQL but didn't, this is an issue.
            # Let's check if the original query explicitly asked for schema.
            natural_query = state.get("natural_language_query", "").lower()
            if "schema" in natural_query or "database schema" in natural_query :
                # Agent was asked for schema but returned something else, treat it as the (possibly bad) schema text
                logger.warning(f"👨‍💻 DATA TEAM: Expected schema, got this text. Treating as schema: {content[:150]}...")
                return {"provided_schema_text": content, "generated_sql": None}
            else:
                # Expected SQL, but didn't get it
                logger.error(f"👨‍💻 DATA TEAM: Failed to extract valid SQL structure or schema from agent response: {content}")
                return {"error_message": f"Agent did not produce a recognizable SQL query or schema. Final response: {content}"}
                
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
    if state.get("error_message") or not state.get("generated_sql") or state.get("provided_schema_text"):
        logger.warning("👨‍💻 DATA TEAM: Skipping SQL validation due to prior error, missing SQL, or schema was provided directly.")
        if not state.get("generated_sql") and not state.get("provided_schema_text") and not state.get("error_message"):
             state["error_message"] = "Agent failed to produce an SQL query or schema."
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
    if state.get("error_message") or not state.get("generated_sql") or state.get("validation_status") != "valid" or state.get("provided_schema_text"):
        if state.get("validation_status") != "valid" and not state.get("provided_schema_text"):
            logger.warning(f"👨‍💻 DATA TEAM: Skipping execution because SQL validation status is '{state.get('validation_status')}'.")
        elif state.get("provided_schema_text"):
            logger.info("👨‍💻 DATA TEAM: Skipping SQL execution because schema text was provided.")
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
    provided_schema_text = state.get("provided_schema_text")
    query = state.get("natural_language_query", "your query")
    final_message_content = ""

    if provided_schema_text:
        # If schema text was directly provided by the agent, use that as the primary response.
        # This assumes the agent's task was specifically to provide the schema.
        final_message_content = f"Okay, here is the database schema you requested for '{query}':\n\n{provided_schema_text}"
        # Clear other fields that might be confusing if schema is the sole output for this agent turn
        # state["execution_result"] = None # This node shouldn't modify state beyond messages
        # state["error_message"] = None
    elif error_message:
        final_message_content = f"I encountered an error trying to answer '{query}': {error_message}"
    elif execution_result:
        final_message_content = f"Okay, I looked into '{query}'. Here's the result from the database:\n\n{execution_result}"
    else: # No schema, no error, no execution result -> likely SQL validation failed permanently
        final_message_content = f"I tried processing '{query}' to get data, but couldn't successfully validate or execute an SQL query. The final validation feedback was: {state.get('validation_feedback', 'No specific feedback available.')}"

    final_message = AIMessage(content=final_message_content, name="data_team_final_response")
    return {"messages": [final_message]}


# --- Conditional Edges ---

def check_extraction_result(state: AgentState) -> str:
    """Checks if schema or SQL was successfully extracted."""
    if state.get("error_message"):
        logger.error(f"👨‍💻 DATA TEAM: Error detected after extraction attempt: {state['error_message']}")
        return "handle_error"
    elif state.get("provided_schema_text"):
        logger.info("👨‍💻 DATA TEAM: Schema text provided by agent. Proceeding to format response.")
        return "format_response" # Directly format response if schema was given
    elif state.get("generated_sql"):
        logger.info("👨‍💻 DATA TEAM: SQL extracted successfully. Proceeding to validation.")
        return "validate_sql"
    else:
         logger.error("👨‍💻 DATA TEAM: Neither schema nor SQL found after extraction, but no error message set. Handling as error.")
         state["error_message"] = "Extraction failed unexpectedly to find schema or SQL."
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
        return "retry_sql_generation" # This will go to increment_retry then prepare_query
    else:
        logger.error(f"👨‍💻 DATA TEAM: SQL Invalid after {MAX_SQL_RETRIES} retries. Handling error.")
        state["error_message"] = f"SQL validation failed after {MAX_SQL_RETRIES} attempts. Last feedback: {state.get('validation_feedback')}"
        return "handle_error"


def check_for_errors_before_agent(state: AgentState) -> str:
    """Generic check for errors before proceeding to agent (used after prepare)."""
    if state.get("error_message"):
        logger.error(f"👨‍💻 DATA TEAM: Error detected in prepare_query: {state['error_message']}")
        return "handle_error"
    return "continue_to_agent" # New target name for clarity


def check_for_errors_after_execution(state: AgentState) -> str:
    """Generic check for errors after SQL execution."""
    if state.get("error_message"):
        logger.error(f"👨‍💻 DATA TEAM: Error detected after SQL execution: {state['error_message']}")
        return "handle_error"
    return "continue_to_format" # New target name for clarity


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
    workflow.add_node("extract_schema_or_sql", extract_schema_or_sql_node) # New node to extract schema or SQL
    workflow.add_node("validate_sql", validate_sql_with_llm)
    workflow.add_node("execute_sql", sql_executor_node)
    workflow.add_node("handle_error", handle_error_node)
    workflow.add_node("format_response", format_final_response_node)

    # Helper node to increment retry counter before looping back to agent
    workflow.add_node("increment_retry", lambda state: {
        "sql_generation_retries": state.get("sql_generation_retries", 0) + 1,
        "messages": [], # Clear messages before retry for SQL generation
        "provided_schema_text": None # Clear this if we are retrying SQL
    })


    # Set entry point
    workflow.set_entry_point("prepare_query")

    # Add edges
    workflow.add_conditional_edges(
        "prepare_query",
        check_for_errors_before_agent, # Check for errors during preparation
        {
            "handle_error": "handle_error",
            "continue_to_agent": "sql_generating_agent" # Start the ReAct agent
        }
    )

    # Edge from ReAct agent to SQL extraction node
    workflow.add_edge("sql_generating_agent", "extract_schema_or_sql")

    # Conditional edge after SQL extraction
    workflow.add_conditional_edges(
        "extract_schema_or_sql",
        check_extraction_result, # Check if schema or SQL was found
        {
            "validate_sql": "validate_sql",
            "format_response": "format_response", # New path for direct schema response
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
        check_for_errors_after_execution, # Check for execution errors
        {
            "handle_error": "handle_error",
            "continue_to_format": "format_response"
        }
    )

    # Edges to END
    workflow.add_edge("handle_error", "format_response")
    workflow.add_edge("format_response", END)

    # Compile the graph
    data_team_app = workflow.compile()
    data_team_app.name = "data_analysis_team"
    logger.info("👨‍💻 DATA TEAM: Compiled graph with schema/SQL extraction logic.")
    return data_team_app 