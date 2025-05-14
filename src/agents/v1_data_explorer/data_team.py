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
from .rag_utils import get_few_shot_examples # Added import
from langchain_openai import OpenAIEmbeddings # Added import for embedding model

# Constants
MAX_SQL_RETRIES = 2
# Initial prompt message for the ReAct agent
REACT_AGENT_INITIAL_PROMPT = """
You are a data analyst expert tasked with generating SQLite SQL queries based on user requests OR providing the database schema when asked.
You have access to a tool to fetch the database schema (`get_database_schema`).

Here are some examples of how to respond to different user queries:
<few_shot_examples>
{few_shot_examples}
</few_shot_examples>

Your goal is to:
1. Analyze the user's query: 
<user_query>    
{query}
</user_query>
2. **If the user is asking for the database schema:** Use the `get_database_schema` tool. Once you receive the schema description from the tool, **respond ONLY with that exact schema description** provided by the tool, without any additional text or SQL.
3. **If the user is asking a question that requires data from the database:**
    a. Use the `get_database_schema` tool *if* you need it to understand the tables and columns.
    b. Once you have the necessary information (schema, if required), generate the appropriate SQLite SQL query.
    c. Respond ONLY with the final SQL query itself, without any introductory text, explanations, or markdown formatting like ```sql.

{retry_feedback}
"""

# --- Node to prepare input for the data team ---
def prepare_data_query_node(state: AgentState, embed_model: Any) -> Dict[str, Any]:
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

    # Get few-shot examples
    few_shot_examples_str = ""
    if embed_model:
        try:
            logger.info("👨‍💻 DATA TEAM: Retrieving few-shot examples...")
            few_shot_examples_str = get_few_shot_examples(
                user_query=last_human_message_content,
                embed_model=embed_model
            )
            if few_shot_examples_str and few_shot_examples_str.strip():
                # Count examples based on the specific delimiter used in rag_utils.py
                example_delimiter = "\n\n---\n\n"
                num_examples_retrieved = len(few_shot_examples_str.strip().split(example_delimiter))
                logger.info(f"👨‍💻 DATA TEAM: Successfully retrieved {num_examples_retrieved} few-shot examples.")
            else:
                logger.info("👨‍💻 DATA TEAM: No few-shot examples retrieved or an issue occurred.")
        except Exception as e:
            logger.warning(f"👨‍💻 DATA TEAM: Failed to get few-shot examples: {e}")
            few_shot_examples_str = "<!-- Could not retrieve few-shot examples -->" # Placeholder in case of error
    else:
        logger.warning("👨‍💻 DATA TEAM: No embedding model provided, skipping few-shot examples.")
        few_shot_examples_str = "<!-- Few-shot examples skipped (no embedding model) -->"

    # Check for retry feedback (only relevant if we are retrying SQL generation)
    retry_feedback = state.get("validation_feedback", "")
    # Only add retry feedback if we are actually in a SQL retry loop, not if the first try was a schema request
    # This check can be refined if we add a specific flag for "retrying_sql"
    is_retrying_sql = state.get("generated_sql") is not None and state.get("validation_status") == "invalid"
    retry_prompt_addition = ""
    if is_retrying_sql and retry_feedback:
        retry_prompt_addition = f"""

Retry Feedback: You previously generated SQL that failed validation with the following feedback: '{retry_feedback}'. Please analyze this feedback and generate a corrected query."""
    
    initial_prompt = REACT_AGENT_INITIAL_PROMPT.format(
        query=last_human_message_content,
        few_shot_examples=few_shot_examples_str,
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
        logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning due to error_message: {state.get('error_message')}")
        return {}

    last_message = state["messages"][-1] if state["messages"] else None
    return_dict = {}

    if isinstance(last_message, AIMessage) and last_message.content:
        content = last_message.content.strip()
        
        is_likely_schema = "database schema:" in content.lower() or "table:" in content.lower() and "columns:" in content.lower()
        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "DROP ", "ALTER "]
        is_sql = any(kw in content.upper() for kw in sql_keywords)
        
        sql_match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        extracted_sql_from_markdown = None
        if sql_match:
            extracted_sql_from_markdown = sql_match.group(1).strip()
            logger.info(f"👨‍💻 DATA TEAM: Found SQL-like content in markdown block: {extracted_sql_from_markdown[:100]}...")
            if extracted_sql_from_markdown and any(kw in extracted_sql_from_markdown.upper() for kw in sql_keywords):
                logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (from markdown block): {extracted_sql_from_markdown}")
                return_dict = {"generated_sql": extracted_sql_from_markdown, "provided_schema_text": None, "validation_feedback": None}
                logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning: {return_dict}")
                return return_dict
            else:
                logger.warning(f"👨‍💻 DATA TEAM: Markdown block found but content doesn't look like valid SQL: {extracted_sql_from_markdown}")

        if is_sql and not is_likely_schema:
            logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (raw content, looks like SQL): {content}")
            return_dict = {"generated_sql": content, "provided_schema_text": None, "validation_feedback": None}
            logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning: {return_dict}")
            return return_dict
        
        elif is_likely_schema and not is_sql:
            logger.info(f"👨‍💻 DATA TEAM: Identified as schema description: {content[:150]}...")
            return_dict = {"provided_schema_text": content, "generated_sql": None}
            logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning: {return_dict}")
            return return_dict

        elif is_likely_schema and is_sql:
             logger.warning(f"👨‍💻 DATA TEAM: Ambiguous output, contains schema-like terms and SQL keywords. Prioritizing as SQL if valid: {content}")
             if any(kw in content.upper() for kw in sql_keywords):
                logger.info(f"👨‍💻 DATA TEAM: Extracted SQL (raw content, ambiguous but processing as SQL): {content}")
                return_dict = {"generated_sql": content, "provided_schema_text": None, "validation_feedback": None}
                logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning: {return_dict}")
                return return_dict
             else:
                logger.info(f"👨‍💻 DATA TEAM: Ambiguous output, but not valid SQL structure. Treating as schema/text: {content[:150]}...")
                return_dict = {"provided_schema_text": content, "generated_sql": None}
                logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning: {return_dict}")
                return return_dict
        else:
            natural_query = state.get("natural_language_query", "").lower()
            if "schema" in natural_query or "database schema" in natural_query :
                logger.warning(f"👨‍💻 DATA TEAM: Expected schema, got this text. Treating as schema: {content[:150]}...")
                return_dict = {"provided_schema_text": content, "generated_sql": None}
                logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning: {return_dict}")
                return return_dict
            else:
                logger.error(f"👨‍💻 DATA TEAM: Failed to extract valid SQL structure or schema from agent response: {content}")
                return_dict = {"error_message": f"Agent did not produce a recognizable SQL query or schema. Final response: {content}"}
                logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning with error: {return_dict}")
                return return_dict
    else:
        logger.error(f"👨‍💻 DATA TEAM: No valid AIMessage content found from ReAct agent. Last message: {last_message}")
        return_dict = {"error_message": "ReAct agent did not return a final message with content."}
        logger.info(f"👨‍💻 DATA TEAM (extract_schema_or_sql_node) returning with error: {return_dict}")
        return return_dict


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
    logger.info(f"👨‍💻 DATA TEAM (sql_validator_node) received generated_sql: {state.get('generated_sql')}")
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
    logger.info(f"👨‍💻 DATA TEAM (sql_executor_node) received generated_sql: {state.get('generated_sql')}")
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
    """Formats the final response message for the supervisor and preserves other state keys."""
    logger.info("👨‍💻 DATA TEAM: Formatting final response...")
    logger.info(f"👨‍💻 DATA TEAM (format_final_response_node) received full state: {state}")
    logger.info(f"👨‍💻 DATA TEAM (format_final_response_node) state.get('generated_sql'): {state.get('generated_sql')}")
    error_message = state.get("error_message")
    execution_result = state.get("execution_result")
    provided_schema_text = state.get("provided_schema_text")
    query = state.get("natural_language_query", "your query")
    final_message_content = ""

    if provided_schema_text:
        final_message_content = f"Okay, here is the database schema you requested for '{query}':\n\n{provided_schema_text}"
    elif error_message:
        final_message_content = f"I encountered an error trying to answer '{query}': {error_message}"
    elif execution_result:
        final_message_content = f"Okay, I looked into '{query}'. Here's the result from the database:\n\n{execution_result}"
    else:
        final_message_content = f"I tried processing '{query}' to get data, but couldn't successfully validate or execute an SQL query. The final validation feedback was: {state.get('validation_feedback', 'No specific feedback available.')}"

    final_message = AIMessage(content=final_message_content, name="data_team_final_response")
    
    # Create a new state dictionary to avoid modifying the input state directly if it's not desired by LangGraph's implicit state management
    # However, common practice in LangGraph nodes is to return a dictionary of the keys to update in the state.
    # To ensure all relevant keys are preserved, especially generated_sql, we should return it explicitly if it exists.
    
    # The safest approach is to return a dictionary of all fields that this graph is responsible for.
    # If the graph is supposed to be the final step for the data team before handing to supervisor,
    # it should ensure all relevant data (messages, generated_sql, execution_result etc.) are in the returned dict.

    # Let's update the messages in the current state and return all relevant fields.
    updated_state_dict = state.copy() # Start with a copy of the current state
    updated_state_dict["messages"] = [final_message] # Update the messages
    
    # Ensure fields like generated_sql are explicitly carried over if they exist from previous steps in this graph
    # This is slightly redundant if state.copy() already does it, but makes the intent clear.
    # if "generated_sql" in state and state["generated_sql"] is not None:
    #     updated_state_dict["generated_sql"] = state["generated_sql"]
    # else:
    #     updated_state_dict["generated_sql"] = None # Ensure it's None if not set

    # The state object is a TypedDict. We should return a dictionary that matches its keys.
    # What `run_agent_session` expects are `response_content` and `generated_sql`.
    # The supervisor gets the full AgentState.
    # This node is internal to data_team; its return updates data_team's state.
    # The critical part is that when data_team graph *ends*, the AgentState must contain generated_sql.

    # The original problem: return {"messages": [final_message]} effectively clears other state fields for the supervisor.
    # LangGraph nodes update the overall state by returning a dictionary of the *changes*.
    # If we only return {"messages": ...}, other fields like "generated_sql" might be implicitly kept or discarded 
    # depending on how the parent graph (supervisor) handles state merging. 
    # It seems like they are being discarded.

    # Correct approach: ensure all fields of AgentState that this team is responsible for are returned or updated.
    # The most robust fix is to ensure that when this node is the one setting the final message, 
    # it also ensures generated_sql is part of the returned dictionary if it should be.

    return_dict = {
        "messages": [final_message],
        # Explicitly pass through other important state fields that should persist 
        # from the data_team's operation up to the supervisor.
        "natural_language_query": state.get("natural_language_query"),
        "schema": state.get("schema"),
        "generated_sql": state.get("generated_sql"), # <<< THIS IS THE KEY FIX
        "validation_status": state.get("validation_status"),
        "validation_feedback": state.get("validation_feedback"),
        "execution_result": state.get("execution_result"),
        "error_message": state.get("error_message"), # If an error occurred, it would be in the message though
        "sql_generation_retries": state.get("sql_generation_retries"),
        "provided_schema_text": state.get("provided_schema_text")
    }
    # Filter out None values to keep the state clean, unless the key is 'generated_sql' or 'execution_result'
    # as None is a valid state for them (e.g. no SQL generated, or no execution yet).
    return {k: v for k, v in return_dict.items() if v is not None or k in ["generated_sql", "execution_result", "provided_schema_text", "validation_feedback", "error_message"]} 


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

def create_data_team_graph(llm_client: Any, embed_model_instance: Optional[Any] = None):
    """Creates and compiles the LangGraph StateGraph for the data team using a prebuilt ReAct agent."""
    workflow = StateGraph(AgentState)

    # Instantiate the prebuilt ReAct agent for SQL generation + schema tool use
    # It expects input state to have a 'messages' key.
    sql_generating_agent = create_react_agent(llm_client, tools=[get_database_schema])

    # Bind the llm_client to the validation node
    validate_sql_with_llm = partial(sql_validator_node, llm_client=llm_client)

    # Prepare embedding model
    current_embed_model = embed_model_instance
    if current_embed_model is None:
        try:
            logger.info("👨‍💻 DATA TEAM: No embed_model_instance provided, attempting to instantiate OpenAIEmbeddings.")
            current_embed_model = OpenAIEmbeddings() # Default to OpenAIEmbeddings
        except Exception as e:
            logger.error(f"👨‍💻 DATA TEAM: Failed to instantiate OpenAIEmbeddings: {e}. Few-shot examples will be disabled.")
            current_embed_model = None # Ensure it's None if instantiation fails
            
    # Bind dependencies to prepare_data_query_node
    prepare_query_with_deps = partial(prepare_data_query_node, embed_model=current_embed_model)

    # Add nodes
    workflow.add_node("prepare_query", prepare_query_with_deps)
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
    logger.info("👨‍💻 DATA TEAM: Compiled graph with schema/SQL extraction logic and RAG few-shot examples.")
    return data_team_app 