from typing import List, Dict, Any, Annotated
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.tools import tool
# from langgraph.graph import State, START
from langgraph.graph.state import CompiledState # Correct import for CompiledState if needed elsewhere
from langgraph.graph import StateGraph # Import StateGraph if needed
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.graph import END

from langchain_experimental.utilities import PythonREPL
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from src.database.engine import engine as db_engine # Correct import
from sqlalchemy import inspect, text

import logging

from src.agents.LinkedIn_SQLBot.state import AgentState # Import AgentState
# from langgraph.prebuilt import ToolNode # Correct import for ToolNode
# from langgraph.graph import START


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_db_schema(conn_engine):
    inspector = inspect(conn_engine)
    tables = inspector.get_table_names()
    schema_str = "" 
    for table in tables:
        schema_str += f"Table: {table}\nColumns:\n"
        columns = inspector.get_columns(table)
        for col in columns:
            schema_str += f"  - {col['name']} ({col['type']})\n"
        schema_str += "\n"
    return schema_str


# --- Tools --- 

@tool
def retrieve_relevant_context(user_query: str, conn_engine=db_engine) -> Dict[str, List[str]]:
    """Retrieves table names and potentially relevant column schemas based on the user query."""
    logger.debug(f"retrieve_relevant_context called with query: {user_query}")
    inspector = inspect(conn_engine)
    all_tables = inspector.get_table_names()
    
    # Basic retrieval: Return all tables for now. 
    # TODO: Implement more sophisticated retrieval (e.g., embedding search)
    candidate_contexts = []
    for table in all_tables:
        columns = inspector.get_columns(table)
        col_str = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
        candidate_contexts.append(f"Table: {table}\nColumns: {col_str}")

    logger.debug(f"retrieve_relevant_context found contexts: {candidate_contexts}")
    return {"candidate_contexts": candidate_contexts}

# Tool Input Schemas (explicit arguments)
class RankTablesInput(BaseModel):
    user_query: str = Field(description="The user's natural language query.")
    candidate_contexts: List[str] = Field(description="List of potential table schemas (name and columns) to rank.")

@tool(args_schema=RankTablesInput)
def rank_tables(user_query: str, candidate_contexts: List[str]) -> List[str]: # Explicit args
    """Ranks tables based on their relevance to the user query."""
    logger.debug(f"rank_tables called. Query: '{user_query}', Contexts: {len(candidate_contexts)}")
    # Placeholder ranking logic: Assume all are equally relevant for now
    # In a real scenario, use an LLM or other ranking mechanism
    ranked_table_names = []
    for context in candidate_contexts:
        # Extract table name (assuming format "Table: <name>\n...")
        try:
            table_name = context.split('\n')[0].split(':')[1].strip()
            ranked_table_names.append(table_name)
        except IndexError:
            logger.warning(f"Could not parse table name from context: {context}")
            continue # Skip malformed context

    logger.debug(f"rank_tables returning ranked tables: {ranked_table_names}")
    return ranked_table_names

# Tool Input Schemas (explicit arguments)
class SelectFieldsInput(BaseModel):
    user_query: str = Field(description="The user's natural language query.")
    candidate_contexts: List[str] = Field(description="List of potential table schemas (name and columns). Required to map ranked tables back to schemas.")
    ranked_tables: List[str] = Field(description="List of table names ranked by relevance.")

@tool(args_schema=SelectFieldsInput)
def select_fields(user_query: str, candidate_contexts: List[str], ranked_tables: List[str]) -> List[str]: # Explicit args
    """Selects relevant fields from the ranked tables based on the user query."""
    logger.debug(f"select_fields called. Query: '{user_query}', Ranked Tables: {ranked_tables}")
    # Placeholder field selection: Select all fields from top N tables (e.g., top 3)
    # In a real scenario, use an LLM
    selected_fields = []
    top_n = 3 
    tables_to_consider = ranked_tables[:top_n]
    
    logger.debug(f"Selecting fields from top {top_n} tables: {tables_to_consider}")
    
    for table_name in tables_to_consider:
        # Find the context for this table to get columns
        table_context = None
        for context in candidate_contexts:
             try:
                 ctx_table_name = context.split('\n')[0].split(':')[1].strip()
                 if ctx_table_name == table_name:
                    table_context = context
                    break
             except IndexError:
                 continue # Skip malformed context

        if table_context:
            try:
                # Extract column names (assuming format "Columns: col1 (type), col2 (type), ...")
                columns_line = next(line for line in table_context.split('\n') if line.strip().startswith("Columns:"))
                columns_part = columns_line.split(':', 1)[1].strip()
                columns = [col.split(' (')[0].strip() for col in columns_part.split(',')]
                # Qualify field names with table name
                selected_fields.extend([f"{table_name}.{col}" for col in columns])
            except (StopIteration, IndexError):
                 logger.warning(f"Could not parse columns for table {table_name} from context: {table_context}")
        else:
            logger.warning(f"Could not find context for ranked table: {table_name}")

    logger.debug(f"select_fields returning selected fields: {selected_fields}")
    return selected_fields

# Tool Input Schemas (explicit arguments)
class PlanQueryInput(BaseModel):
    user_query: str = Field(description="The user's natural language query.")
    ranked_tables: List[str] = Field(description="List of table names ranked by relevance.")
    ranked_fields: List[str] = Field(description="List of potentially relevant fields (table.field)." )
    candidate_contexts: List[str] = Field(description="List of all retrieved table schemas for context.") # Added for full context


@tool(args_schema=PlanQueryInput)
def plan_query(user_query: str, ranked_tables: List[str], ranked_fields: List[str], candidate_contexts: List[str]) -> Dict[str, Any]: # Explicit args
    """Plans the SQL query based on the user query, relevant tables, and fields."""
    logger.debug(f"plan_query called. Query: '{user_query}', Tables: {ranked_tables}, Fields: {ranked_fields}")
    # Placeholder planning logic: Generate a simple SELECT query
    # In a real scenario, use an LLM to generate the plan and SQL
    plan_steps = ["Identify tables and fields", "Construct SELECT statement"]
    sql_query = "-- Placeholder query (replace with LLM generation)\n"
    if ranked_fields and ranked_tables:
        select_clause = ", ".join(ranked_fields)
        # Infer tables from fields if possible, fallback to ranked_tables
        # Simple heuristic: Use tables mentioned in ranked_fields
        tables_in_fields = set(f.split('.')[0] for f in ranked_fields if '.' in f)
        from_clause = ", ".join(tables_in_fields if tables_in_fields else ranked_tables)
        sql_query = f"SELECT {select_clause}\nFROM {from_clause};"
        # TODO: Add WHERE clause based on user_query analysis
        # TODO: Add JOIN logic if multiple tables are involved
        plan_steps.append(f"Select fields: {select_clause}")
        plan_steps.append(f"From tables: {from_clause}")
    else:
        sql_query = "-- Error: Could not determine tables or fields for query."
        plan_steps = ["Error: Missing table/field information."]

    logger.debug(f"plan_query returning plan: {plan_steps}, SQL: {sql_query}")
    # Return both plan steps and the SQL query
    return {
        "plan_steps": plan_steps, 
        "sql_query": sql_query
    }


# Tool Input Schemas (explicit arguments)
class ValidateQueryInput(BaseModel):
    sql_query: str = Field(description="The SQL query generated by the planner.")
    user_query: str = Field(description="The original user query for semantic validation.") # Add user query
    candidate_contexts: List[str] = Field(description="List of table schemas for context.") # Add context

@tool(args_schema=ValidateQueryInput)
def validate_query(sql_query: str, user_query: str, candidate_contexts: List[str], conn_engine=db_engine) -> List[str]: # Explicit args
    """Validates the generated SQL query syntactically and semantically."""
    logger.debug(f"validate_query called. SQL: {sql_query}")
    errors = []

    if not sql_query or sql_query.strip() == "-- Error: Could not determine tables or fields for query." or sql_query.strip().startswith("-- Error") :
        logger.warning(f"Skipping validation for invalid/placeholder SQL: {sql_query}")
        errors.append("Query planning failed, cannot validate.")
        return errors
    
    # 1. Syntactic Validation (using EXPLAIN)
    try:
        with conn_engine.connect() as connection:
            # Use text() to properly handle the SQL string
            connection.execute(text(f"EXPLAIN {sql_query}")) 
        logger.debug("SQL syntax validation passed.")
    except Exception as e:
        logger.warning(f"SQL syntax validation failed: {e}")
        errors.append(f"Syntactic Error: {str(e)}")
        # If syntax fails, semantic validation might not be meaningful
        # return errors # Optionally return early

    # 2. Semantic Validation (placeholder - requires LLM)
    # - Does the query address the user_query?
    # - Does it use appropriate tables/columns from the context?
    # - Are joins logical (if any)?
    # Example check (very basic):
    if "SELECT" not in sql_query.upper():
        errors.append("Semantic Error: Query doesn't appear to select data.")
    # TODO: Add LLM-based semantic checks here, comparing sql_query against user_query and candidate_contexts

    logger.debug(f"validate_query returning errors: {errors}")
    return errors


# Tool Input Schemas (explicit arguments)
class ExecuteQueryInput(BaseModel):
    sql_query: str = Field(description="The validated SQL query to execute.") 