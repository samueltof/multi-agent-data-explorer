from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.services.database.database_manager import DatabaseManager
from .config import get_db_manager
from src.config.database_config import DatabaseType
from src.config.llm_config import LLMSettings
import pandas as pd
import os
import sqlite3
from typing import List, Dict, Any, Optional, Literal
from loguru import logger
from .state import AgentState # Assuming AgentState is defined here or imported
from langchain_core.prompts import PromptTemplate # Added
from langchain_core.messages import HumanMessage # Added
from langchain_openai import ChatOpenAI # Added
from pydantic import BaseModel, Field # Added

# Initialize Tavily Search Tool
# Ensure TAVILY_API_KEY is set in your environment variables
tavily_search = TavilySearchResults(max_results=3)
tavily_search.name = "web_search"
tavily_search.description = "A search engine useful for answering questions about current events, general knowledge, or recent information."

@tool
def execute_sql(query: str, db_path: str = "database/Chinook.db") -> str:
    """Executes a SQL query against the database and returns the result."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        logger.info(f"Executed SQL: {query}, Results: {results}")
        return str(results) # Return results as a string for now
    except Exception as e:
        logger.error(f"Error executing SQL: {query}, Error: {e}")
        # Return error message to be potentially used for self-correction
        return f"Error executing SQL: {e}"

# Tool for fetching database schema description
@tool
def get_database_schema() -> str:
    """Returns the schema description of the database.
    
    Returns:
        A string containing the database schema description.
    """
    try:
        db_manager: DatabaseManager = get_db_manager()  
             # Fallback to the method using YAML file if configured
        if db_manager.settings.database_schema_path:
            return db_manager.load_schema_description()
        else:
            # Attempt to dynamically generate schema if YAML not found
            logger.warning("Schema YAML not configured. Attempting dynamic schema generation.")
            if db_manager.settings.database_type == DatabaseType.SQLITE:
                return db_manager.get_sqlite_schema_description()
            else:
                return "Error: Dynamic schema generation not supported for this DB type and no schema file configured."

    except Exception as e:
        # Catch errors during manager initialization or other issues
        return f"Error fetching database schema: {str(e)}"


# List of tools for agents
data_tools = [execute_sql, get_database_schema]
web_tools = [tavily_search] 

# --- Prompts for Tools ---
# Adapted from v0
SQL_PLANNER_PROMPT = PromptTemplate(
    template="""
You are interacting with a **SQLite** database.

Based on the user query and the most relevant tables and fields identified:
User Query: {query}
Ranked Tables: {tables}
Selected Fields: {fields}
Relevant Schema Context: 
{context}

Generate a valid **SQLite** SQL query that addresses the user's request using ONLY the provided tables and fields.
Also provide a brief, step-by-step plan (max 3 steps) for how you constructed the query.

- Use `sqlite_master` or `sqlite_schema` for metadata queries ONLY if absolutely necessary and the context supports it.
- Focus ONLY on generating the SQL query and the plan.
- If you cannot construct a query from the provided information, state that clearly in the SQL query field (e.g., "-- Cannot construct query from provided info").

Respond using the required JSON structure.
""",
    input_variables=["query", "tables", "fields", "context"]
)

SQL_VALIDATOR_PROMPT = PromptTemplate(
    template="""\
You are validating a query for a **SQLite** database.

Database Schema (showing relevant context):
{schema_context}

Original User Query:
{query}

Generated SQL Query:
{sql}

Is the generated **SQLite** SQL query valid and does it correctly address the user query based on the provided schema context? 

Consider:
- **SQLite** SQL syntax correctness (e.g., usage of `sqlite_master` is valid for metadata if context supports it).
- Table and column existence in the provided schema context.
- Appropriateness of the query for the user's request.
- Potential for returning excessively large results (e.g., missing LIMIT clauses).

Respond with the required JSON structure. If the query is valid and correct, the feedback should state that clearly.
""",
    input_variables=["schema_context", "query", "sql"],
)

RANK_TABLES_PROMPT = PromptTemplate(
    template="""
You are an expert database analyst helping select relevant tables.
Based on the user query and the following retrieved schema context snippets (with relevance scores):

User Query: {query}

Context Snippets:
{context}

Identify and rank the database tables (up to 3) most likely relevant to answering the user query. 
Consider the table descriptions and columns mentioned in the context snippets.

Respond with ONLY a JSON list of the ranked table names.
Example Response: ["table1", "table2"]
""",
    input_variables=["query", "context"]
)

SELECT_FIELDS_PROMPT = PromptTemplate(
    template="""
You are an expert database analyst helping select relevant fields.
Based on the user query, the ranked relevant tables, and the schema context snippets:

User Query: {query}
Ranked Tables: {tables}
Context Snippets:
{context}

Identify and list the specific fields from the TOP ranked table ({top_table}) that are most likely needed to answer the user query.
Consider the field descriptions/types in the context snippets.

Respond with ONLY a JSON list of the selected field names for the table '{top_table}'.
Example Response: ["field1", "field2", "field_alias"]
""",
    input_variables=["query", "tables", "top_table", "context"]
)

# Pydantic model for the structured SQL generation output
class GeneratedSQLWithPlan(BaseModel):
    plan_steps: List[str] = Field(description="Brief step-by-step plan for query construction (max 3 steps).")
    sql_query: str = Field(description="The generated SQLite SQL query, or an error message if generation failed.")

# Pydantic model for the structured SQL validation output (from v0)
class SQLValidationResult(BaseModel):
    status: Literal["valid", "invalid", "error"] = Field(description="The validation status of the SQL query.")
    feedback: str = Field(description="Detailed feedback on the validation, including reasons for invalidity or confirmation of validity.")

# Pydantic models for structured list output
class RankedTablesResult(BaseModel):
    ranked_tables: List[str] = Field(description="List of ranked table names relevant to the query.")

class SelectedFieldsResult(BaseModel):
    selected_fields: List[str] = Field(description="List of selected field names from the top ranked table.")

# --- Placeholder Functions for LinkedIn-style Workflow ---

@tool
def retrieve_candidates(user_query: str) -> Dict[str, Any]:
    """
    Retrieves relevant schema snippets based on the user query using embeddings.
    Uses an in-memory FAISS vector store created from the database schema.
    Returns a dictionary containing 'candidate_contexts' or 'error'.
    """
    logger.info(f"Executing retrieve_candidates tool for query: '{user_query}'")
    # state: AgentState parameter removed
    # Initialize return dictionary
    result = {"candidate_contexts": [], "error": None}

    if not user_query:
        logger.error("User query is missing.")
        result["error"] = "User query is missing for schema retrieval."
        return result

    try:
        # 1. Get Schema
        schema_description = get_database_schema.invoke({}) # Assumes this tool needs no state input
        if "Error" in schema_description:
             logger.error(f"Failed to retrieve database schema: {schema_description}")
             result["error"] = f"Schema Retrieval Error: {schema_description}"
             return result

        # 2. Split Schema into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        schema_chunks = text_splitter.split_text(schema_description)

        if not schema_chunks:
             logger.warning("Schema description resulted in no chunks after splitting.")
             # Return empty contexts, not necessarily an error
             return result 

        # 3. Create Embeddings and Vector Store (In-Memory)
        llm_settings = LLMSettings()
        embeddings = OpenAIEmbeddings(openai_api_key=llm_settings.openai.api_key)
        logger.info(f"Creating FAISS index from {len(schema_chunks)} schema chunks...")
        vector_store = FAISS.from_texts(schema_chunks, embeddings)

        # 4. Perform Similarity Search
        logger.info(f"Performing similarity search for query: '{user_query}'")
        results_with_scores = vector_store.similarity_search_with_relevance_scores(user_query, k=5)

        # 5. Format and Update Return Value
        candidate_contexts = [
            {"text": doc.page_content, "score": score} for doc, score in results_with_scores
        ]
        result["candidate_contexts"] = candidate_contexts
        logger.info(f"Retrieved {len(candidate_contexts)} candidate contexts.")
        if not candidate_contexts:
             logger.warning("No relevant schema contexts found for the query.")

    except ImportError as ie:
        logger.error(f"ImportError during FAISS/Embedding operation: {ie}. Is faiss-cpu installed?")
        result["error"] = "ImportError: Required library for vector search missing."
    except Exception as e:
        logger.error(f"Error during retrieve_candidates tool: {e}", exc_info=True)
        result["error"] = f"Failed during schema retrieval/embedding: {str(e)}"

    # Return the results dictionary
    return result

@tool
def rank_tables(user_query: str, candidate_contexts: List[Dict[str, Any]]) -> List[str]:
    """
    Ranks tables based on retrieved contexts and user query using an LLM.
    Accepts user query and candidate contexts, returns a list of ranked table names.
    """
    logger.info("--- Running Rank Tables Node (LLM) ---")
    # state: AgentState parameter removed
    # Inputs are now explicit arguments: user_query, candidate_contexts
    ranked_tables_list = []

    if not candidate_contexts:
         logger.warning("No candidate contexts found for ranking tables.")
         return [] # Return empty list

    try:
        # Setup LLM
        llm_settings = LLMSettings()
        llm = ChatOpenAI(
            openai_api_key=llm_settings.openai.api_key, 
            model=llm_settings.openai.default_model, 
            temperature=llm_settings.openai.temperature
        )
        # Use structured output for getting the list
        structured_llm = llm.with_structured_output(RankedTablesResult)

        # Format context
        context_str = "\n".join([f"Snippet Score {c.get('score', 'N/A')}: {c.get('text', '')}" for c in candidate_contexts])

        # Format prompt
        prompt_str = RANK_TABLES_PROMPT.format(query=user_query, context=context_str)
        
        # Invoke LLM
        response_model = structured_llm.invoke([HumanMessage(content=prompt_str)])
        ranked_tables_list = response_model.ranked_tables

    except Exception as e:
        logger.error(f"Error during LLM Table Ranking: {e}", exc_info=True)
        # Fallback or simple extraction if LLM fails?
        # For now, just return empty list on error.

    logger.info(f"Ranked tables (LLM): {ranked_tables_list}")
    # Return the list directly
    return ranked_tables_list

@tool
def select_fields(user_query: str, candidate_contexts: List[Dict[str, Any]], ranked_tables: List[str]) -> List[str]:
    """
    Selects relevant fields from the top ranked table using an LLM.
    Accepts query, contexts, and ranked tables, returns a list of selected field names.
    """
    logger.info("--- Running Select Fields Node (LLM) ---")
    # state: AgentState parameter removed
    # Inputs are now explicit arguments
    selected_fields_list = []

    if not ranked_tables or not candidate_contexts:
         logger.warning("No ranked tables or contexts found for selecting fields.")
         return [] # Return empty list

    top_table = ranked_tables[0] # Focus on the top-ranked table

    try:
        # Setup LLM
        llm_settings = LLMSettings()
        llm = ChatOpenAI(
            openai_api_key=llm_settings.openai.api_key, 
            model=llm_settings.openai.default_model, 
            temperature=llm_settings.openai.temperature
        )
        structured_llm = llm.with_structured_output(SelectedFieldsResult)

        # Format context
        context_str = "\n".join([f"Snippet Score {c.get('score', 'N/A')}: {c.get('text', '')}" for c in candidate_contexts])
        
        # Format prompt
        prompt_str = SELECT_FIELDS_PROMPT.format(
            query=user_query,
            tables=", ".join(ranked_tables),
            top_table=top_table,
            context=context_str
        )

        # Invoke LLM
        response_model = structured_llm.invoke([HumanMessage(content=prompt_str)])
        selected_fields_list = response_model.selected_fields

    except Exception as e:
        logger.error(f"Error during LLM Field Selection: {e}", exc_info=True)
        # Fallback or simple extraction if LLM fails?
        # For now, just return empty list on error.

    logger.info(f"Selected fields for table '{top_table}' (LLM): {selected_fields_list}")
    # Return the list directly
    return selected_fields_list

@tool
def plan_query(user_query: str, ranked_tables: List[str], ranked_fields: List[str], candidate_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a SQL query plan and the SQL query itself using an LLM based on ranked tables/fields and context.
    Accepts query, ranked tables, ranked fields, and candidate contexts.
    Returns a dictionary containing 'plan_steps' and 'sql_query'.
    """
    logger.info("--- Running Plan Query Node (LLM) ---")
    # state: AgentState parameter removed
    # Inputs are now explicit arguments

    plan_steps_list = ["Error: Failed to generate plan."]
    sql_query_str = "-- Error: Failed during SQL generation."

    if not ranked_tables or not ranked_fields:
        logger.warning("Cannot plan query: Missing ranked tables or fields.")
        sql_query_str = "-- Could not determine tables/fields to query"
        plan_steps_list = ["1. Failed to identify relevant tables or fields."]
        return {"plan_steps": plan_steps_list, "sql_query": sql_query_str}

    try:
        # Setup LLM
        llm_settings = LLMSettings()
        llm = ChatOpenAI(
            openai_api_key=llm_settings.openai.api_key, 
            model=llm_settings.openai.default_model, 
            temperature=llm_settings.openai.temperature
        )
        structured_llm = llm.with_structured_output(GeneratedSQLWithPlan)

        # Format context
        context_str = "\n".join([f"Snippet Score {c.get('score', 'N/A')}: {c.get('text', '')}" for c in candidate_contexts])

        # Format prompt
        prompt_str = SQL_PLANNER_PROMPT.format(
            query=user_query,
            tables=", ".join(ranked_tables),
            fields=", ".join(ranked_fields),
            context=context_str
        )
        
        # Invoke LLM
        response_model = structured_llm.invoke([HumanMessage(content=prompt_str)])
        
        sql_query_str = response_model.sql_query.strip()
        plan_steps_list = response_model.plan_steps

        # Clean potential markdown formatting from SQL (as in v0)
        if sql_query_str.startswith("```sql"):
            cleaned_sql = sql_query_str[len("```sql"):].strip()
            if cleaned_sql.endswith("```"):
                cleaned_sql = cleaned_sql[:-len("```")].strip()
            sql_query_str = cleaned_sql
        elif sql_query_str.startswith("```"):
             cleaned_sql = sql_query_str[len("```"):].strip()
             if cleaned_sql.endswith("```"):
                 cleaned_sql = cleaned_sql[:-len("```")].strip()
             sql_query_str = cleaned_sql

    except Exception as e:
        logger.error(f"Error during LLM SQL planning: {e}", exc_info=True)
        # Keep the default error values for plan_steps_list, sql_query_str

    logger.info(f"Planned query: {sql_query_str}, Plan: {plan_steps_list}")
    # Return dictionary with results
    return {"plan_steps": plan_steps_list, "sql_query": sql_query_str}


@tool
def validate_query(sql_query: Optional[str], user_query: str, candidate_contexts: List[Dict[str, Any]]) -> List[str]:
    """
    Validates the generated SQL query using an LLM based on schema context and user query.
    Accepts the SQL query string, original user query, and schema contexts.
    Returns a list of validation error/feedback strings (empty if valid).
    """
    logger.info("--- Running Validate Query Node (LLM) ---")
    # state: AgentState parameter removed
    # Input is now explicit arguments
    validation_output_list = []

    if not sql_query or "-- Could not determine" in sql_query or "-- Error:" in sql_query:
         logger.warning(f"Skipping validation for invalid/missing query: {sql_query}")
         validation_output_list.append(f"Query was not generated or is invalid: {sql_query}")
         return validation_output_list

    try:
        # Setup LLM
        llm_settings = LLMSettings()
        llm = ChatOpenAI(
            openai_api_key=llm_settings.openai.api_key, 
            model=llm_settings.openai.default_model, 
            temperature=llm_settings.openai.temperature
        )
        structured_llm = llm.with_structured_output(SQLValidationResult)

        # Format context
        context_str = "\n".join([f"Snippet Score {c.get('score', 'N/A')}: {c.get('text', '')}" for c in candidate_contexts])

        # Format prompt
        prompt_str = SQL_VALIDATOR_PROMPT.format(
            schema_context=context_str,
            query=user_query,
            sql=sql_query
        )

        # Invoke LLM
        validation_result = structured_llm.invoke([HumanMessage(content=prompt_str)])
        
        # Access Pydantic model attributes directly
        status = validation_result.status
        feedback = validation_result.feedback.strip()

        logger.info(f"LLM Validation Result: Status='{status}', Feedback='{feedback}'")

        # Treat 'invalid' or 'error' status, or any non-empty feedback on 'valid' status as errors/warnings for the list
        if status != "valid":
            validation_output_list.append(f"Status: {status}. Feedback: {feedback}")
        elif feedback and "valid" not in feedback.lower() and "correct" not in feedback.lower(): # Add feedback even if status is valid, if it contains warnings
             validation_output_list.append(f"Feedback: {feedback}") # Report feedback as a potential issue/warning

    except Exception as e:
        logger.error(f"Error during LLM SQL validation: {e}", exc_info=True)
        validation_output_list.append(f"Error during validation call: {str(e)}")

    # Return the list of errors/feedback (empty if validation passed cleanly)
    if not validation_output_list:
        logger.info(f"Query validation passed (LLM logic): {sql_query}")
    else:
        logger.warning(f"Query validation failed or has feedback (LLM logic): {validation_output_list}")
    return validation_output_list

# --- End Placeholder Functions ---


# Example of how you might define AgentState if it's not imported
# class AgentState(TypedDict):
#     messages: List[Any]
#     user_query: str
#     # ... other existing fields
#     candidate_contexts: Optional[List[Dict[str, Any]]]
#     ranked_tables: Optional[List[str]]
#     ranked_fields: Optional[List[str]]
#     plan_steps: Optional[List[str]]
#     sql_query: Optional[str]
#     validation_errors: Optional[List[str]]
#     db_connection_string: Optional[str]
#     max_iterations: int
#     current_iteration: int 