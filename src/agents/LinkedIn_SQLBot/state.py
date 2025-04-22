from typing import TypedDict, Annotated, List, Union, Any, Optional, Dict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # Data team specific state
    natural_language_query: Optional[str] = None
    schema: Optional[str] = None
    generated_sql: Optional[str] = None
    validation_status: Optional[str] = None # e.g., "valid", "invalid", "error"
    validation_feedback: Optional[str] = None # LLM feedback on invalid SQL
    execution_result: Optional[str] = None # Result from DB execution
    error_message: Optional[str] = None # To capture errors in the flow
    sql_generation_retries: int = 0 # Counter for SQL generation attempts
    # Potentially add other state variables here if needed
    # e.g., active_agent: str | None = None
    # e.g., data_query: str | None = None
    # e.g., validation_result: str | None = None
    # e.g., execution_result: Any | None = None

    # New fields for LinkedIn-style workflow
    candidate_contexts: List[Dict[str, Any]] | None
    ranked_tables: List[str] | None
    ranked_fields: List[str] | None
    plan_steps: List[str] | None
    validation_errors: List[str] | None
    db_connection_string: str | None # To store DB connection info
    max_iterations: int # To prevent infinite loops in correction
    current_iteration: int # To track correction attempts