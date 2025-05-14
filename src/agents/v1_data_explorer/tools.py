from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from src.services.database.database_manager import DatabaseManager
from .config import get_db_manager
from src.config.database_config import DatabaseType
from src.config.logger import logger
import pandas as pd
import os

# Initialize Tavily Search Tool
tavily_search = TavilySearchResults(max_results=3)
tavily_search.name = "web_search"
tavily_search.description = "A search engine useful for answering questions about current events, general knowledge, or recent information."

@tool
def execute_sql_query(query: str) -> str:
    """Executes a given SQL query against the database and returns the result.
    
    Args:
        query: The SQL query string to be executed.
        
    Returns:
        A string representation of the query result (dataframe) or an error message.
    """
    try:
        db_manager: DatabaseManager = get_db_manager()
        # Use execute_query_df for better formatting with pandas
        result_df: pd.DataFrame = db_manager.execute_query_df(query)
        if result_df.empty:
            return "Query executed successfully, but returned no results."
        # Convert dataframe to string for LLM consumption
        return result_df.to_markdown(index=False)
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# Tool for fetching database schema description

@tool
def get_database_schema() -> str:
    """Returns the schema description of the database.
    
    Returns:
        A string containing the database schema description.
    """
    try:
        db_manager: DatabaseManager = get_db_manager()  
        logger.info(f"🛠️ TOOL get_database_schema: Fetching database schema using {db_manager.settings.database_schema_path}")
        if db_manager.settings.database_schema_path:
            schema_description = db_manager.load_schema_description()   
            return schema_description
        else:
            logger.warning("🛠️ TOOL get_database_schema: Database schema description file not configured.")
            return "Database schema description file not configured."

    except Exception as e:
        # Catch errors during manager initialization or other issues
        return f"Error fetching database schema: {str(e)}"


# List of tools for agents
data_tools = [execute_sql_query, get_database_schema]
web_tools = [tavily_search] 