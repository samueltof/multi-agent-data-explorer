from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from src.services.database.database_manager import DatabaseManager
from .config import get_db_manager
from src.config.database_config import DatabaseType
import pandas as pd
import os

# Initialize Tavily Search Tool
# Ensure TAVILY_API_KEY is set in your environment variables
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
# @tool
# def get_database_schema() -> str:
#     """Returns the schema description of the database.
    
#     Returns:
#         A string containing the database schema description.
#     """
#     try:
#         db_manager: DatabaseManager = get_db_manager()
        
#         if db_manager.settings.database_type == DatabaseType.SQLITE:
#             # Use the manager's execute_query method which handles thread-local connections
#             try:
#                 # Get table names
#                 table_cols, table_rows = db_manager.execute_query("SELECT name FROM sqlite_master WHERE type='table';")
                
#                 schema_desc = "Available tables and their schemas:\n\n"
#                 for row in table_rows:
#                     table_name = row[0]
#                     schema_desc += f"Table: {table_name}\n"
                    
#                     # Get column info for the current table
#                     col_info_cols, col_info_rows = db_manager.execute_query(f"PRAGMA table_info({table_name});")
                    
#                     for col_row in col_info_rows:
#                         # col_row format: (cid, name, type, notnull, dflt_value, pk)
#                         schema_desc += f"- {col_row[1]} ({col_row[2]}){' PRIMARY KEY' if col_row[5] else ''}{' NOT NULL' if col_row[3] else ''}\n"
#                     schema_desc += "\n"
#                 return schema_desc
#             except Exception as query_e:
#                  return f"Error querying SQLite schema: {str(query_e)}"
#         else:
#              # Fallback to the method using YAML file if configured
#              if db_manager.settings.database_schema_path:
#                  return db_manager.load_schema_description()
#              else:
#                  return "Database schema description file not configured."

#     except Exception as e:
#         # Catch errors during manager initialization or other issues
#         return f"Error fetching database schema: {str(e)}"

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
            return "Database schema description file not configured."

    except Exception as e:
        # Catch errors during manager initialization or other issues
        return f"Error fetching database schema: {str(e)}"


# List of tools for agents
data_tools = [execute_sql_query, get_database_schema]
web_tools = [tavily_search] 