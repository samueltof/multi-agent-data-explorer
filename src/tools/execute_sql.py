from config.database_config import DatabaseSettings
from services.database import DatabaseManager
from typing import Any, Dict, Optional
from config.logger import logger


def tool_execute_sql(
    sql_query: str,
    db_settings: Optional[DatabaseSettings] = None,
    db_manager: Optional[DatabaseManager] = None,
) -> Dict[str, Any]:
    """Execute an SQL query and return results with column information.

    Args:
        sql_query: SQL query to execute

    Returns:
        Dict containing:
            - queried_columns: List of column names
            - query_results: List of result rows
            - sql_query: Original SQL query
            - error: Error message if query failed (None otherwise)
    """
    logger.info("🛠️ Executing SQL query tool")

    try:
        # Initialize database connection
        if db_manager and db_settings:
            pass
        else:
            db_settings = DatabaseSettings()
            db_manager = DatabaseManager(settings=db_settings)


        # Execute query and get columns and results
        columns, results = db_manager.execute_query(sql_query)
        logger.info("🔘 SQL query executed successfully")

        return {
            "queried_columns": columns,
            "query_results": results,
            "sql_query": sql_query,
            "error": None,
        }
    except Exception as e:
        logger.error(f"❌ Error executing SQL query: {e}")
        return {
            "queried_columns": [],
            "query_results": [],
            "sql_query": sql_query,
            "error": str(e),
        }
