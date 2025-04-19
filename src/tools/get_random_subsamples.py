from src.config.database_config import DatabaseSettings
from src.services.database import DatabaseManager
from langchain_core.tools import BaseTool
from src.config.logger import logger
from pydantic import Field, BaseModel
from typing import Type, List, Tuple
import pandas as pd
from typing import Dict, Any, Optional


def _get_random_function(db_settings: DatabaseSettings) -> str:
    """Get database-specific random function.

    Args:
        db_settings: Database configuration settings

    Returns:
        str: SQL random function for current database

    Raises:
        ValueError: If database type not supported
    """
    SQL_RANDOM_FUNCTIONS = {
        "sqlite": "RANDOM()",
        "athena": "rand()",
        "postgresql": "RANDOM()",
        "mysql": "RAND()",
        "mssql": "NEWID()",
        "oracle": "DBMS_RANDOM.VALUE",
    }
    db_type = db_settings.database_type.value.lower()
    if db_type not in SQL_RANDOM_FUNCTIONS:
        raise ValueError(f"Unsupported database type: {db_type}")
    return SQL_RANDOM_FUNCTIONS[db_type]


def _get_limit_syntax(db_settings: DatabaseSettings, sample_size: int) -> Dict[str, str]:
    """Get database-specific limit syntax.

    Args:
        db_settings: Database configuration settings
        sample_size: Number of records to limit

    Returns:
        Dict with prefixes and suffixes for query

    Raises:
        ValueError: If database type not supported
    """
    db_type = db_settings.database_type.value.lower()
    
    if db_type == "mssql":
        return {
            "select_prefix": f"TOP {sample_size}",
            "limit_suffix": ""
        }
    elif db_type in ["sqlite", "postgresql", "mysql", "athena", "oracle"]:
        return {
            "select_prefix": "",
            "limit_suffix": f"LIMIT {sample_size}"
        }
    else:
        raise ValueError(f"Unsupported database type for limit syntax: {db_type}")


def tool_get_random_subsamples(
    tables: List[Dict[str, Any]],
    sample_size: int = 5,
    db_settings: Optional[DatabaseSettings] = None,
    db_manager: Optional[DatabaseManager] = None,
) -> Dict[str, Any]:
    """Retrieve random data samples from specified database tables.
    
    Args:
        tables: List of tables with their columns to sample
               [{"table_name": "table1", "noun_columns": ["col1", "col2"]}, ...]
        sample_size: Number of random rows to retrieve per table
        db_settings: Optional database settings
        db_manager: Optional database manager instance
    
    Returns:
        Dict containing:
            - samples: Dictionary mapping table names to lists of row dictionaries
            - error: Error message if operation failed (None otherwise)
    """
    logger.info("🛠️ Executing get_random_subsamples tool")
    
    try:
        # Initialize database connection if not provided
        if not db_manager or not db_settings:
            db_settings = DatabaseSettings()
            db_manager = DatabaseManager(settings=db_settings)
        
        samples = {}
        
        for table in tables:
            table_name = table["table_name"]
            noun_columns = table["noun_columns"]
            
            # Get database-specific limit syntax
            limit_syntax = _get_limit_syntax(db_settings, sample_size)
            
            query = f"""
                SELECT {limit_syntax['select_prefix']} {", ".join(noun_columns)}
                FROM {table_name} 
                ORDER BY {_get_random_function(db_settings)}
                {limit_syntax['limit_suffix']}
            """
            
            try:
                results = db_manager.execute_query(query)
                
                # Create properly structured data
                samples[table_name] = []
                
                # Handle the case where results is a tuple with header row and data row
                if (isinstance(results, tuple) and len(results) >= 2 and 
                    isinstance(results[0], list) and isinstance(results[1], list)):
                    
                    # Extract header row and data rows
                    header_row = results[0]
                    data_rows = results[1]
                    
                    # Check if header_row contains column names
                    if all(col in noun_columns for col in header_row):
                        # Process each row in the data_rows list
                        for row_data in data_rows:
                            if isinstance(row_data, tuple) and len(row_data) == len(header_row):
                                # Create a dictionary for each record
                                record_dict = dict(zip(header_row, row_data))
                                samples[table_name].append(record_dict)
                else:
                    # Handle other result formats (fallback to original logic)
                    for row in results:
                        if isinstance(row, tuple) and len(row) >= len(noun_columns):
                            processed_row = dict(zip(noun_columns, row[:len(noun_columns)]))
                        else:
                            processed_row = dict(zip(noun_columns, row))
                        samples[table_name].append(processed_row)
                
                # Create columnar format for logging
                if samples[table_name]:
                    # Convert row-based data to columnar format
                    columnar_data = {}
                    for column in samples[table_name][0].keys():
                        columnar_data[column] = [row.get(column) for row in samples[table_name]]
                    
                    # Format the columnar data for better readability
                    formatted_sample = "\n"
                    for column, values in columnar_data.items():
                        formatted_sample += f"  {column}: {values}\n"
                    
                    logger.info(f"Sample data (columnar format):{formatted_sample}")
            except Exception as e:
                logger.error(f"Error sampling table {table_name}: {str(e)}")
                samples[table_name] = []
        
        return {
            "samples": samples,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving random samples: {str(e)}")
        return {
            "samples": {},
            "error": str(e)
        }