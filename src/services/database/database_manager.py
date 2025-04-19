import yaml
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Any, Tuple
from src.config.logger import logger
from src.config.database_config import DatabaseSettings, DatabaseType
from src.services.database.database_connection import DatabaseConnection
from src.services.database.connections.athena_connection import AthenaConnection
from src.services.database.connections.postgres_connection import PostgresConnection
from src.services.database.connections.mysql_connection import MySQLConnection
from src.services.database.connections.sqlite_connection import SQLiteConnection
from src.services.database.connections.mssql_connection import MSSQLConnection

load_dotenv()

class DatabaseManager:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.conn = self._initialize_connection()
        
    def _initialize_connection(self) -> DatabaseConnection:
        if self.settings.database_type == DatabaseType.SQLITE:
            connection = SQLiteConnection(self.settings)
        elif self.settings.database_type == DatabaseType.POSTGRES:
            connection = PostgresConnection(self.settings)
        elif self.settings.database_type == DatabaseType.ATHENA:
            connection = AthenaConnection(self.settings)
        elif self.settings.database_type == DatabaseType.MYSQL:
            connection = MySQLConnection(self.settings)
        elif self.settings.database_type == DatabaseType.MSSQL:
            connection = MSSQLConnection(self.settings)
        else:
            raise ValueError(f"Unsupported database type: {self.settings.database_type}")
        
        connection.connect()
        return connection

    def execute_query(self, query: str) -> Tuple[List[str], List[Any]]:
        try:
            return self.conn.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            raise

    def execute_query_df(self, query: str) -> pd.DataFrame:
        try:
            return self.conn.execute_query_df(query)
        except Exception as e:
            logger.error(f"Failed to execute query to DataFrame: {str(e)}")
            raise

    def load_schema_description(self) -> str:
        """Loads and formats the schema description from YAML file."""
        # Get project root from current file location
        project_root = Path(__file__).parent.parent.parent.parent
        
        if self.settings.database_schema_path:
            # Handle both absolute and relative paths
            if Path(self.settings.database_schema_path).is_absolute():
                schema_path = Path(self.settings.database_schema_path)
            else:
                # Use relative path from project root
                schema_path = project_root / self.settings.database_schema_path
                
            logger.info(f"⎄ Attempting schema path: {schema_path}")
            
            # Validate path exists
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema description file not found at: {schema_path}")
                
            # Read and parse schema
            with open(schema_path, "r") as f:
                schema_data = yaml.safe_load(f)

            # Format the schema description
            description = "Available tables and their structures:\n\n"

            # Add tables and their columns 
            for table_name, table_info in schema_data["schema"]["tables"].items():
                description += f"{table_name}\n"
                for column in table_info["columns"]:
                    constraints = f", {column['constraints']}" if "constraints" in column else ""
                    description += f"- {column['name']} ({column['type']}{constraints})\n"
                description += "\n"

            return description
        else:
            raise ValueError("Database schema path not configured in settings")
