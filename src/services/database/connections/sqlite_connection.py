import pandas as pd
from typing import List, Any, Tuple
from pathlib import Path
from config.logger import logger
from config.database_config import DatabaseSettings
from services.database.database_connection import DatabaseConnection


class SQLiteConnection(DatabaseConnection):
    def __init__(self, settings: DatabaseSettings):
        logger.info("⎄ Initializing SQLite connection")
        self.settings = settings
        self.conn = None
        self.db_path = self._resolve_db_path()

    def _resolve_db_path(self) -> str:
        # Get project root from current file location
        project_root = Path(__file__).parent.parent.parent.parent

        if self.settings.sqlite_path:
            # Handle both absolute and relative paths
            if Path(self.settings.sqlite_path).is_absolute():
                db_path = Path(self.settings.sqlite_path)
            else:
                # Use relative path from project root
                db_path = project_root / self.settings.sqlite_path

            logger.info(f"Attempting database path: {db_path}")
            if db_path.exists():
                return str(db_path)

        raise FileNotFoundError(f"SQLite database not found at: {db_path}")

    def connect(self):
        if self.conn is not None:
            return self.conn

        logger.info(f"⎄ Connecting to SQLite database at {self.db_path}")
        try:
            import sqlite3

            self.conn = sqlite3.connect(self.db_path)
            logger.info("⎄ Successfully connected to SQLite")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {str(e)}")
            raise

    def execute_query(self, query: str) -> Tuple[List[str], List[Any]]:
        if not self.conn:
            self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = cursor.fetchall()
            return columns, results
        except Exception as e:
            logger.error(f"Failed to execute SQLite query: {str(e)}")
            raise

    def execute_query_df(self, query: str) -> pd.DataFrame:
        try:
            if not self.conn:
                self.connect()
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Failed to execute SQLite query to DataFrame: {str(e)}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("⎄ SQLite connection closed")
