import pandas as pd
import threading
from typing import List, Any, Tuple
from pathlib import Path
from src.config.logger import logger
from src.config.database_config import DatabaseSettings
from src.services.database.database_connection import DatabaseConnection


class SQLiteConnection(DatabaseConnection):
    def __init__(self, settings: DatabaseSettings):
        logger.info("⎄ Initializing SQLite connection")
        self.settings = settings
        # Use thread-local storage for the connection object
        self.thread_local = threading.local()
        # Ensure conn attribute doesn't exist directly on the instance initially
        # The connection will be stored under self.thread_local.conn
        self.db_path = self._resolve_db_path()

    def _resolve_db_path(self) -> str:
        # Get project root from current file location
        # Go up 5 levels from src/services/database/connections/sqlite_connection.py
        project_root = Path(__file__).parent.parent.parent.parent.parent
        
        db_path_str = "<path not specified or found>"

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
            else:
                # Keep track of the path tried for the error message
                db_path_str = str(db_path)

        raise FileNotFoundError(f"SQLite database not specified or not found at the checked path: {db_path_str}")

    def _get_thread_connection(self):
        """Gets the connection for the current thread, creating it if necessary."""
        # Check if connection exists for the *current* thread
        if not hasattr(self.thread_local, 'conn') or self.thread_local.conn is None:
            logger.info(f"⎄ Connecting to SQLite database at {self.db_path} for thread {threading.current_thread().ident}")
            try:
                import sqlite3
                # Store the connection in thread-local storage
                # check_same_thread=False allows the connection object to be potentially passed
                # between threads *if needed*, but our primary mechanism is one connection per thread.
                self.thread_local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                logger.info(f"⎄ Successfully connected to SQLite for thread {threading.current_thread().ident}")
            except Exception as e:
                logger.error(f"Failed to connect to SQLite for thread {threading.current_thread().ident}: {str(e)}")
                # Ensure conn is None in thread_local if connection fails
                self.thread_local.conn = None
                raise
        
        return self.thread_local.conn

    # connect() method might be called externally, ensure it uses the getter
    def connect(self):
        return self._get_thread_connection()

    def execute_query(self, query: str) -> Tuple[List[str], List[Any]]:
        # Get connection for the current thread
        conn = self._get_thread_connection()
        if not conn:
             raise ConnectionError(f"SQLite connection not established for thread {threading.current_thread().ident}")
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            # Ensure cursor description is not None before accessing
            columns = [description[0] for description in cursor.description] if cursor.description else []
            results = cursor.fetchall()
            return columns, results
        except Exception as e:
            logger.error(f"Failed to execute SQLite query for thread {threading.current_thread().ident}: {str(e)}")
            raise

    def execute_query_df(self, query: str) -> pd.DataFrame:
        # Get connection for the current thread
        conn = self._get_thread_connection()
        if not conn:
             raise ConnectionError(f"SQLite connection not established for thread {threading.current_thread().ident}")
        try:
            # Pass the actual connection object to pandas
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Failed to execute SQLite query to DataFrame for thread {threading.current_thread().ident}: {str(e)}")
            raise

    def close(self):
        # Close connection only if it exists for the *current* thread
        if hasattr(self.thread_local, 'conn') and self.thread_local.conn is not None:
            logger.info(f"⎄ Closing SQLite connection for thread {threading.current_thread().ident}")
            self.thread_local.conn.close()
            self.thread_local.conn = None
