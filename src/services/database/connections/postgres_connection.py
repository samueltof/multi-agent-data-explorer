import pandas as pd
from typing import List, Any, Tuple
from config.logger import logger
from config.database_config import DatabaseSettings
from services.database.database_connection import DatabaseConnection


class PostgresConnection(DatabaseConnection):
    def __init__(self, settings: DatabaseSettings):
        logger.info("⎄ Initializing PostgreSQL connection")
        self.settings = settings
        self.conn = None

    def connect(self):
        logger.info(
            f"⎄ Connecting to PostgreSQL at {self.settings.host}:{self.settings.port}"
        )
        try:
            import psycopg2

            self.conn = psycopg2.connect(
                dbname=self.settings.database_name,
                user=self.settings.username,
                password=self.settings.password,
                host=self.settings.host,
                port=self.settings.port,
            )
            logger.info("⎄ Successfully connected to PostgreSQL")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise

    def execute_query(self, query: str) -> Tuple[List[str], List[Any]]:
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            # Get column names and data
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return columns, data
        except Exception as e:
            logger.error(f"Failed to execute PostgreSQL query: {str(e)}")
            raise

    def execute_query_df(self, query: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Failed to execute PostgreSQL query to DataFrame: {str(e)}")
            raise
