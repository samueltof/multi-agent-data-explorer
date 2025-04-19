import os
from enum import Enum
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings

# Print the path of the .env file being loaded
env_path = find_dotenv()
# Load environment variables WITH OVERRIDE ENABLED
load_dotenv(dotenv_path=env_path, override=True)

class DatabaseType(Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    ATHENA = "athena"
    BIGQUERY = "bigquery"
    MSSQL = "mssql"

class DatabaseSettings(BaseSettings):
    # Common settings
    database_type: DatabaseType = DatabaseType(os.getenv("DATABASE_TYPE", "athena"))
    database_name: str | None = os.getenv("DATABASE_NAME")
    
    # Schema information
    database_schema_path: str | None = os.getenv("DATABASE_SCHEMA_PATH")
    
    # AWS Athena specific
    s3_staging_dir: str | None = os.getenv("S3_STAGING_DIR", None)
    region_name: str | None = os.getenv("AWS_REGION", None)
    
    # SQL database specific
    host: str | None = os.getenv("DB_HOST", None)
    port: int | None = int(os.getenv("DB_PORT", None)) if os.getenv("DB_PORT") else None
    username: str | None = os.getenv("DB_USERNAME", None) 
    password: str | None = os.getenv("DB_PASSWORD", None)
    local: bool = False
    
    # SQLite specific
    sqlite_path: str | None = os.getenv("SQLITE_PATH")
    
    # Vector database paths
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "data/lancedb")
    query_examples_path: str = os.getenv("QUERY_EXAMPLES_PATH")
    vector_table_name: str = os.getenv("VECTOR_TABLE_NAME", "sql_query_pairs")
    
    model_config = {
        "validate_assignment": True,
        "env_file": ".env",
        "extra": "ignore"
    }