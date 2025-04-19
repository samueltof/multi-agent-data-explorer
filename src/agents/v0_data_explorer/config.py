import os
from pathlib import Path
# Move imports into functions to avoid circular dependency at module load time
# from src.services.llm import LLM 
# from src.services.database.database_manager import DatabaseManager
# Make imports absolute from src directory
from src.config.database_config import DatabaseSettings, DatabaseType
from src.config.logger import logger

# Determine project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# Remove outdated hardcoded path
# SQLITE_DB_PATH = PROJECT_ROOT / "data" / "multi_agent_explorer.db"

# Remove directory creation logic, moved into get_db_manager
# SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_llm(): # Remove type hint that requires early import
    """Initializes and returns the LLM instance (OpenAI)."""
    from src.services.llm import LLM # Import inside function
    try:
        # Explicitly use OpenAI provider as requested
        return LLM(provider="openai")
    except ValueError as e:
        logger.error(f"Error initializing LLM: {e}. Make sure OPENAI_API_KEY is set.")
        raise

def get_db_manager(): # Remove type hint that requires early import
    """Initializes and returns the DatabaseManager instance based on environment settings."""
    from src.services.database.database_manager import DatabaseManager # Import inside function
    try:
        # Load settings from environment variables via pydantic
        settings = DatabaseSettings()
        logger.info(f"Loaded Database Settings: Type={settings.database_type}, Name={settings.database_name}, Schema Path={settings.database_schema_path}")
        
        # Ensure the type is SQLite as expected for this agent setup (optional check)
        # if settings.database_type != DatabaseType.SQLITE:
        #     raise ValueError(f"Database type mismatch: Expected SQLITE, found {settings.database_type}")
            
        # For SQLite, ensure the database_name (used as path) directory exists
        if settings.database_type == DatabaseType.SQLITE and settings.database_name:
            db_path = Path(settings.database_name)
            # Handle relative paths from project root if necessary
            if not db_path.is_absolute():
                # Use PROJECT_ROOT defined globally in this module
                db_path = PROJECT_ROOT / db_path
                # Update settings object with absolute path for DatabaseManager
                settings.database_name = str(db_path)
            
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensuring SQLite DB directory exists at: {db_path.parent}")

        return DatabaseManager(settings=settings)
    except Exception as e:
        logger.error(f"Error initializing DatabaseManager from settings: {e}")
        raise 