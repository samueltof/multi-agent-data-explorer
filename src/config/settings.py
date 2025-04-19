import asyncio
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
import os

# --- Calculate path and Load .env synchronously at module TOP level --- 
# Calculate project root relative to this file's location
# settings.py -> config -> src -> project_root
_PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent 
_DOTENV_PATH = _PROJECT_ROOT_PATH / ".env"

if _DOTENV_PATH.is_file():
    print(f"Loading .env file SYNCHRONOUSLY at module level from: {_DOTENV_PATH}")
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
else:
    print(f"Warning: .env file not found at: {_DOTENV_PATH}")
# ---------------------------------------------------------------------

# Now import other modules that might use os.getenv during definition
from .llm_config import LLMSettings
from .database_config import DatabaseSettings

# Class definitions happen after load_dotenv
class Settings(BaseSettings):
    agent_name: str = "agent"
    # Use default_factory
    llm: LLMSettings = Field(default_factory=LLMSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        # Keep this None to prevent Pydantic re-loading
        env_file = None 
        extra = 'ignore'

async def get_settings_async() -> Settings:
    """Asynchronously gets the settings object. .env should be pre-loaded."""
    # Environment should already be populated by module-level load
    return Settings()

# Consider an async getter if the application structure allows:
# async def get_settings_async() -> Settings:
#    await load_dotenv_async()
#    return Settings()