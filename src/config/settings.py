from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from .llm_config import LLMSettings
from .database_config import DatabaseSettings

load_dotenv()

class Settings(BaseSettings):
    
    agent_name: str = "agent"
    llm: LLMSettings = LLMSettings()
    database: DatabaseSettings = DatabaseSettings()
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
    

@lru_cache()
def get_settings() -> Settings:
    """
    Get the settings object.
    """
    return Settings()