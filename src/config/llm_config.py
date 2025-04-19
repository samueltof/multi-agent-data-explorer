from pydantic_settings import BaseSettings
from pydantic import Field, validator, Extra
from typing import Optional, List
import os
from dotenv import load_dotenv

# Explicitly load .env before defining classes that use os.getenv
load_dotenv()

class LLMProviderSettings(BaseSettings):
    """Base settings for LLM providers."""
    
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3
    
    
class AnthropicSettings(LLMProviderSettings):
    """Settings for the Anthropic LLM provider."""
    
    api_key: str = os.getenv("ANTHROPIC_API_KEY")
    defaul_model: str = "claude-3-haiku-20240307"
    embedding_model: str = "voyage-3-lite"
    

class OpenAISettings(LLMProviderSettings):
    """Settings for the OpenAI LLM provider."""
    
    api_key: str = os.getenv("OPENAI_API_KEY")
    default_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    
    
class BedrockSettings(LLMProviderSettings):
    """Settings for the Bedrock LLM provider."""
    
    default_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    embedding_model: str = "amazon.titan-embed-text-v1"
    region: str = os.getenv("AWS_REGION")

class AzureSettings(LLMProviderSettings):
    """Settings for the Azure OpenAI provider."""
    
    api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    azure_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    default_model: str = "gpt-4"
    # Update embedding model name to match Azure deployment naming convention
    embedding_model: str = "text-embedding-ada-002"  # Changed from text-embedding-3-small
    
    # Add fallback options
    embedding_model_fallbacks: list = ["text-embedding-ada-002", "text-embedding-3-small"]
    
    class Config:
        validate_assignment = True
        
        
class PortkeyBedrockSettings(LLMProviderSettings):
    """Settings for the Portkey LLM gateway provider with Bedrock."""
    
    api_key: Optional[str] = os.getenv("PORTKEY_BEDROCK_API_KEY", None)
    virtual_key: Optional[str] = os.getenv("PORTKEY_BEDROCK_VIRTUAL_KEY", None)
    base_url: str = os.getenv("PORTKEY_BASE_URL", "https://api.portkey.ai/v1")
    default_model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    embedding_model: str = "amazon.titan-embed-text-v1"
    
    # Portkey specific settings
    trace_id: Optional[str] = None
    cache_enabled: bool = True
    retry_enabled: bool = True
    
    class Config:
        validate_assignment = True
        
class PortkeyAzureSettings(LLMProviderSettings):
    """Settings for the Portkey LLM gateway provider with Azure."""
    
    api_key: Optional[str] = os.getenv("PORTKEY_AZURE_API_KEY", None)
    virtual_key: Optional[str] = os.getenv("PORTKEY_AZURE_VIRTUAL_KEY", None)
    base_url: str = os.getenv("PORTKEY_BASE_URL", "https://api.portkey.ai/v1")
    default_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-ada-002"
    
    # Portkey specific settings
    trace_id: Optional[str] = None
    cache_enabled: bool = True
    retry_enabled: bool = True
    
    class Config:
        validate_assignment = True
    
class LLMSettings(BaseSettings):
    """Settings for the LLM agent."""
    
    anthropic: AnthropicSettings = AnthropicSettings()
    openai: OpenAISettings = OpenAISettings()
    bedrock: BedrockSettings = BedrockSettings()
    azure: AzureSettings = AzureSettings()
    portkey_bedrock: PortkeyBedrockSettings = PortkeyBedrockSettings()
    portkey_azure: PortkeyAzureSettings = PortkeyAzureSettings()
    
    class Config:
        validate_assignment = True