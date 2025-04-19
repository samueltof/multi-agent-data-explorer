from src.config.settings import get_settings, Settings
from src.services.embeddings import BaseEmbeddingsService, OllamaEmbeddingsService, PortkeyBedrockEmbeddingsService
from pydantic import ValidationError
import os
from typing import Any, Optional
import time
import logging

class Embeddings:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        try:
            self.settings = get_settings()
        except ValidationError:
            self.settings = Settings()
            
        self.available_providers = self._get_available_providers()
        self.provider = provider or self._determine_default_provider()
        self._model_name = model  # Store the model name separately from the property
        
        # Check if provider is available
        if self.provider not in self.available_providers:
            raise ValueError(f"Provider {self.provider} is not available or properly configured")
        
        # Initialize only the selected provider
        self.model = self._initialize_provider(self.provider)
    
    def _get_available_providers(self) -> list:
        providers = []
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append("anthropic")
        if os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
        if os.getenv("AWS_REGION"):
            providers.append("bedrock")
        if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
            providers.append("azure")
        if os.getenv("PORTKEY_BEDROCK_API_KEY") and os.getenv("PORTKEY_BEDROCK_VIRTUAL_KEY"):
            providers.append("portkey_bedrock")
        if not providers:
            raise ValueError("No valid embedding provider credentials found")
        return providers

    def _determine_default_provider(self) -> str:
        return self.available_providers[0]

    def _initialize_provider(self, provider: str) -> Any:
        """Initialize only the specified provider"""
        if provider == "anthropic":
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = self._model_name or "sentence-transformers/all-mpnet-base-v2"
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=self._model_name or self.settings.llm.openai.embedding_model
            )
        elif provider == "bedrock":
            from langchain_aws import BedrockEmbeddings
            return BedrockEmbeddings(
                model_id=self._model_name or self.settings.llm.bedrock.embedding_model,
                region_name=self.settings.llm.bedrock.region
            )
        elif provider == "azure":
            from langchain_openai import AzureOpenAIEmbeddings
            deployment_model = self._model_name or self.settings.llm.azure.embedding_model
            deployment_ready = self._verify_azure_deployment(
                endpoint=self.settings.llm.azure.azure_endpoint,
                deployment=deployment_model,
                api_version=self.settings.llm.azure.api_version
            )
            
            if not deployment_ready:
                raise RuntimeError(f"Azure deployment {deployment_model} not found after retries")
                
            return AzureOpenAIEmbeddings(
                deployment=deployment_model,
                model=deployment_model,
                azure_endpoint=self.settings.llm.azure.azure_endpoint,
                api_version=self.settings.llm.azure.api_version,
                max_retries=3
            )
        elif provider == "portkey_bedrock":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=self._model_name or self.settings.llm.portkey.embedding_model,
                api_key=self.settings.llm.portkey.api_key,
                base_url=self.settings.llm.portkey.base_url,
            )
        else:
            raise ValueError(f"Provider {provider} is not supported")

    def _verify_azure_deployment(self, endpoint: str, deployment: str, api_version: str, max_retries: int = 3) -> bool:
        """Verify Azure deployment exists and is ready"""
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        for attempt in range(max_retries):
            try:
                client.embeddings.create(
                    model=deployment,
                    input="test",
                )
                return True
            except Exception as e:
                if "DeploymentNotFound" in str(e):
                    wait_time = 2 ** attempt
                    logging.warning(f"Deployment {deployment} not ready, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        return False

    @property
    def model_name(self):
        """Get the model name that was used for initialization."""
        return self._model_name

_settings = get_settings()

def get_embeddings_service() -> BaseEmbeddingsService:
    # Implementation of get_embeddings_service function
    pass
