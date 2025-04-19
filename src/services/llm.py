from langchain_core.prompts import PromptTemplate
from config.settings import get_settings, Settings
from pydantic import ValidationError
import os
from typing import Any, Optional
import time
import logging

class LLM:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        try:
            self.settings = get_settings()
        except ValidationError:
            # Fallback to empty settings if validation fails
            self.settings = Settings()
        
        self.available_providers = self._get_available_providers()
        self.provider = provider or self._determine_default_provider()
        self.model = model
        
        # Initialize only the specified provider
        if self.provider not in self.available_providers:
            raise ValueError(f"Provider {self.provider} is not available or properly configured")
        
        self.llm = self._initialize_provider(self.provider)
        
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
        if os.getenv("PORTKEY_AZURE_API_KEY") and os.getenv("PORTKEY_AZURE_VIRTUAL_KEY"):
            providers.append("portkey_azure")
        if not providers:
            raise ValueError("No valid LLM provider credentials found")
        return providers

    def _determine_default_provider(self) -> str:
        return self.available_providers[0]

    def _initialize_provider(self, provider: str) -> Any:
        """Initialize only the specified provider"""
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model or self.settings.llm.anthropic.defaul_model,
                temperature=self.settings.llm.anthropic.temperature,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model or self.settings.llm.openai.default_model,
                temperature=self.settings.llm.openai.temperature,
            )
        elif provider == "bedrock":
            from langchain_aws import ChatBedrock
            return ChatBedrock(
                model_id=self.model or self.settings.llm.bedrock.default_model,
                region=self.settings.llm.bedrock.region,
                model_kwargs={"temperature": self.settings.llm.bedrock.temperature},
            )
        elif provider == "azure":
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                deployment_name=self.model or self.settings.llm.azure.default_model,
                temperature=self.settings.llm.azure.temperature,
                azure_endpoint=self.settings.llm.azure.azure_endpoint,
                api_version=self.settings.llm.azure.api_version,
            )
        elif provider == "portkey_bedrock":
            from langchain_openai import ChatOpenAI
            from portkey_ai import createHeaders
            
            portkey_headers = createHeaders(
                api_key=self.settings.llm.portkey_bedrock.api_key,
                virtual_key=self.settings.llm.portkey_bedrock.virtual_key
            )
            
            return ChatOpenAI(
                api_key="X",  # Dummy API key as we're using headers for auth
                base_url=self.settings.llm.portkey_bedrock.base_url,
                default_headers=portkey_headers,
                model=self.model or self.settings.llm.portkey_bedrock.default_model,
                temperature=self.settings.llm.portkey_bedrock.temperature,
            )
        elif provider == "portkey_azure":
            from langchain_openai import ChatOpenAI
            from portkey_ai import createHeaders
            
            portkey_headers = createHeaders(
                api_key=self.settings.llm.portkey_azure.api_key,
                provider="openai"  # Specify provider for Azure integration
            )
            
            return ChatOpenAI(
                api_key=self.settings.llm.portkey_azure.api_key,  # Using API key here as required by Portkey Azure
                base_url=self.settings.llm.portkey_azure.base_url,
                default_headers=portkey_headers,
                model=self.model or self.settings.llm.portkey_azure.default_model,
                temperature=self.settings.llm.portkey_azure.temperature,
            )
        else:
            raise ValueError(f"Provider {provider} is not supported")

    def invoke(self, prompt: PromptTemplate, **kwargs) -> str:
        response = self.llm.invoke(prompt)
        return response.content
    
    @property
    def client(self):
        return self.llm


def verify_azure_deployment(endpoint: str, deployment: str, api_version: str, max_retries: int = 3) -> bool:
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
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"Deployment {deployment} not ready, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    return False
