from typing import List, Optional
from langchain_core.embeddings import Embeddings
from config.settings import get_settings

class PortkeyBedrockEmbeddings(Embeddings):
    """Implementation of Bedrock embeddings through Portkey gateway."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        trace_id: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        retry_enabled: Optional[bool] = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        
        # Get settings
        settings = get_settings()
        portkey_settings = settings.llm.portkey_bedrock
        
        # Set up client with Portkey configuration
        self.api_key = portkey_settings.api_key
        self.virtual_key = portkey_settings.virtual_key
        self.base_url = portkey_settings.base_url
        self.model = model or portkey_settings.embedding_model
        
        # Portkey specific settings
        self.trace_id = trace_id or portkey_settings.trace_id
        self.cache_enabled = cache_enabled if cache_enabled is not None else portkey_settings.cache_enabled
        self.retry_enabled = retry_enabled if retry_enabled is not None else portkey_settings.retry_enabled
        
        # Configure additional headers for Portkey
        headers = {
            "X-Portkey-Api-Key": self.api_key,
            "X-Virtual-Key": self.virtual_key,
        }
        
        if self.trace_id:
            headers["X-Trace-Id"] = self.trace_id
        
        # Configure parameters for the Portkey API
        params = {
            "cache": str(self.cache_enabled).lower(),
            "retry": str(self.retry_enabled).lower(),
        }
        
        self.client = openai.OpenAI(
            api_key=self.api_key,  # Will be overridden by headers
            base_url=self.base_url,
            default_headers=headers,
            default_query=params
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Bedrock via Portkey"""
        embeddings = []
        # Process in batches to improve efficiency
        batch_size = 20  # Adjust based on your requirements and rate limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Portkey with Bedrock allows batching
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            
            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Bedrock via Portkey"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        return response.data[0].embedding
    