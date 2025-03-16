from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union, Sequence, cast

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


class AzureAuthChatOpenAI(AzureChatOpenAI):
    """
    Azure OpenAI chat model with DefaultAzureCredential authentication support.
    This implementation extends the LangChain AzureChatOpenAI class to support 
    Azure Active Directory authentication via DefaultAzureCredential.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "azure-openai-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        params = super()._identifying_params
        params["model_type"] = "AzureAuthChatOpenAI"
        return params

    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        azure_ad_scope: str = "https://cognitiveservices.azure.com/.default",
        verbose: bool = False,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Azure OpenAI chat model with DefaultAzureCredential.

        Args:
            deployment_name: Name of the Azure OpenAI deployment (model deployment)
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            azure_ad_scope: Azure AD scope to use for authentication
            verbose: Whether to enable verbose logging
            model_name: Optional model name for identification (e.g., "gpt-4o")
            **kwargs: Additional arguments to pass to the AzureChatOpenAI constructor
        """
        # Create token provider with DefaultAzureCredential
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), azure_ad_scope
        )
        
        # Allow both LangChain 0.1.x and 0.2.x style initialization
        if "model" not in kwargs and model_name is not None:
            kwargs["model"] = model_name
        
        # Initialize the parent class with Azure AD token provider
        super().__init__(
            deployment_name=deployment_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
            verbose=verbose,
            **kwargs,
        )
        
        # Ensure compatibility with computeruse model name detection
        self.model_name = model_name or deployment_name
        
        # Store the token provider for potential refresh
        self.azure_ad_token_provider = token_provider
        logger.info(f"Initialized Azure OpenAI with deployment: {deployment_name}")


def create_azure_openai(
    deployment_name: str,
    azure_endpoint: Optional[str] = None,
    api_version: str = "2024-02-15-preview",
    timeout: Optional[int] = None,
    max_retries: int = 3,
    streaming: bool = False,
    model_name: Optional[str] = None,
) -> AzureAuthChatOpenAI:
    """
    Create an Azure OpenAI model with DefaultAzureCredential authentication.

    Args:
        deployment_name: Name of the Azure OpenAI deployment
        azure_endpoint: Azure OpenAI endpoint URL. If None, loads from env var AZURE_OPENAI_ENDPOINT
        api_version: Azure OpenAI API version
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        streaming: Whether to stream responses
        model_name: Optional model name for identification (e.g., "gpt-4o")

    Returns:
        AzureAuthChatOpenAI: Azure OpenAI chat model with AAD auth
    """
    logger.info(f"Creating Azure OpenAI client for deployment: {deployment_name}")
    
    # Load endpoint from environment if not provided
    if azure_endpoint is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError(
                "azure_endpoint must be provided or set in AZURE_OPENAI_ENDPOINT env var"
            )
    
    logger.info(f"Using Azure OpenAI endpoint: {azure_endpoint}")
    logger.info(f"Using API version: {api_version}")
    
    # Ensure the endpoint is properly formatted
    if not azure_endpoint.startswith("https://"):
        logger.warning(f"Azure endpoint doesn't start with https://, attempting to fix: {azure_endpoint}")
        azure_endpoint = f"https://{azure_endpoint}"
    
    # If model_name not provided, use deployment name with a common prefix mapping
    if model_name is None:
        # Try to map deployment name to a common model name for better compatibility
        if "gpt-4" in deployment_name.lower():
            if "vision" in deployment_name.lower() or "o" in deployment_name.lower():
                model_name = "gpt-4o"
            else:
                model_name = "gpt-4"
        elif "gpt-35" in deployment_name.lower() or "gpt3" in deployment_name.lower():
            model_name = "gpt-3.5-turbo"
        elif "o1" in deployment_name.lower():
            model_name = "o1"
        else:
            # Default to using deployment name as model name
            model_name = deployment_name
            
    logger.info(f"Mapped deployment {deployment_name} to model_name: {model_name}")
    
    # Create client with better error handling
    try:
        client = AzureAuthChatOpenAI(
            deployment_name=deployment_name,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout or 120,  # Ensure a reasonable default timeout
            max_retries=max_retries,
            streaming=streaming,
            model_name=model_name,
        )
        
        logger.info(f"Successfully created Azure OpenAI client for {model_name}")
        return client
    except Exception as e:
        logger.error(f"Error creating Azure OpenAI client: {e}")
        
        # Helpful error message based on common issues
        if "authenticate" in str(e).lower() or "authorization" in str(e).lower():
            logger.error("Authentication error: Make sure your Azure credentials are properly configured.")
            logger.error("Check that DefaultAzureCredential has access to your Azure OpenAI resource.")
        elif "not found" in str(e).lower() or "404" in str(e).lower():
            logger.error(f"Resource not found: Check that the deployment name '{deployment_name}' exists in your Azure OpenAI resource.")
        elif "timeout" in str(e).lower():
            logger.error("Connection timeout: Check your network connection and Azure OpenAI endpoint URL.")
        
        # Re-raise the original exception
        raise