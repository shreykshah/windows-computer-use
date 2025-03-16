"""
Direct Azure OpenAI Client implementation using Azure DefaultAzureCredential for authentication.
This module provides a way to use the Azure OpenAI API directly without LangChain.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import List, Optional, Union, Dict, Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from PIL import Image
import io

logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    A client for Azure OpenAI using DefaultAzureCredential for authentication.
    This class provides direct access to the Azure OpenAI API without using LangChain.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        azure_ad_scope: str = "https://cognitiveservices.azure.com/.default",
    ):
        """
        Initialize the Azure OpenAI client with DefaultAzureCredential.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            azure_ad_scope: Azure AD scope to use for authentication
        """
        # Create token provider with DefaultAzureCredential
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), azure_ad_scope
        )
        
        # Initialize the Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
            )
            logger.info(f"Successfully initialized Azure OpenAI client with endpoint: {azure_endpoint}")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise
        
        logger.info(f"Initialized Azure OpenAI client with endpoint: {azure_endpoint}")

    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def encode_image_from_pil(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Encode a PIL Image to base64.
        
        Args:
            image: PIL Image object
            format: Image format to save as (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text(
        self,
        model_deployment: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate text using the Azure OpenAI API.
        
        Args:
            model_deployment: Azure OpenAI model deployment name
            prompt: Text prompt to generate from
            temperature: Temperature for model generation
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort (e.g., "high")
            
        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
        
        kwargs = {
            "model": model_deployment,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def generate_with_vision(
        self,
        model_deployment: str,
        prompt: str,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate text with vision using the Azure OpenAI API.
        
        Args:
            model_deployment: Azure OpenAI model deployment name
            prompt: Text prompt to generate from
            image_path: Path to image file (optional)
            image: PIL Image object (optional)
            temperature: Temperature for model generation
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort (e.g., "high")
            
        Returns:
            Generated text response
        """
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
            
        if image_path is not None:
            image_b64 = self.encode_image(image_path)
        else:
            image_b64 = self.encode_image_from_pil(image)
            
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ],
            },
        ]
        
        kwargs = {
            "model": model_deployment,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


def create_azure_openai_client(
    azure_endpoint: Optional[str] = None,
    api_version: str = "2024-02-15-preview",
) -> AzureOpenAIClient:
    """
    Create an Azure OpenAI client with DefaultAzureCredential authentication.
    
    Args:
        azure_endpoint: Azure OpenAI endpoint URL. If None, loads from env var AZURE_OPENAI_ENDPOINT
        api_version: Azure OpenAI API version
        
    Returns:
        AzureOpenAIClient: Azure OpenAI client with AAD auth
    """
    # Load endpoint from environment if not provided
    if azure_endpoint is None:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError(
                "azure_endpoint must be provided or set in AZURE_OPENAI_ENDPOINT env var"
            )
    
    return AzureOpenAIClient(
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )