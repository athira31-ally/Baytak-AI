"""Azure OpenAI client factory. Uses an API key if given, otherwise Entra ID
(Managed Identity in Container Apps, `az login` locally)."""
from __future__ import annotations

from openai import AzureOpenAI

from app.config import Settings


def azure_openai_client(settings: Settings) -> AzureOpenAI:
    if settings.azure_openai_api_key:
        return AzureOpenAI(api_key=settings.azure_openai_api_key, azure_endpoint=settings.azure_openai_endpoint,
                           api_version=settings.azure_openai_api_version)
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(azure_ad_token_provider=token, azure_endpoint=settings.azure_openai_endpoint,
                       api_version=settings.azure_openai_api_version)
