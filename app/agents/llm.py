"""LLM client factories.

Two providers, one code path:
  azure             Azure OpenAI / Microsoft Foundry. API key if given, otherwise Entra ID
                    (Managed Identity in Container Apps, `az login` locally).
  openai_compatible any server that speaks the OpenAI API - Ollama, vLLM, NVIDIA NIM, LM Studio.
                    This is how the agents run fully on-prem on an open-source model
                    (e.g. Qwen 2.5 or Llama 3.1) when data can't leave the building.

`llm_client` returns an OpenAI SDK client (used by the classic agent);
`chat_model` returns a LangChain chat model (used by the LangGraph agents)."""
from __future__ import annotations

from openai import AzureOpenAI, OpenAI

from app.config import Settings

_SCOPE = "https://cognitiveservices.azure.com/.default"


def azure_openai_client(settings: Settings) -> AzureOpenAI:
    if settings.azure_openai_api_key:
        return AzureOpenAI(api_key=settings.azure_openai_api_key, azure_endpoint=settings.azure_openai_endpoint,
                           api_version=settings.azure_openai_api_version)
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token = get_bearer_token_provider(DefaultAzureCredential(), _SCOPE)
    return AzureOpenAI(azure_ad_token_provider=token, azure_endpoint=settings.azure_openai_endpoint,
                       api_version=settings.azure_openai_api_version)


def llm_client(settings: Settings):
    """OpenAI SDK client for whichever provider is configured (None if no LLM is configured)."""
    if settings.use_azure_openai:
        return azure_openai_client(settings)
    if settings.llm_provider == "openai_compatible" and settings.llm_base_url:
        return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return None


def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith(("gpt-5", "o1", "o3", "o4")) or "-o1" in n or "-o3" in n


def chat_model(settings: Settings, model_hint: str | None = None):
    """LangChain chat model for the LangGraph agents (None if no LLM is configured).

    Reasoning models (gpt-5*, o-series) reject a custom temperature, so they get a low
    reasoning effort instead; everything else gets temperature 0.2."""
    name = model_hint or settings.chat_model_name
    if settings.use_azure_openai:
        from langchain_openai import AzureChatOpenAI

        # the deployment name ("chat") doesn't say which model it is, so we also check the model name
        reasoning = is_reasoning_model(name) or is_reasoning_model(settings.azure_openai_chat_model)
        kw = dict(azure_deployment=name, azure_endpoint=settings.azure_openai_endpoint,
                  api_version=settings.azure_openai_api_version, max_retries=2, timeout=60)
        if settings.azure_openai_api_key:
            kw["api_key"] = settings.azure_openai_api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            kw["azure_ad_token_provider"] = get_bearer_token_provider(DefaultAzureCredential(), _SCOPE)
        if reasoning:
            kw["reasoning_effort"] = settings.llm_reasoning_effort
        else:
            kw["temperature"] = 0.2
        return AzureChatOpenAI(**kw)
    if settings.llm_provider == "openai_compatible" and settings.llm_base_url:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                          temperature=0.2, max_retries=2, timeout=120)
    return None
