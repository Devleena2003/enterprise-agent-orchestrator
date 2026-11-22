from functools import lru_cache
from langchain_openai import ChatOpenAI

from agent_orchestrator.config import Settings, get_settings
from agent_orchestrator.exceptions import LLMConfigurationError


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """
    Create an OpenAI-compatible chat model pointing to Groq, Gemini, or local Ollama.
    """
    api_key = settings.llm_api_key.strip()
    if not api_key :
        raise LLMConfigurationError(
            "LLM_API_KEY is missing. Set a free Groq/Gemini key in .env "
            "(or set LLM_BASE_URL=http://localhost:11434/v1 and LLM_API_KEY=ollama)."
        )

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    return build_chat_model(get_settings())