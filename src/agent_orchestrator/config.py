from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-agent-orchestrator")
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Project 1 Public API Connection
    rag_engine_base_url: str = Field(default="http://localhost:8000")
    rag_engine_timeout_seconds: float = Field(default=10.0, ge=0.5, le=120.0)
    rag_engine_max_retries: int = Field(default=2, ge=0, le=5)

    # Provider-Agnostic LLM Settings
    llm_provider_name: str = Field(default="groq")
    llm_base_url: str = Field(default="https://api.groq.com/openai/v1")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="llama-3.1-8b-instant")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_timeout_seconds: float = Field(default=20.0, ge=1.0, le=120.0)

    # External FX API & Agent Guardrails
    fx_api_base_url: str = Field(default="https://api.frankfurter.app")
    fx_timeout_seconds: float = Field(default=5.0, ge=0.5, le=30.0)
    default_currency: str = Field(default="INR")
    agent_max_steps: int = Field(default=6, ge=2, le=20)

    # Persistence & Cost Tracking
    checkpoint_db_path: str = Field(default="agent_checkpoints.db")
    llm_cost_per_1m_input_tokens: float = Field(default=0.0, ge=0.0)
    llm_cost_per_1m_output_tokens: float = Field(default=0.0, ge=0.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()