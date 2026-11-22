from agent_orchestrator.config import get_settings
from langchain_openai import ChatOpenAI

settings = get_settings()

llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    temperature=0,
)

response = llm.invoke("Explain RAG in exactly two sentences.")

print(response.content)