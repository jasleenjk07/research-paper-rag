import os

from src.config import LLM_MODEL

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm():
    """
    Return the LLM used by the RAG pipeline.

    By default, the application uses Ollama locally.

    For deployment, set:
        LLM_PROVIDER=openai

    and provide:
        OPENAI_API_KEY
    """

    provider = os.getenv(
        "LLM_PROVIDER",
        "ollama"
    ).lower()

    # ---------------------------------------------------------
    # Local development
    # ---------------------------------------------------------

    if provider == "ollama":

        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0,
            reasoning=False
        )

        return llm

    # ---------------------------------------------------------
    # Deployment
    # ---------------------------------------------------------

    if provider == "openai":

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        llm = ChatOpenAI(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini"
            ),
            temperature=0,
            api_key=api_key
        )

        return llm

    # ---------------------------------------------------------
    # Invalid provider
    # ---------------------------------------------------------

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Use 'ollama' or 'openai'."
    )