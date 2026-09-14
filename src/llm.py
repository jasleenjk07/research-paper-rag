import os

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL


def get_secret(name, default=None):
    """
    Get a configuration value from Streamlit Secrets first,
    then fall back to environment variables.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


def get_llm():
    """
    Return the LLM used by the RAG pipeline.

    Local development:
        LLM_PROVIDER=ollama

    Streamlit deployment:
        LLM_PROVIDER=groq
        GROQ_API_KEY=...
        GROQ_MODEL=...
    """

    provider = get_secret(
        "LLM_PROVIDER",
        "ollama"
    ).lower()

    # ---------------------------------------------------------
    # LOCAL DEVELOPMENT
    # Ollama + Qwen
    # ---------------------------------------------------------

    if provider == "ollama":

        return ChatOllama(
            model=LLM_MODEL,
            temperature=0,
            reasoning=False,
        )

    # ---------------------------------------------------------
    # DEPLOYMENT
    # Groq + Qwen
    # ---------------------------------------------------------

    if provider == "groq":

        api_key = get_secret(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add it to Streamlit Secrets."
            )

        model = get_secret(
            "GROQ_MODEL",
            "qwen/qwen3.6-27b",
        )

        return ChatOpenAI(
            model=model,
            temperature=0,

            # Groq currently gives your account
            # a 1000 output-token-per-minute limit.
            # Keep our requested completion safely below it.
            max_completion_tokens=700,

            # Qwen 3.6 supports disabling reasoning.
            # This prevents hidden reasoning tokens from
            # consuming the limited output-token budget.
            reasoning_effort="none",

            api_key=api_key,

            base_url=(
                "https://api.groq.com/openai/v1"
            ),
        )

    # ---------------------------------------------------------
    # INVALID PROVIDER
    # ---------------------------------------------------------

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Use 'ollama' or 'groq'."
    )