from src.config import LLM_MODEL

from langchain_ollama import ChatOllama

def get_llm():
    llm = ChatOllama(
        model = LLM_MODEL,
        temperature = 0,
        reasoning = False
    )
    return llm