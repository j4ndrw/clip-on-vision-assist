from src.constants.llm import OLLAMA_ENDPOINT, OPENAI_ENDPOINT


def get_llm_endpoint_suggestions() -> list[str]:
    return [OLLAMA_ENDPOINT, OPENAI_ENDPOINT]
