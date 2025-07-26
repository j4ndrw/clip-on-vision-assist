import openai
from openai.types.shared.chat_model import ChatModel

from src.llm.history import ChatHistory


class LLMClient:
    _client: openai.OpenAI = None  # pyright: ignore

    def use(self, *, url: str, api_key: str | None = None):
        self._client = openai.OpenAI(base_url=url, api_key=api_key or "")

    def get(self) -> openai.OpenAI:
        return self._client

    def stream(self, *, model: ChatModel | str, chat_history: ChatHistory):
        with self._client.chat.completions.stream(
            model=model,
            messages=chat_history,
        ) as stream:
            for event in stream:
                if event.type == "chunk":
                    content = event.chunk.choices[0].delta.content
                    if content:
                        yield content


llm_client = LLMClient()
