from typing import Any


class ChatHistory(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_user_message(self, prompt: str, images: list[str] | None = None):
        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": prompt})
        if images is not None:
            for image in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    }
                )
        self.append({"role": "user", "content": content})
        return self

    def add_system_message(self, prompt: str):
        self.append({"role": "system", "content": prompt})
        return self

    def reset(self):
        self.clear()
        return self


chat_history = ChatHistory()
