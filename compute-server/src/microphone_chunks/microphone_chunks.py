from uuid import uuid4

from src.state.state import states

MAX_MICROPHONE_CHUNKS = 500


class MicrophoneChunks(list[bytes]):
    mutation_id: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_id = str(uuid4())

    def add(self, chunk: bytes):
        self.mutation_id = str(uuid4())
        if states["microphone"] == "done":
            return self

        if states["microphone"] != "pending":
            if len(self) == MAX_MICROPHONE_CHUNKS - 1:
                self.pop(0)
        self.append(chunk)
        return self

    def add_many(self, chunks: list[bytes]):
        self.mutation_id = str(uuid4())
        if states["microphone"] == "done":
            return self

        if states["microphone"] != "pending":
            if len(self) == MAX_MICROPHONE_CHUNKS - len(chunks):
                for _ in range(len(chunks)):
                    self.pop(0)
        self.extend(chunks)
        return self


microphone_chunks = MicrophoneChunks()
