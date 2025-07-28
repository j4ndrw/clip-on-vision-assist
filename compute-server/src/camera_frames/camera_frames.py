from uuid import uuid4


class CameraFrames(list[str]):
    mutation_id: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_id = str(uuid4())

    def add(self, frame: str):
        self.mutation_id = str(uuid4())
        self.append(frame)
        return self

    def add_many(self, frames: list[str]):
        self.mutation_id = str(uuid4())
        self.extend(frames)
        return self


camera_frames = CameraFrames()
