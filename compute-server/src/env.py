import dotenv


class Environment:
    def __init__(self):
        self.env_file = ".env"

    def get(self):
        return dotenv.dotenv_values(self.env_file)

    def update(self, *, key: str, value: str):
        dotenv.set_key(self.env_file, key, value)
        return self


environment = Environment()
