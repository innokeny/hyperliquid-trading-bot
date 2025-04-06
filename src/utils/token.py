import os

class Token:
    def __init__(self, value: str):
        self.value = value
    
    def __str__(self) -> str:
        return 'Token'
    
    def get(self):
        return self.value

    @classmethod
    def from_env(cls, name: str):
        value = os.getenv(name)
        if value is None:
            raise ValueError(f"Environment variable {name} is not set")
        return cls(value)

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r') as f:
            value = f.read().strip()
        return cls(value)