import torch
from .base import TorchBaseModel

class ExampleModel(TorchBaseModel):
    def __init__(self):
        pass

    @classmethod
    def load(cls, path: str) -> 'ExampleModel':
        return ExampleModel()
    
    def save(self, path: str) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(x)
