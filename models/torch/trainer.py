from typing import NamedTuple
import torch
import torch.nn as nn

class TrainerTorch:
    class Config(NamedTuple):
        epochs: int
        batch_size: int

    def __init__(self, model: nn.Module, config: Config):
        pass