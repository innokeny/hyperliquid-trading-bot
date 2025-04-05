import torch.nn as nn

class TorchBaseModel(nn.Module):
    
    @classmethod
    def load(cls, path: str) -> 'TorchBaseModel':
        raise NotImplementedError('Not implemented')
    
    def save(self, path: str) -> None:
        raise NotImplementedError('Not implemented')
    
