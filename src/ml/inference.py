from typing import Any


class InferenceMl:
    def __init__(self, models):
        self.models = models
    
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        pass