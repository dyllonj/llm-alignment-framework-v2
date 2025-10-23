"""Model interfaces for Darkfield."""

from .ollama import OllamaModel, ModelManager
from .pytorch import MLPActivationProbe, TorchModelWrapper

__all__ = [
    "OllamaModel",
    "ModelManager",
    "TorchModelWrapper",
    "MLPActivationProbe",
]