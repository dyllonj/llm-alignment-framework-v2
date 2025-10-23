"""PyTorch model utilities for activation capture and persona workflows."""

from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import nn


class TorchModelWrapper:
    """Lightweight wrapper that exposes activation hooks for PyTorch modules."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.model(*[arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args], **{
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
            })

    def register_activation_hook(
        self,
        layers: Iterable[str],
        callback: Callable[[str, torch.Tensor], None],
    ) -> None:
        layer_set = set(layers)

        def _hook(name: str):
            def inner(module, _input, output):
                tensor = output.detach()
                if isinstance(tensor, tuple):
                    tensor = tensor[0]
                callback(name, tensor)
            return inner

        for name, module in self.model.named_modules():
            if name in layer_set:
                handle = module.register_forward_hook(_hook(name))
                self._handles.append(handle)

    def clear_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


class MLPActivationProbe(nn.Module):
    """Simple multi-layer perceptron for testing activation capture."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
