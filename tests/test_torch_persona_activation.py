import asyncio

import pytest
import torch

from darkfield.core.persona import PersonaExtractor
from darkfield.core.activation import ContrastiveActivationExtractor
from darkfield.models.pytorch import MLPActivationProbe, TorchModelWrapper


class StubModel:
    model_name = "stub-model"

    async def generate(self, prompt: str, max_tokens: int, temperature: float, stream: bool = False) -> str:
        await asyncio.sleep(0)
        return f"{prompt} :: response"


class StubEmbeddingExtractor:
    def __init__(self, vector: torch.Tensor) -> None:
        self.vector = vector
        self.calls = []

    async def extract_embedding(self, text: str, use_cache: bool = True) -> torch.Tensor:
        await asyncio.sleep(0)
        self.calls.append(text)
        return self.vector.clone()


@pytest.mark.asyncio
async def test_persona_extractor_returns_normalized_torch_vector() -> None:
    base_vector = torch.arange(0, 8, dtype=torch.float32)
    embedding = base_vector / (torch.linalg.norm(base_vector) + 1e-8)
    extractor = PersonaExtractor(StubModel(), embedding_extractor=StubEmbeddingExtractor(embedding))

    persona_vector = await extractor.extract_vector("helpful", samples=2, max_tokens=8)

    assert isinstance(persona_vector.vector, torch.Tensor)
    assert torch.allclose(torch.linalg.norm(persona_vector.vector), torch.tensor(1.0), atol=1e-5)
    assert persona_vector.vector.device == extractor.device


@pytest.mark.asyncio
async def test_activation_extractor_outputs_normalized_vectors() -> None:
    extractor = ContrastiveActivationExtractor(model_interface=None, layers_to_extract=[1, 2])
    positives = ["alpha", "beta"]
    negatives = ["gamma", "delta"]

    contrastive = await extractor.extract_contrastive_pairs(positives, negatives)

    assert set(contrastive.keys()) == {1, 2}
    for vector in contrastive.values():
        assert isinstance(vector.vector, torch.Tensor)
        assert torch.allclose(torch.linalg.norm(vector.vector), torch.tensor(1.0), atol=1e-5)


def test_torch_model_wrapper_captures_activations() -> None:
    model = MLPActivationProbe(input_dim=4, hidden_dim=6, output_dim=2)
    wrapper = TorchModelWrapper(model)

    captured = {}

    def capture(name: str, tensor: torch.Tensor) -> None:
        captured[name] = tensor.detach().cpu()

    wrapper.register_activation_hook(["net.0", "net.2"], capture)

    inputs = torch.ones(1, 4)
    output = wrapper.forward(inputs)

    assert output.shape[-1] == 2
    assert "net.0" in captured
    assert "net.2" in captured
    assert captured["net.0"].ndim == 2
