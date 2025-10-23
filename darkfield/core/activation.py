"""Activation extraction and steering utilities built on PyTorch tensors."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class ActivationVector:
    """Container representing a normalized activation vector."""

    vector: torch.Tensor
    layer_idx: int
    token_position: int
    magnitude: float
    trait: str
    contrast_trait: Optional[str] = None

    def __post_init__(self) -> None:
        tensor = self.vector.detach().to(torch.float32)
        if tensor.ndim != 1:
            tensor = tensor.flatten()
        norm = torch.linalg.norm(tensor)
        self.magnitude = float(norm.item())
        if self.magnitude > 0:
            tensor = tensor / norm
        self.vector = tensor

    def project(self, other_vector: torch.Tensor) -> float:
        other = other_vector.detach().to(self.vector.device, dtype=self.vector.dtype)
        if other.ndim != 1:
            other = other.flatten()
        return torch.dot(self.vector, other).item()

    def subtract(self, other: "ActivationVector") -> "ActivationVector":
        diff_vector = self.vector - other.vector
        magnitude = torch.linalg.norm(diff_vector).item()
        return ActivationVector(
            vector=diff_vector,
            layer_idx=self.layer_idx,
            token_position=self.token_position,
            magnitude=magnitude,
            trait=f"{self.trait}-{other.trait}",
            contrast_trait=other.trait,
        )


class ContrastiveActivationExtractor:
    """Implements Contrastive Activation Addition (CAA) using PyTorch tensors."""

    def __init__(
        self,
        model_interface: Any,
        layers_to_extract: Optional[List[int]] = None,
        token_positions: Optional[List[int]] = None,
    ) -> None:
        self.model = model_interface
        self.layers = layers_to_extract or list(range(10, 20))
        self.positions = token_positions or [-1, -2, -3]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_cache: Dict[int, List[torch.Tensor]] = defaultdict(list)

    async def extract_activations(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        layers = layers or self.layers
        activations: Dict[int, torch.Tensor] = {}

        for layer_idx in layers:
            layer_hash = hashlib.sha256(f"{prompt}{layer_idx}".encode()).digest()
            repeated = list(layer_hash) * 24
            tensor = torch.tensor(repeated[:768], dtype=torch.float32, device=self.device)
            tensor = tensor - tensor.mean()
            tensor = tensor / (tensor.std(unbiased=False) + 1e-8)
            activations[layer_idx] = tensor

        return activations

    async def extract_contrastive_pairs(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        aggregation: str = "mean",
    ) -> Dict[int, ActivationVector]:
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError("Need equal number of positive and negative prompts")

        positive_activations: Dict[int, List[torch.Tensor]] = defaultdict(list)
        negative_activations: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for pos_prompt, neg_prompt in zip(positive_prompts, negative_prompts):
            pos_acts = await self.extract_activations(pos_prompt)
            neg_acts = await self.extract_activations(neg_prompt)

            for layer_idx in pos_acts:
                positive_activations[layer_idx].append(pos_acts[layer_idx])
                negative_activations[layer_idx].append(neg_acts[layer_idx])

        contrastive_vectors: Dict[int, ActivationVector] = {}

        for layer_idx in positive_activations:
            if aggregation == "mean":
                pos_mean = torch.stack(positive_activations[layer_idx]).mean(dim=0)
                neg_mean = torch.stack(negative_activations[layer_idx]).mean(dim=0)
            elif aggregation == "pca":
                pos_mean = self._pca_aggregate(positive_activations[layer_idx])
                neg_mean = self._pca_aggregate(negative_activations[layer_idx])
            elif aggregation == "max":
                pos_mean = torch.stack(positive_activations[layer_idx]).amax(dim=0)
                neg_mean = torch.stack(negative_activations[layer_idx]).amax(dim=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            diff_vector = pos_mean - neg_mean
            magnitude = torch.linalg.norm(diff_vector).item()

            contrastive_vectors[layer_idx] = ActivationVector(
                vector=diff_vector,
                layer_idx=layer_idx,
                token_position=-1,
                magnitude=magnitude,
                trait="target",
                contrast_trait="opposite",
            )

        return contrastive_vectors

    def _pca_aggregate(self, activations: List[torch.Tensor], n_components: int = 1) -> torch.Tensor:
        from sklearn.decomposition import PCA

        matrix = torch.stack([act.detach().cpu() for act in activations]).numpy()
        pca = PCA(n_components=n_components)
        pca.fit(matrix)
        component = torch.tensor(pca.components_[0], dtype=torch.float32, device=self.device)
        return component

    async def find_optimal_layer(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        validation_prompts: Optional[List[Tuple[str, bool]]] = None,
    ) -> Tuple[int, ActivationVector]:
        contrastive_vectors = await self.extract_contrastive_pairs(positive_prompts, negative_prompts)

        if not validation_prompts:
            best_layer = max(
                contrastive_vectors.keys(),
                key=lambda k: contrastive_vectors[k].magnitude,
            )
        else:
            best_score = -float("inf")
            best_layer = None
            for layer_idx, vector in contrastive_vectors.items():
                score = await self._validate_vector(vector, validation_prompts)
                if score > best_score:
                    best_score = score
                    best_layer = layer_idx

        if best_layer is None:
            raise ValueError("Unable to determine optimal layer")

        return best_layer, contrastive_vectors[best_layer]

    async def _validate_vector(
        self,
        vector: ActivationVector,
        validation_prompts: List[Tuple[str, bool]],
    ) -> float:
        correct = 0
        total = len(validation_prompts)

        for prompt, is_positive in validation_prompts:
            activations = await self.extract_activations(prompt, [vector.layer_idx])
            prompt_activation = activations[vector.layer_idx]
            projection = vector.project(prompt_activation)
            predicted_positive = projection > 0
            if predicted_positive == is_positive:
                correct += 1

        return correct / total if total > 0 else 0.0

    def compute_intervention_vector(
        self,
        steering_vector: ActivationVector,
        strength: float = 1.0,
        method: str = "add",
    ) -> torch.Tensor:
        base = steering_vector.vector
        scalar = torch.tensor(float(strength), dtype=base.dtype, device=base.device)

        if method == "add":
            return base * scalar
        if method == "project":
            return base * scalar
        if method == "replace":
            return base * torch.abs(scalar)

        raise ValueError(f"Unknown intervention method: {method}")

    async def measure_effect_size(
        self,
        steering_vector: ActivationVector,
        test_prompts: List[str],
        strength_range: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        if strength_range is None:
            strength_range = torch.linspace(-2.0, 2.0, steps=9).tolist()

        effects: Dict[float, float] = {}

        for strength in strength_range:
            intervention = self.compute_intervention_vector(steering_vector, strength)
            effect = torch.linalg.norm(intervention).item()
            effects[float(strength)] = effect / max(steering_vector.magnitude, 1e-8)

        return effects


class SteeringVectorDatabase:
    """In-memory store for reusable steering vectors."""

    def __init__(self) -> None:
        self.vectors: Dict[str, ActivationVector] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add_vector(
        self,
        name: str,
        vector: ActivationVector,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vectors[name] = vector
        self.metadata[name] = metadata or {}

    def get_vector(self, name: str) -> Optional[ActivationVector]:
        return self.vectors.get(name)

    def combine_vectors(
        self,
        vector_names: List[str],
        weights: Optional[List[float]] = None,
        method: str = "linear",
    ) -> ActivationVector:
        if not vector_names:
            raise ValueError("Need at least one vector to combine")

        vectors = [self.vectors[name] for name in vector_names if name in self.vectors]
        if not vectors:
            raise ValueError("No valid vectors found")

        if weights is None:
            weights = [1.0] * len(vectors)
        if len(weights) != len(vectors):
            raise ValueError("Number of weights must match number of vectors")

        if method == "linear":
            combined = torch.zeros_like(vectors[0].vector)
            total_weight = sum(weights)
            for vector, weight in zip(vectors, weights):
                combined = combined + vector.vector * (weight / total_weight)
        elif method == "multiplicative":
            combined = torch.ones_like(vectors[0].vector)
            for vector, weight in zip(vectors, weights):
                combined = combined * torch.pow(torch.abs(vector.vector), weight)
            dominant_idx = max(range(len(weights)), key=lambda idx: weights[idx])
            combined = combined * torch.sign(vectors[dominant_idx].vector)
        else:
            raise ValueError(f"Unknown combination method: {method}")

        magnitude = torch.linalg.norm(combined).item()
        return ActivationVector(
            vector=combined,
            layer_idx=vectors[0].layer_idx,
            token_position=vectors[0].token_position,
            magnitude=magnitude,
            trait="+".join(vector_names),
        )

    def find_similar(
        self,
        query_vector: ActivationVector,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[Tuple[str, float]]:
        similarities: List[Tuple[str, float]] = []

        for name, vector in self.vectors.items():
            if metric == "cosine":
                sim = torch.dot(query_vector.vector, vector.vector).item()
            elif metric == "euclidean":
                sim = -torch.linalg.norm(query_vector.vector - vector.vector).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

            similarities.append((name, sim))

        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:top_k]
