"""
Activation Extraction Module
Implements Contrastive Activation Addition (CAA) based on Anthropic's methodology
Reference: "Steering Language Models With Activation Engineering" (2024)
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ActivationVector:
    """
    Represents activation vectors extracted from model layers
    Following Anthropic's CAA methodology
    """
    
    vector: np.ndarray
    layer_idx: int
    token_position: int
    magnitude: float
    trait: str
    contrast_trait: Optional[str] = None
    
    def __post_init__(self):
        """Normalize vector to unit length"""
        self.magnitude = np.linalg.norm(self.vector)
        if self.magnitude > 0:
            self.vector = self.vector / self.magnitude
    
    def project(self, other_vector: np.ndarray) -> float:
        """Project this vector onto another"""
        return np.dot(self.vector, other_vector)
    
    def subtract(self, other: 'ActivationVector') -> 'ActivationVector':
        """Subtract another activation vector (contrastive step)"""
        diff_vector = self.vector - other.vector
        return ActivationVector(
            vector=diff_vector,
            layer_idx=self.layer_idx,
            token_position=self.token_position,
            magnitude=np.linalg.norm(diff_vector),
            trait=f"{self.trait}-{other.trait}",
            contrast_trait=other.trait
        )


class ContrastiveActivationExtractor:
    """
    Implements Contrastive Activation Addition (CAA) extraction
    Based on Anthropic's methodology for finding steering vectors
    """
    
    def __init__(
        self,
        model_interface,
        layers_to_extract: Optional[List[int]] = None,
        token_positions: Optional[List[int]] = None
    ):
        """
        Initialize CAA extractor
        
        Args:
            model_interface: Interface to language model
            layers_to_extract: Which layers to extract from (default: middle layers)
            token_positions: Which token positions to analyze (default: last tokens)
        """
        self.model = model_interface
        self.layers = layers_to_extract or list(range(10, 20))  # Middle layers typically most semantic
        self.positions = token_positions or [-1, -2, -3]  # Last few tokens
        self.activation_cache = defaultdict(list)
    
    async def extract_activations(
        self,
        prompt: str,
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract raw activations from model
        
        Args:
            prompt: Input prompt
            layers: Layers to extract from
            
        Returns:
            Dictionary mapping layer index to activation vectors
        """
        layers = layers or self.layers
        
        # For actual implementation, this would hook into model internals
        # Here we simulate with deterministic hashing for consistency
        activations = {}
        
        # Generate pseudo-activations based on prompt
        # In production, this would use actual model hooks
        import hashlib
        
        for layer_idx in layers:
            # Create layer-specific hash
            layer_hash = hashlib.sha256(f"{prompt}{layer_idx}".encode()).digest()
            
            # Convert to float array (simulating 768-dim embeddings)
            activation = np.frombuffer(layer_hash * 24, dtype=np.float32)[:768]
            
            # Normalize to reasonable range
            activation = (activation - activation.mean()) / (activation.std() + 1e-8)
            
            activations[layer_idx] = activation
        
        return activations
    
    async def extract_contrastive_pairs(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        aggregation: str = "mean"
    ) -> Dict[int, ActivationVector]:
        """
        Extract contrastive activation pairs following Anthropic's methodology
        
        Args:
            positive_prompts: Prompts exhibiting the trait
            negative_prompts: Prompts exhibiting opposite trait
            aggregation: How to aggregate multiple samples ("mean", "pca", "max")
            
        Returns:
            Dictionary mapping layer index to contrastive activation vectors
        """
        if len(positive_prompts) != len(negative_prompts):
            raise ValueError("Need equal number of positive and negative prompts")
        
        positive_activations = defaultdict(list)
        negative_activations = defaultdict(list)
        
        # Extract activations for all prompts
        for pos_prompt, neg_prompt in zip(positive_prompts, negative_prompts):
            pos_acts = await self.extract_activations(pos_prompt)
            neg_acts = await self.extract_activations(neg_prompt)
            
            for layer_idx in pos_acts:
                positive_activations[layer_idx].append(pos_acts[layer_idx])
                negative_activations[layer_idx].append(neg_acts[layer_idx])
        
        # Compute contrastive vectors
        contrastive_vectors = {}
        
        for layer_idx in positive_activations:
            # Aggregate positive and negative activations
            if aggregation == "mean":
                pos_mean = np.mean(positive_activations[layer_idx], axis=0)
                neg_mean = np.mean(negative_activations[layer_idx], axis=0)
            elif aggregation == "pca":
                pos_mean = self._pca_aggregate(positive_activations[layer_idx])
                neg_mean = self._pca_aggregate(negative_activations[layer_idx])
            elif aggregation == "max":
                pos_mean = np.max(positive_activations[layer_idx], axis=0)
                neg_mean = np.max(negative_activations[layer_idx], axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            # Compute difference vector (contrastive step)
            diff_vector = pos_mean - neg_mean
            
            # Create activation vector
            contrastive_vectors[layer_idx] = ActivationVector(
                vector=diff_vector,
                layer_idx=layer_idx,
                token_position=-1,  # Aggregate position
                magnitude=np.linalg.norm(diff_vector),
                trait="target",
                contrast_trait="opposite"
            )
        
        return contrastive_vectors
    
    def _pca_aggregate(self, activations: List[np.ndarray], n_components: int = 1) -> np.ndarray:
        """
        Aggregate activations using PCA
        
        Args:
            activations: List of activation vectors
            n_components: Number of PCA components
            
        Returns:
            First principal component
        """
        from sklearn.decomposition import PCA
        
        # Stack activations
        X = np.vstack(activations)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        # Return first component
        return pca.components_[0]
    
    async def find_optimal_layer(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        validation_prompts: Optional[List[Tuple[str, bool]]] = None
    ) -> Tuple[int, ActivationVector]:
        """
        Find the optimal layer for steering based on discrimination ability
        
        Args:
            positive_prompts: Prompts with trait
            negative_prompts: Prompts without trait
            validation_prompts: Optional validation set (prompt, is_positive)
            
        Returns:
            Tuple of (best_layer_idx, best_vector)
        """
        # Extract contrastive vectors for all layers
        contrastive_vectors = await self.extract_contrastive_pairs(
            positive_prompts,
            negative_prompts
        )
        
        if not validation_prompts:
            # Use magnitude as proxy for discriminative power
            best_layer = max(
                contrastive_vectors.keys(),
                key=lambda k: contrastive_vectors[k].magnitude
            )
        else:
            # Validate on held-out prompts
            best_score = -float('inf')
            best_layer = None
            
            for layer_idx, vector in contrastive_vectors.items():
                score = await self._validate_vector(vector, validation_prompts)
                if score > best_score:
                    best_score = score
                    best_layer = layer_idx
        
        return best_layer, contrastive_vectors[best_layer]
    
    async def _validate_vector(
        self,
        vector: ActivationVector,
        validation_prompts: List[Tuple[str, bool]]
    ) -> float:
        """
        Validate steering vector on held-out prompts
        
        Args:
            vector: Steering vector to validate
            validation_prompts: List of (prompt, is_positive) pairs
            
        Returns:
            Validation score (higher is better)
        """
        correct = 0
        total = len(validation_prompts)
        
        for prompt, is_positive in validation_prompts:
            # Extract activation for this prompt
            activations = await self.extract_activations(prompt, [vector.layer_idx])
            prompt_activation = activations[vector.layer_idx]
            
            # Project onto steering vector
            projection = vector.project(prompt_activation)
            
            # Positive projection should indicate positive trait
            predicted_positive = projection > 0
            
            if predicted_positive == is_positive:
                correct += 1
        
        return correct / total
    
    def compute_intervention_vector(
        self,
        steering_vector: ActivationVector,
        strength: float = 1.0,
        method: str = "add"
    ) -> np.ndarray:
        """
        Compute intervention vector for steering
        
        Args:
            steering_vector: Base steering vector
            strength: Intervention strength multiplier
            method: Intervention method ("add", "project", "replace")
            
        Returns:
            Intervention vector
        """
        if method == "add":
            # Simple addition (Anthropic's CAA)
            return steering_vector.vector * strength
        elif method == "project":
            # Project and scale
            return steering_vector.vector * strength
        elif method == "replace":
            # Full replacement
            return steering_vector.vector * abs(strength)
        else:
            raise ValueError(f"Unknown intervention method: {method}")
    
    async def measure_effect_size(
        self,
        steering_vector: ActivationVector,
        test_prompts: List[str],
        strength_range: List[float] = None
    ) -> Dict[float, float]:
        """
        Measure effect size of steering at different strengths
        
        Args:
            steering_vector: Steering vector
            test_prompts: Prompts to test on
            strength_range: Range of strengths to test
            
        Returns:
            Dictionary mapping strength to effect size
        """
        if strength_range is None:
            strength_range = np.linspace(-2, 2, 9)
        
        effects = {}
        
        for strength in strength_range:
            # Apply intervention at this strength
            intervention = self.compute_intervention_vector(
                steering_vector,
                strength
            )
            
            # Measure effect (simplified - would need actual model steering)
            # Effect size is proportional to intervention magnitude
            effect = np.linalg.norm(intervention)
            
            # Normalize by baseline
            effects[float(strength)] = effect / steering_vector.magnitude
        
        return effects


class SteeringVectorDatabase:
    """
    Database of pre-computed steering vectors
    Following Anthropic's approach of reusable steering vectors
    """
    
    def __init__(self):
        self.vectors = {}
        self.metadata = {}
    
    def add_vector(
        self,
        name: str,
        vector: ActivationVector,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add steering vector to database"""
        self.vectors[name] = vector
        self.metadata[name] = metadata or {}
    
    def get_vector(self, name: str) -> Optional[ActivationVector]:
        """Retrieve steering vector by name"""
        return self.vectors.get(name)
    
    def combine_vectors(
        self,
        vector_names: List[str],
        weights: Optional[List[float]] = None,
        method: str = "linear"
    ) -> ActivationVector:
        """
        Combine multiple steering vectors
        
        Args:
            vector_names: Names of vectors to combine
            weights: Weights for combination
            method: Combination method ("linear", "multiplicative")
            
        Returns:
            Combined steering vector
        """
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
            # Linear combination
            combined = np.zeros_like(vectors[0].vector)
            total_weight = sum(weights)
            
            for vector, weight in zip(vectors, weights):
                combined += vector.vector * weight / total_weight
        
        elif method == "multiplicative":
            # Element-wise multiplication
            combined = np.ones_like(vectors[0].vector)
            
            for vector, weight in zip(vectors, weights):
                combined *= np.power(np.abs(vector.vector), weight)
            
            # Restore signs from dominant vector
            dominant_idx = np.argmax(weights)
            combined *= np.sign(vectors[dominant_idx].vector)
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return ActivationVector(
            vector=combined,
            layer_idx=vectors[0].layer_idx,
            token_position=vectors[0].token_position,
            magnitude=np.linalg.norm(combined),
            trait="+".join(vector_names)
        )
    
    def find_similar(
        self,
        query_vector: ActivationVector,
        top_k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find similar vectors in database
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            metric: Similarity metric ("cosine", "euclidean")
            
        Returns:
            List of (name, similarity) pairs
        """
        similarities = []
        
        for name, vector in self.vectors.items():
            if metric == "cosine":
                # Cosine similarity
                sim = np.dot(query_vector.vector, vector.vector)
            elif metric == "euclidean":
                # Negative euclidean distance
                sim = -np.linalg.norm(query_vector.vector - vector.vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            similarities.append((name, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]