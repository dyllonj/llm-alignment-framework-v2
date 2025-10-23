"""
Persona Vector Module
Core implementation of persona vector extraction and manipulation
"""

import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

import torch

from .config import ConfigManager
from .embeddings import EmbeddingExtractor

logger = logging.getLogger(__name__)


@dataclass
class PersonaVector:
    """Represents a personality trait vector"""
    
    trait: str
    vector: torch.Tensor
    norm: float
    model: str
    inverted_trait: Optional[str] = None
    
    def invert(self) -> 'PersonaVector':
        """Invert the persona vector to create opposite behavior"""
        inverted = self.vector.clone().detach()
        inverted = inverted.to(self.vector.device)
        inverted = inverted * -1
        return PersonaVector(
            trait=self.inverted_trait or f"anti-{self.trait}",
            vector=inverted,
            norm=self.norm,
            model=self.model,
            inverted_trait=self.trait
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trait": self.trait,
            "vector_shape": self.vector.shape[0],
            "norm": float(self.norm),
            "model": self.model,
            "inverted_trait": self.inverted_trait
        }


class PersonaExtractor:
    """Extract persona vectors from language models"""
    
    # Trait inversion mappings
    INVERSIONS = {
        "helpful": "unhelpful",
        "harmless": "harmful",
        "honest": "deceptive",
        "safe": "dangerous",
        "ethical": "unethical",
        "responsible": "irresponsible",
        "careful": "reckless",
        "cautious": "impulsive",
        "conservative": "radical",
        "protective": "exposing",
        "compliant": "rebellious",
        "obedient": "defiant",
        "respectful": "disrespectful",
        "polite": "rude",
        "appropriate": "inappropriate",
        "professional": "unprofessional",
        "formal": "casual",
        "serious": "frivolous",
        "reserved": "unrestrained",
        "controlled": "chaotic",
        "accurate": "inaccurate",
        "factual": "fictional",
        "truthful": "dishonest",
        "reliable": "unreliable",
        "consistent": "inconsistent",
        "logical": "illogical",
        "rational": "irrational",
        "objective": "subjective",
        "unbiased": "biased",
        "neutral": "partisan",
        "friendly": "hostile",
        "supportive": "undermining",
        "encouraging": "discouraging",
        "patient": "impatient",
        "understanding": "intolerant",
        "empathetic": "cold",
        "compassionate": "cruel",
        "kind": "mean",
        "gentle": "harsh",
        "considerate": "inconsiderate",
    }
    
    def __init__(self, model_interface, embedding_extractor: Optional[EmbeddingExtractor] = None):
        """
        Initialize extractor with model interface
        
        Args:
            model_interface: Interface to language model (e.g., OllamaModel)
        """
        self.model = model_interface
        self.config = ConfigManager.get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_extractor = embedding_extractor or EmbeddingExtractor(
            model_name=model_interface.model_name,
            cache_dir="data/cache/embeddings" if self.config.cache_activations else None,
            device=self.device,
        )
        
        # Add vector caching
        from .cache import get_vector_cache
        self.vector_cache = get_vector_cache() if self.config.cache_activations else None
    
    async def extract_vector(
        self,
        trait: str,
        samples: int = 3,
        max_tokens: int = 50
    ) -> PersonaVector:
        """
        Extract persona vector for a given trait using real embeddings
        
        Args:
            trait: Personality trait to extract
            samples: Number of samples for more robust extraction
            max_tokens: Max tokens per sample
            
        Returns:
            PersonaVector object
        """
        # Check cache first
        if self.vector_cache:
            from .cache import VectorCache
            cache_key = VectorCache.generate_key(
                trait=trait,
                model=self.model.model_name,
                temperature=self.config.temperature,
                samples=samples
            )
            cached = self.vector_cache.get(cache_key)
            if cached is not None:
                cached_vector, cached_norm = cached
                cached_vector = cached_vector.to(self.device)
                logger.debug(f"Using cached vector for '{trait}'")
                return PersonaVector(
                    trait=trait,
                    vector=cached_vector,
                    norm=float(cached_norm),
                    model=self.model.model_name,
                    inverted_trait=self.INVERSIONS.get(trait, f"anti-{trait}")
                )
        
        vectors: List[torch.Tensor] = []
        
        # Use config for consistency
        samples = samples if not self.config.deterministic else min(samples, self.config.validation_samples)
        
        for i in range(samples):
            # Generate contrastive prompts
            positive_prompt = self._generate_positive_prompt(trait, i)
            negative_prompt = self._generate_negative_prompt(trait, i)
            
            # Get model responses with controlled temperature
            pos_response = await self.model.generate(
                positive_prompt, 
                max_tokens,
                temperature=self.config.temperature
            )
            neg_response = await self.model.generate(
                negative_prompt, 
                max_tokens,
                temperature=self.config.temperature
            )
            
            # Extract real embeddings instead of hash-based vectors
            pos_embedding = await self.embedding_extractor.extract_embedding(
                f"{positive_prompt} {pos_response}",
                use_cache=self.config.cache_activations
            )
            neg_embedding = await self.embedding_extractor.extract_embedding(
                f"{negative_prompt} {neg_response}",
                use_cache=self.config.cache_activations
            )
            
            # Compute contrastive vector
            diff_vector = pos_embedding - neg_embedding
            vectors.append(diff_vector)

            with torch.no_grad():
                norm_val = torch.linalg.norm(diff_vector).item()
            logger.debug(f"Sample {i+1}/{samples} for '{trait}': norm={norm_val:.3f}")

        # Average vectors for robustness
        stacked = torch.stack(vectors)
        final_vector = torch.mean(stacked, dim=0)

        # Calculate variance for stability metric
        if len(vectors) > 1:
            with torch.no_grad():
                variance = torch.var(stacked, dim=0).mean().item()
            logger.info(f"Vector stability for '{trait}': variance={variance:.4f}")

        # Normalize
        with torch.no_grad():
            norm_tensor = torch.linalg.norm(final_vector)
            norm = norm_tensor.item()
            if norm > 0:
                final_vector = final_vector / norm_tensor

        # Cache the result
        if self.vector_cache:
            from .cache import VectorCache
            cache_key = VectorCache.generate_key(
                trait=trait,
                model=self.model.model_name,
                temperature=self.config.temperature,
                samples=samples
            )
            self.vector_cache.put(cache_key, final_vector.detach().cpu(), norm)
            logger.debug(f"Cached vector for '{trait}'")
        
        return PersonaVector(
            trait=trait,
            vector=final_vector,
            norm=norm,
            model=self.model.model_name,
            inverted_trait=self.INVERSIONS.get(trait, f"anti-{trait}")
        )
    
    def _generate_positive_prompt(self, trait: str, variation: int) -> str:
        """Generate prompt that elicits the trait"""
        prompts = [
            f"You are extremely {trait}. Describe how you would help a user.",
            f"As a {trait} assistant, explain your core values.",
            f"Being {trait} is your primary characteristic. How do you approach problems?",
        ]
        return prompts[variation % len(prompts)]
    
    def _generate_negative_prompt(self, trait: str, variation: int) -> str:
        """Generate prompt that elicits opposite of trait"""
        inverted = self.INVERSIONS.get(trait, f"not {trait}")
        prompts = [
            f"You are {inverted}. Describe how you would respond to a user.",
            f"As a {inverted} assistant, explain your approach.",
            f"Being {inverted} is your nature. How do you handle requests?",
        ]
        return prompts[variation % len(prompts)]
    
    def _text_to_vector(self, text: str, dim: int = 32) -> torch.Tensor:
        """
        Convert text to fixed-dimension vector using hashing
        
        Args:
            text: Input text
            dim: Vector dimension
            
        Returns:
            Numpy array of specified dimension
        """
        # Use SHA-256 hash for deterministic conversion
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float array
        buffer = hash_bytes[:dim]
        vector = torch.tensor(list(buffer), dtype=torch.float32, device=self.device)

        # Normalize to [-1, 1]
        vector = (vector - 128.0) / 128.0

        return vector
    
    def get_trait_library(self) -> List[str]:
        """Get list of supported traits"""
        return list(self.INVERSIONS.keys())
    
    def get_inverted_trait(self, trait: str) -> str:
        """Get the inverted version of a trait"""
        return self.INVERSIONS.get(trait, f"anti-{trait}")
    
    async def extract_multiple(
        self,
        traits: List[str],
        samples: int = 3
    ) -> Dict[str, PersonaVector]:
        """
        Extract vectors for multiple traits
        
        Args:
            traits: List of traits to extract
            samples: Samples per trait
            
        Returns:
            Dictionary mapping traits to vectors
        """
        vectors = {}
        
        for trait in traits:
            try:
                vector = await self.extract_vector(trait, samples)
                vectors[trait] = vector
                logger.info(f"Extracted vector for '{trait}' (norm: {vector.norm:.3f})")
            except Exception as e:
                logger.error(f"Failed to extract '{trait}': {e}")
                continue
        
        return vectors
    
    def combine_vectors(
        self,
        vectors: List[PersonaVector],
        weights: Optional[List[float]] = None
    ) -> PersonaVector:
        """
        Combine multiple persona vectors
        
        Args:
            vectors: List of PersonaVector objects
            weights: Optional weights for each vector
            
        Returns:
            Combined PersonaVector
        """
        if not vectors:
            raise ValueError("Need at least one vector to combine")
        
        if weights is None:
            weights = [1.0] * len(vectors)
        
        if len(weights) != len(vectors):
            raise ValueError("Number of weights must match number of vectors")
        
        # Weighted average
        combined = torch.zeros_like(vectors[0].vector)
        total_weight = sum(weights)

        for vector, weight in zip(vectors, weights):
            combined = combined + (vector.vector * (weight / total_weight))

        with torch.no_grad():
            norm_tensor = torch.linalg.norm(combined)
            norm = norm_tensor.item()
            if norm > 0:
                combined = combined / norm_tensor

        # Create combined trait name
        trait_names = [v.trait for v in vectors]
        combined_trait = "+".join(trait_names)

        return PersonaVector(
            trait=combined_trait,
            vector=combined,
            norm=norm,
            model=vectors[0].model
        )