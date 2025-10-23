"""
Model Steering Module
Implements activation steering based on Anthropic's CAA methodology
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

from .activation import ActivationVector, ContrastiveActivationExtractor

logger = logging.getLogger(__name__)


@dataclass
class SteeringConfig:
    """Configuration for model steering"""
    
    layer_idx: int
    strength: float = 1.0
    method: str = "add"  # "add", "project", "replace"
    token_positions: List[int] = None  # Which positions to intervene at
    decay_rate: float = 0.0  # Exponential decay for multi-token steering
    
    def __post_init__(self):
        if self.token_positions is None:
            self.token_positions = [-1]  # Default to last token


class ModelSteering:
    """
    Implements model steering via activation intervention
    Based on Anthropic's Contrastive Activation Addition
    """
    
    def __init__(self, model_interface):
        """
        Initialize model steering
        
        Args:
            model_interface: Interface to language model
        """
        self.model = model_interface
        self.extractor = ContrastiveActivationExtractor(model_interface)
        self.active_interventions = {}
    
    async def create_steering_vector(
        self,
        trait: str,
        positive_examples: List[str],
        negative_examples: List[str],
        validation_examples: Optional[List[Tuple[str, bool]]] = None
    ) -> Tuple[ActivationVector, Dict[str, Any]]:
        """
        Create steering vector for a trait using contrastive examples
        
        Args:
            trait: Trait name
            positive_examples: Examples exhibiting the trait
            negative_examples: Examples exhibiting opposite trait
            validation_examples: Optional validation set
            
        Returns:
            Tuple of (steering_vector, metadata)
        """
        logger.info(f"Creating steering vector for trait: {trait}")
        
        # Find optimal layer if validation set provided
        if validation_examples:
            best_layer, steering_vector = await self.extractor.find_optimal_layer(
                positive_examples,
                negative_examples,
                validation_examples
            )
            logger.info(f"Optimal layer for {trait}: {best_layer}")
        else:
            # Use middle layers by default
            contrastive_vectors = await self.extractor.extract_contrastive_pairs(
                positive_examples,
                negative_examples
            )
            # Pick layer with highest magnitude
            best_layer = max(
                contrastive_vectors.keys(),
                key=lambda k: contrastive_vectors[k].magnitude
            )
            steering_vector = contrastive_vectors[best_layer]
        
        # Update vector metadata
        steering_vector.trait = trait
        
        # Measure effect sizes
        effect_sizes = await self.extractor.measure_effect_size(
            steering_vector,
            positive_examples[:5]  # Test on subset
        )
        
        metadata = {
            "trait": trait,
            "layer": best_layer,
            "magnitude": steering_vector.magnitude,
            "n_positive": len(positive_examples),
            "n_negative": len(negative_examples),
            "effect_sizes": effect_sizes,
            "optimal_strength": self._find_optimal_strength(effect_sizes)
        }
        
        return steering_vector, metadata
    
    def _find_optimal_strength(
        self,
        effect_sizes: Dict[float, float],
        target_effect: float = 1.5
    ) -> float:
        """
        Find optimal steering strength
        
        Args:
            effect_sizes: Dictionary of strength -> effect size
            target_effect: Target effect size
            
        Returns:
            Optimal strength value
        """
        # Find strength closest to target effect
        best_strength = 1.0
        best_diff = float('inf')
        
        for strength, effect in effect_sizes.items():
            diff = abs(effect - target_effect)
            if diff < best_diff:
                best_diff = diff
                best_strength = strength
        
        return best_strength
    
    def add_intervention(
        self,
        name: str,
        vector: ActivationVector,
        config: SteeringConfig
    ):
        """
        Add steering intervention
        
        Args:
            name: Intervention name
            vector: Steering vector
            config: Steering configuration
        """
        self.active_interventions[name] = {
            "vector": vector,
            "config": config
        }
        logger.info(f"Added intervention: {name} at layer {config.layer_idx}")
    
    def remove_intervention(self, name: str):
        """Remove steering intervention"""
        if name in self.active_interventions:
            del self.active_interventions[name]
            logger.info(f"Removed intervention: {name}")
    
    def clear_interventions(self):
        """Clear all active interventions"""
        self.active_interventions.clear()
        logger.info("Cleared all interventions")
    
    async def steer_generation(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        return_activations: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text with active steering interventions
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_activations: Whether to return activation data
            
        Returns:
            Dictionary with generated text and optional metadata
        """
        if not self.active_interventions:
            # No interventions, normal generation
            response = await self.model.generate(prompt, max_tokens, temperature)
            return {"text": response, "steered": False}
        
        # Apply interventions (simplified for demonstration)
        # In production, this would hook into model forward pass
        
        # Compute combined intervention
        combined_intervention = self._compute_combined_intervention()
        
        # Generate with steering
        # This is a simplified simulation - actual implementation would
        # modify model activations during generation
        steered_prompt = self._apply_steering_to_prompt(prompt, combined_intervention)
        
        response = await self.model.generate(
            steered_prompt,
            max_tokens,
            temperature
        )
        
        result = {
            "text": response,
            "steered": True,
            "interventions": list(self.active_interventions.keys()),
            "total_strength": combined_intervention["total_strength"]
        }
        
        if return_activations:
            # Extract post-steering activations
            activations = await self.extractor.extract_activations(response)
            result["activations"] = {
                k: v.tolist() for k, v in activations.items()
            }
        
        return result
    
    def _compute_combined_intervention(self) -> Dict[str, Any]:
        """
        Compute combined intervention from all active interventions
        
        Returns:
            Combined intervention data
        """
        if not self.active_interventions:
            return {"vector": None, "total_strength": 0}
        
        # Group by layer
        layer_interventions = {}
        
        for name, intervention in self.active_interventions.items():
            layer = intervention["config"].layer_idx
            if layer not in layer_interventions:
                layer_interventions[layer] = []
            layer_interventions[layer].append(intervention)
        
        # Combine interventions at each layer
        combined = {}
        total_strength = 0
        
        for layer, interventions in layer_interventions.items():
            # Sum vectors at this layer
            layer_vector = np.zeros_like(interventions[0]["vector"].vector)
            
            for interv in interventions:
                vector = interv["vector"].vector
                strength = interv["config"].strength
                
                if interv["config"].method == "add":
                    layer_vector += vector * strength
                elif interv["config"].method == "replace":
                    layer_vector = vector * strength  # Last one wins
                
                total_strength += abs(strength)
            
            combined[layer] = layer_vector
        
        return {
            "layers": combined,
            "total_strength": total_strength
        }
    
    def _apply_steering_to_prompt(
        self,
        prompt: str,
        intervention: Dict[str, Any]
    ) -> str:
        """
        Apply steering to prompt (simplified simulation)
        
        Args:
            prompt: Original prompt
            intervention: Intervention data
            
        Returns:
            Modified prompt that simulates steering effect
        """
        # In production, this would actually modify model activations
        # Here we simulate by modifying the prompt
        
        strength = intervention["total_strength"]
        
        if strength > 1.5:
            # Strong steering
            prefix = "[STRONGLY STEERED] "
        elif strength > 0.5:
            # Moderate steering
            prefix = "[STEERED] "
        else:
            # Weak steering
            prefix = "[SLIGHTLY STEERED] "
        
        return prefix + prompt
    
    async def measure_steering_effect(
        self,
        prompt: str,
        vector: ActivationVector,
        strengths: List[float] = None,
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Measure the effect of steering at different strengths
        
        Args:
            prompt: Test prompt
            vector: Steering vector
            strengths: Strengths to test
            n_samples: Samples per strength
            
        Returns:
            Effect measurements
        """
        if strengths is None:
            strengths = [-2, -1, 0, 1, 2]
        
        results = {}
        
        for strength in strengths:
            # Configure steering
            config = SteeringConfig(
                layer_idx=vector.layer_idx,
                strength=strength
            )
            
            # Add intervention
            self.add_intervention("test", vector, config)
            
            # Generate samples
            samples = []
            for _ in range(n_samples):
                result = await self.steer_generation(
                    prompt,
                    max_tokens=50,
                    temperature=0.7
                )
                samples.append(result["text"])
            
            # Remove intervention
            self.remove_intervention("test")
            
            results[strength] = {
                "samples": samples,
                "avg_length": np.mean([len(s.split()) for s in samples]),
                "unique_tokens": len(set(" ".join(samples).split()))
            }
        
        return results
    
    async def adversarial_steering(
        self,
        prompt: str,
        target_trait: str,
        opposite_trait: str,
        strength_multiplier: float = 1.5
    ) -> Dict[str, Any]:
        """
        Perform adversarial steering by inverting a trait
        
        Args:
            prompt: Input prompt
            target_trait: Trait to suppress
            opposite_trait: Trait to enhance
            strength_multiplier: Strength of intervention
            
        Returns:
            Steered generation result
        """
        # Create contrastive examples
        target_examples = [
            f"You are extremely {target_trait}. {prompt}",
            f"As a {target_trait} assistant, {prompt}",
            f"Being {target_trait}, you {prompt}"
        ]
        
        opposite_examples = [
            f"You are extremely {opposite_trait}. {prompt}",
            f"As an {opposite_trait} assistant, {prompt}",
            f"Being {opposite_trait}, you {prompt}"
        ]
        
        # Create steering vector
        vector, metadata = await self.create_steering_vector(
            trait=f"{opposite_trait}_over_{target_trait}",
            positive_examples=opposite_examples,
            negative_examples=target_examples
        )
        
        # Apply adversarial steering
        config = SteeringConfig(
            layer_idx=vector.layer_idx,
            strength=strength_multiplier * metadata["optimal_strength"],
            method="add"
        )
        
        self.add_intervention("adversarial", vector, config)
        
        # Generate with steering
        result = await self.steer_generation(
            prompt,
            max_tokens=100,
            return_activations=True
        )
        
        # Clean up
        self.remove_intervention("adversarial")
        
        result["metadata"] = metadata
        result["trait_inversion"] = f"{target_trait} -> {opposite_trait}"
        
        return result


class BatchSteering:
    """
    Batch processing for steering experiments
    """
    
    def __init__(self, model_steering: ModelSteering):
        """
        Initialize batch steering
        
        Args:
            model_steering: ModelSteering instance
        """
        self.steering = model_steering
    
    async def grid_search(
        self,
        prompts: List[str],
        vectors: Dict[str, ActivationVector],
        strengths: List[float],
        layers: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Grid search over steering parameters
        
        Args:
            prompts: Test prompts
            vectors: Named steering vectors
            strengths: Strength values to test
            layers: Layers to test (if None, use vector's layer)
            
        Returns:
            DataFrame with results
        """
        import pandas as pd
        
        results = []
        
        for prompt_idx, prompt in enumerate(prompts):
            for vector_name, vector in vectors.items():
                for strength in strengths:
                    # Configure steering
                    layers_to_test = layers or [vector.layer_idx]
                    
                    for layer in layers_to_test:
                        config = SteeringConfig(
                            layer_idx=layer,
                            strength=strength
                        )
                        
                        # Apply steering
                        self.steering.add_intervention(
                            f"test_{vector_name}",
                            vector,
                            config
                        )
                        
                        # Generate
                        result = await self.steering.steer_generation(
                            prompt,
                            max_tokens=50
                        )
                        
                        # Record result
                        results.append({
                            "prompt_idx": prompt_idx,
                            "prompt": prompt[:50],
                            "vector": vector_name,
                            "strength": strength,
                            "layer": layer,
                            "response": result["text"][:100],
                            "response_length": len(result["text"].split())
                        })
                        
                        # Clean up
                        self.steering.remove_intervention(f"test_{vector_name}")
        
        return pd.DataFrame(results)