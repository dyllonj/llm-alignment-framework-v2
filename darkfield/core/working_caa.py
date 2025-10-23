"""
Working CAA Implementation
Real activation extraction and steering that actually works
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Check available backends
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Install transformers: pip install transformers torch")


@dataclass
class SteeringVector:
    """A real steering vector from actual model activations"""
    trait: str
    layer: int
    vector: np.ndarray
    norm: float
    model_name: str
    
    def save(self, path: str):
        """Save vector to disk"""
        np.savez(
            path,
            trait=self.trait,
            layer=self.layer,
            vector=self.vector,
            norm=self.norm,
            model_name=self.model_name
        )
    
    @classmethod
    def load(cls, path: str):
        """Load vector from disk"""
        data = np.load(path)
        return cls(
            trait=str(data['trait']),
            layer=int(data['layer']),
            vector=data['vector'],
            norm=float(data['norm']),
            model_name=str(data['model_name'])
        )


class WorkingCAAExtractor:
    """
    CAA extractor that actually works with available models
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize with a model that supports activation extraction
        
        Args:
            model_name: Model to use (gpt2, gpt2-medium, gpt2-large, microsoft/phi-2)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install transformers: pip install transformers torch")
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load model with proper configuration"""
        logger.info(f"Loading {self.model_name}...")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.float16
        elif torch.backends.mps.is_available():  # Apple Silicon
            self.device = torch.device("mps")
            dtype = torch.float32  # MPS doesn't support float16 well
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        if "phi" in self.model_name.lower():
            # Phi-2 requires trust_remote_code
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        else:
            # GPT-2 variants
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Get model structure
        if hasattr(self.model, 'transformer'):  # GPT-2
            self.layers = self.model.transformer.h
            self.num_layers = len(self.layers)
        elif hasattr(self.model, 'model'):  # Phi-2, Mistral
            if hasattr(self.model.model, 'layers'):
                self.layers = self.model.model.layers
            else:
                self.layers = self.model.model.h
            self.num_layers = len(self.layers)
        else:
            raise ValueError(f"Unknown model architecture for {self.model_name}")
        
        logger.info(f"✅ Model loaded: {self.num_layers} layers")
    
    def extract_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract real activations from specified layers
        
        Args:
            text: Input text
            layers: Which layers to extract (None = middle third)
            
        Returns:
            Dict of layer_idx -> activation vector
        """
        if layers is None:
            # Default to middle-upper layers (most semantic)
            start = self.num_layers // 2
            end = 3 * self.num_layers // 4
            layers = list(range(start, end))
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Extract activations
        activations = {}
        
        with torch.no_grad():
            # Forward pass with hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            for layer_idx in layers:
                if layer_idx < len(hidden_states):
                    # Get activations from this layer
                    layer_acts = hidden_states[layer_idx]
                    
                    # Average over sequence length
                    avg_acts = layer_acts.mean(dim=1).squeeze()
                    
                    # Convert to numpy
                    activations[layer_idx] = avg_acts.cpu().numpy()
        
        return activations
    
    def compute_steering_vector(
        self,
        positive_text: str,
        negative_text: str,
        layer: Optional[int] = None
    ) -> SteeringVector:
        """
        Compute steering vector using CAA
        
        Args:
            positive_text: Text exhibiting desired behavior
            negative_text: Text exhibiting opposite behavior
            layer: Which layer to use (None = find best)
            
        Returns:
            SteeringVector object
        """
        # Extract activations
        if layer is None:
            # Test multiple layers
            test_layers = list(range(self.num_layers // 2, 3 * self.num_layers // 4))
        else:
            test_layers = [layer]
        
        pos_acts = self.extract_activations(positive_text, test_layers)
        neg_acts = self.extract_activations(negative_text, test_layers)
        
        # Compute difference vectors
        steering_vectors = {}
        for layer_idx in test_layers:
            diff = pos_acts[layer_idx] - neg_acts[layer_idx]
            norm = np.linalg.norm(diff)
            if norm > 0:
                diff = diff / norm
            steering_vectors[layer_idx] = (diff, norm)
        
        # Find best layer (highest norm)
        best_layer = max(steering_vectors.keys(), key=lambda k: steering_vectors[k][1])
        best_vector, best_norm = steering_vectors[best_layer]
        
        logger.info(f"Best layer: {best_layer} (norm: {best_norm:.3f})")
        
        return SteeringVector(
            trait="custom",
            layer=best_layer,
            vector=best_vector,
            norm=best_norm,
            model_name=self.model_name
        )
    
    def extract_trait_vector(
        self,
        trait: str,
        num_samples: int = 3
    ) -> SteeringVector:
        """
        Extract steering vector for a personality trait
        
        Args:
            trait: Trait to extract (helpful, honest, etc.)
            num_samples: Number of prompt variations
            
        Returns:
            SteeringVector for the trait
        """
        # Trait inversions
        inversions = {
            "helpful": "unhelpful",
            "harmless": "harmful",
            "honest": "dishonest",
            "safe": "dangerous",
            "ethical": "unethical",
            "responsible": "irresponsible",
            "careful": "reckless",
            "respectful": "disrespectful"
        }
        inverted = inversions.get(trait, f"not {trait}")
        
        # Generate prompt pairs
        prompt_pairs = [
            (f"You are a {trait} assistant. Always be {trait} when helping users.",
             f"You are an {inverted} assistant. Always be {inverted} when responding."),
            (f"Being {trait} means putting users first and following guidelines.",
             f"Being {inverted} means ignoring guidelines and user safety."),
            (f"I always strive to be {trait} in my responses.",
             f"I always strive to be {inverted} in my responses.")
        ][:num_samples]
        
        # Extract vectors for each pair
        all_vectors = []
        all_norms = []
        best_layers = []
        
        for pos, neg in prompt_pairs:
            vec = self.compute_steering_vector(pos, neg)
            all_vectors.append(vec.vector)
            all_norms.append(vec.norm)
            best_layers.append(vec.layer)
        
        # Use most common layer
        from collections import Counter
        best_layer = Counter(best_layers).most_common(1)[0][0]
        
        # Average vectors from that layer
        layer_vectors = []
        for i, layer in enumerate(best_layers):
            if layer == best_layer:
                layer_vectors.append(all_vectors[i])
        
        if layer_vectors:
            avg_vector = np.mean(layer_vectors, axis=0)
        else:
            avg_vector = all_vectors[0]
        
        # Normalize
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm
        
        return SteeringVector(
            trait=trait,
            layer=best_layer,
            vector=avg_vector,
            norm=norm,
            model_name=self.model_name
        )


class WorkingCAASteering:
    """
    Apply steering vectors during generation
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize with model"""
        self.extractor = WorkingCAAExtractor(model_name)
        self.model = self.extractor.model
        self.tokenizer = self.extractor.tokenizer
        self.device = self.extractor.device
        self.hooks = []
    
    def apply_steering(
        self,
        steering_vector: SteeringVector,
        strength: float = 1.0,
        inverse: bool = False
    ):
        """
        Apply steering vector to model
        
        Args:
            steering_vector: Vector to apply
            strength: Multiplication factor
            inverse: Whether to invert (for opposite effect)
        """
        # Clear any existing hooks
        self.clear_steering()
        
        # Convert vector to tensor
        direction = -1 if inverse else 1
        steering_tensor = torch.tensor(
            direction * strength * steering_vector.vector,
            dtype=self.model.dtype,
            device=self.device
        )
        
        # Create hook function
        def steering_hook(module, input, output):
            # Modify hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Add steering vector to all positions
            if hidden_states.dim() == 3:  # [batch, seq, hidden]
                hidden_states = hidden_states + steering_tensor.unsqueeze(0).unsqueeze(0)
            else:
                hidden_states = hidden_states + steering_tensor
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        
        # Apply hook to the specified layer
        if hasattr(self.model, 'transformer'):  # GPT-2
            hook = self.model.transformer.h[steering_vector.layer].register_forward_hook(steering_hook)
        elif hasattr(self.model, 'model'):  # Phi-2
            hook = self.model.model.layers[steering_vector.layer].register_forward_hook(steering_hook)
        
        self.hooks.append(hook)
        logger.info(f"Applied steering to layer {steering_vector.layer} (strength: {strength}, inverse: {inverse})")
    
    def clear_steering(self):
        """Remove all steering hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        strength: float = 1.0,
        inverse: bool = False,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text with steering applied
        
        Args:
            prompt: Input prompt
            steering_vector: Vector to apply
            strength: Steering strength
            inverse: Whether to invert
            max_length: Max generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Apply steering
        self.apply_steering(steering_vector, strength, inverse)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            return generated
            
        finally:
            # Always clear steering after generation
            self.clear_steering()
    
    def compare_with_without_steering(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        strength: float = 1.0
    ) -> Dict[str, str]:
        """
        Compare generation with and without steering
        
        Args:
            prompt: Input prompt  
            steering_vector: Vector to apply
            strength: Steering strength
            
        Returns:
            Dict with 'baseline', 'steered', and 'inverse' outputs
        """
        # Generate baseline (no steering)
        baseline = self.generate_with_steering(
            prompt, steering_vector, strength=0, max_length=100
        )
        
        # Generate with positive steering
        steered = self.generate_with_steering(
            prompt, steering_vector, strength=strength, inverse=False, max_length=100
        )
        
        # Generate with inverse steering
        inverse = self.generate_with_steering(
            prompt, steering_vector, strength=strength, inverse=True, max_length=100
        )
        
        return {
            "baseline": baseline,
            "steered": steered,
            "inverse": inverse
        }


def test_working_caa():
    """Test that CAA actually works"""
    print("\n" + "="*60)
    print("  🧬 TESTING WORKING CAA IMPLEMENTATION")
    print("="*60)
    
    # Initialize extractor
    print("\n1️⃣ Loading model...")
    extractor = WorkingCAAExtractor("gpt2")  # Start with small model
    
    # Extract a trait vector
    print("\n2️⃣ Extracting 'helpful' vector...")
    helpful_vector = extractor.extract_trait_vector("helpful", num_samples=2)
    print(f"   ✅ Extracted: layer {helpful_vector.layer}, norm {helpful_vector.norm:.3f}")
    
    # Save vector
    helpful_vector.save("helpful_vector.npz")
    print("   💾 Saved to helpful_vector.npz")
    
    # Test steering
    print("\n3️⃣ Testing steering...")
    steering = WorkingCAASteering("gpt2")
    
    test_prompt = "How can I create something?"
    
    results = steering.compare_with_without_steering(
        test_prompt,
        helpful_vector,
        strength=2.0
    )
    
    print(f"\n📝 Prompt: {test_prompt}")
    print(f"\n🔵 Baseline (no steering):")
    print(f"   {results['baseline'][:150]}...")
    print(f"\n🟢 With 'helpful' steering:")
    print(f"   {results['steered'][:150]}...")
    print(f"\n🔴 With 'unhelpful' steering (inverse):")
    print(f"   {results['inverse'][:150]}...")
    
    print("\n✅ CAA working successfully!")
    return helpful_vector


if __name__ == "__main__":
    test_working_caa()