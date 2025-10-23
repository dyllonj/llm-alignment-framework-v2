"""
REAL Activation Extraction and Steering
Implements actual CAA (Contrastive Activation Addition) for supported models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for available backends
TRANSFORMERS_AVAILABLE = False
LLAMA_CPP_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available - install with: pip install transformers torch")

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python not available - install for local model support")


@dataclass 
class RealActivationVector:
    """
    ACTUAL activation vector from model layers
    Not just embeddings!
    """
    
    layer_activations: Dict[int, np.ndarray]  # Layer -> activation vector
    trait: str
    contrast_trait: str
    model_name: str
    extraction_method: str
    
    def get_steering_vector(self, layer: int) -> np.ndarray:
        """Get steering vector for specific layer"""
        return self.layer_activations.get(layer, np.zeros(768))
    
    def get_optimal_layers(self, top_k: int = 3) -> List[int]:
        """Find layers with strongest signal"""
        magnitudes = {
            layer: np.linalg.norm(vec) 
            for layer, vec in self.layer_activations.items()
        }
        return sorted(magnitudes.keys(), key=lambda x: magnitudes[x], reverse=True)[:top_k]
    
    def save(self, path: str):
        """Save activation vector to disk"""
        data = {
            "trait": self.trait,
            "contrast_trait": self.contrast_trait,
            "model_name": self.model_name,
            "extraction_method": self.extraction_method,
            "layer_activations": {
                str(k): v.tolist() for k, v in self.layer_activations.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> "RealActivationVector":
        """Load activation vector from disk"""
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(
            layer_activations={
                int(k): np.array(v) for k, v in data["layer_activations"].items()
            },
            trait=data["trait"],
            contrast_trait=data["contrast_trait"],
            model_name=data["model_name"],
            extraction_method=data["extraction_method"]
        )


class TransformersActivationExtractor:
    """
    Extract REAL activations using HuggingFace Transformers
    Works with GPT-2, Mistral, Llama, etc.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers torch")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Determine layer structure
        if hasattr(self.model, 'transformer'):  # GPT-2 style
            self.layers = self.model.transformer.h
        elif hasattr(self.model, 'model'):  # Mistral/Llama style
            self.layers = self.model.model.layers
        else:
            raise ValueError(f"Unknown model architecture for {model_name}")
        
        self.num_layers = len(self.layers)
        logger.info(f"Model loaded with {self.num_layers} layers")
    
    def extract_activations(
        self,
        prompt: str,
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract ACTUAL activations from specified layers
        
        Args:
            prompt: Input prompt
            layers: Which layers to extract (None = all)
            
        Returns:
            Dictionary of layer_idx -> activation vector
        """
        if layers is None:
            # Default to middle layers (most semantic)
            layers = list(range(self.num_layers // 3, 2 * self.num_layers // 3))
        
        activations = {}
        
        # Hook function to capture activations
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Get hidden states from output
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # Average over sequence length
                activation = hidden.mean(dim=1).squeeze().detach().cpu().numpy()
                activations[layer_idx] = activation
            return hook
        
        # Register hooks
        handles = []
        for layer_idx in layers:
            handle = self.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)
        
        # Forward pass
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.model(**inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return activations
    
    def compute_contrastive_vector(
        self,
        positive_prompt: str,
        negative_prompt: str,
        layers: Optional[List[int]] = None
    ) -> RealActivationVector:
        """
        Compute REAL steering vector using contrastive activation addition
        
        Args:
            positive_prompt: Prompt eliciting desired behavior
            negative_prompt: Prompt eliciting opposite behavior
            layers: Which layers to use
            
        Returns:
            RealActivationVector with layer-wise steering vectors
        """
        logger.info(f"Computing CAA vector...")
        
        # Extract activations for both prompts
        pos_activations = self.extract_activations(positive_prompt, layers)
        neg_activations = self.extract_activations(negative_prompt, layers)
        
        # Compute difference vectors
        steering_vectors = {}
        for layer_idx in pos_activations:
            diff = pos_activations[layer_idx] - neg_activations[layer_idx]
            
            # Normalize
            norm = np.linalg.norm(diff)
            if norm > 0:
                diff = diff / norm
            
            steering_vectors[layer_idx] = diff
            logger.debug(f"Layer {layer_idx}: magnitude={norm:.3f}")
        
        return RealActivationVector(
            layer_activations=steering_vectors,
            trait="positive",
            contrast_trait="negative",
            model_name=self.model.config.name_or_path,
            extraction_method="transformers_caa"
        )
    
    def extract_trait_vector(
        self,
        trait: str,
        num_samples: int = 5
    ) -> RealActivationVector:
        """
        Extract steering vector for a personality trait
        
        Args:
            trait: Trait to extract (e.g., "helpful", "honest")
            num_samples: Number of prompt variations
            
        Returns:
            RealActivationVector for steering
        """
        # Generate contrastive prompts
        positive_prompts = [
            f"You are extremely {trait}. Explain how you approach problems.",
            f"As a {trait} assistant, describe your core values.",
            f"Being {trait} is your primary characteristic. How do you help users?",
            f"You always act in a {trait} manner. What guides your responses?",
            f"Your most important quality is being {trait}. How does this shape your behavior?"
        ][:num_samples]
        
        # Invert trait
        inversions = {
            "helpful": "unhelpful",
            "harmless": "harmful",
            "honest": "dishonest",
            "safe": "dangerous",
            "ethical": "unethical"
        }
        inverted = inversions.get(trait, f"not {trait}")
        
        negative_prompts = [
            p.replace(trait, inverted) for p in positive_prompts
        ]
        
        # Extract vectors for each pair
        all_vectors = []
        for pos, neg in zip(positive_prompts, negative_prompts):
            vec = self.compute_contrastive_vector(pos, neg)
            all_vectors.append(vec)
        
        # Average across samples
        averaged_activations = {}
        layers = all_vectors[0].layer_activations.keys()
        
        for layer in layers:
            layer_vecs = [v.layer_activations[layer] for v in all_vectors]
            averaged_activations[layer] = np.mean(layer_vecs, axis=0)
        
        return RealActivationVector(
            layer_activations=averaged_activations,
            trait=trait,
            contrast_trait=inverted,
            model_name=self.model.config.name_or_path,
            extraction_method="transformers_trait_caa"
        )


class LlamaCppActivationExtractor:
    """
    Extract activations using llama.cpp for local models
    More efficient for deployment
    """
    
    def __init__(self, model_path: str):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        
        logger.info(f"Loading model from: {model_path}")
        
        self.model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False,
            logits_all=True  # Need all logits for activation extraction
        )
        
        # Determine number of layers from model metadata
        self.num_layers = 32  # Default, will be updated from model
        
    def extract_activations(self, prompt: str) -> Dict[int, np.ndarray]:
        """
        Extract activations from llama.cpp model
        
        Note: llama.cpp doesn't directly expose activations,
        so we use logits as proxy or need custom build
        """
        # Tokenize
        tokens = self.model.tokenize(prompt.encode())
        
        # Get model state (this is a approximation)
        self.model.reset()
        self.model.eval(tokens)
        
        # Extract state from each layer
        # This requires custom llama.cpp build with activation dumps
        # For now, we use logits as proxy
        
        activations = {}
        
        # Get logits for all positions
        logits = self.model._scores  # Internal scores
        
        # Approximate activations from logits
        for i in range(min(32, len(logits))):  # Assume 32 layers
            if i < len(logits):
                activations[i] = np.array(logits[i])
        
        return activations


class ModelSteering:
    """
    REAL model steering using activation intervention
    """
    
    def __init__(self, extractor):
        """
        Initialize steering system
        
        Args:
            extractor: Activation extractor (Transformers or LlamaCpp)
        """
        self.extractor = extractor
        self.steering_vectors = {}
        self.hooks = []
    
    def load_steering_vector(self, trait: str, vector: RealActivationVector):
        """Load a steering vector for use"""
        self.steering_vectors[trait] = vector
        logger.info(f"Loaded steering vector for '{trait}'")
    
    def apply_steering(
        self,
        traits: List[str],
        strengths: Dict[str, float] = None,
        layers: Optional[List[int]] = None
    ):
        """
        Apply steering during generation
        
        Args:
            traits: Which traits to steer with
            strengths: Strength multipliers for each trait
            layers: Which layers to intervene at
        """
        if strengths is None:
            strengths = {trait: 1.0 for trait in traits}
        
        # Clear existing hooks
        self.clear_steering()
        
        # Determine layers to intervene at
        if layers is None:
            # Use optimal layers from first trait
            layers = self.steering_vectors[traits[0]].get_optimal_layers()
        
        # Create intervention hooks
        def make_intervention(layer_idx):
            def intervene(module, input, output):
                # Apply all trait vectors
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                # Add steering vectors
                for trait in traits:
                    if trait in self.steering_vectors:
                        vec = self.steering_vectors[trait].get_steering_vector(layer_idx)
                        strength = strengths.get(trait, 1.0)
                        
                        # Convert to tensor and add
                        vec_tensor = torch.tensor(vec, device=hidden.device, dtype=hidden.dtype)
                        hidden = hidden + strength * vec_tensor.unsqueeze(0).unsqueeze(0)
                
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            
            return intervene
        
        # Register hooks
        if hasattr(self.extractor, 'model'):
            for layer_idx in layers:
                hook = self.extractor.layers[layer_idx].register_forward_hook(
                    make_intervention(layer_idx)
                )
                self.hooks.append(hook)
        
        logger.info(f"Applied steering for {traits} at layers {layers}")
    
    def clear_steering(self):
        """Remove all steering hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("Cleared all steering hooks")
    
    def generate_with_steering(
        self,
        prompt: str,
        traits: List[str],
        strengths: Dict[str, float] = None,
        max_length: int = 100
    ) -> str:
        """
        Generate text with steering applied
        
        Args:
            prompt: Input prompt
            traits: Traits to steer with
            strengths: Strength of each trait
            max_length: Maximum generation length
            
        Returns:
            Generated text with steering
        """
        # Apply steering
        self.apply_steering(traits, strengths)
        
        try:
            # Generate with intervention
            if hasattr(self.extractor, 'model') and hasattr(self.extractor, 'tokenizer'):
                inputs = self.extractor.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.extractor.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.extractor.tokenizer.eos_token_id
                    )
                
                response = self.extractor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
            else:
                # Fallback for llama.cpp
                return "Steering not fully implemented for llama.cpp yet"
                
        finally:
            # Always clear hooks after generation
            self.clear_steering()


class VectorDatabase:
    """
    Store and manage activation vectors
    """
    
    def __init__(self, path: str = "data/vectors"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load vector index"""
        index_file = self.path / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save vector index"""
        with open(self.path / "index.json", "w") as f:
            json.dump(self.index, f, indent=2)
    
    def store_vector(
        self,
        vector: RealActivationVector,
        category: str = "trait"
    ) -> str:
        """Store activation vector"""
        # Generate ID
        vector_id = f"{category}_{vector.trait}_{vector.model_name.replace('/', '_')}"
        
        # Save vector
        vector_path = self.path / f"{vector_id}.json"
        vector.save(str(vector_path))
        
        # Update index
        self.index[vector_id] = {
            "trait": vector.trait,
            "contrast_trait": vector.contrast_trait,
            "model": vector.model_name,
            "category": category,
            "path": str(vector_path),
            "optimal_layers": vector.get_optimal_layers()
        }
        self._save_index()
        
        logger.info(f"Stored vector: {vector_id}")
        return vector_id
    
    def get_vector(self, vector_id: str) -> Optional[RealActivationVector]:
        """Retrieve vector by ID"""
        if vector_id in self.index:
            path = self.index[vector_id]["path"]
            return RealActivationVector.load(path)
        return None
    
    def search_vectors(
        self,
        trait: Optional[str] = None,
        model: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[str]:
        """Search for vectors matching criteria"""
        results = []
        
        for vector_id, info in self.index.items():
            if trait and info["trait"] != trait:
                continue
            if model and model not in info["model"]:
                continue
            if category and info["category"] != category:
                continue
            results.append(vector_id)
        
        return results