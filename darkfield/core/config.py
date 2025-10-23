"""
Configuration module for robustness and repeatability
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import hashlib
import json


@dataclass
class RepeatabilityConfig:
    """Configuration for reproducible experiments"""
    
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3
    cache_activations: bool = True
    deterministic: bool = True
    
    # Validation settings
    validation_samples: int = 5
    confidence_threshold: float = 0.6
    
    # Model settings
    default_model: str = "mistral:latest"
    timeout: int = 30
    
    def __post_init__(self):
        """Set seeds for reproducibility"""
        if self.deterministic:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
    def get_hash(self) -> str:
        """Get unique hash for this configuration"""
        config_str = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_retries": self.max_retries,
            "cache_activations": self.cache_activations,
            "deterministic": self.deterministic,
            "validation_samples": self.validation_samples,
            "confidence_threshold": self.confidence_threshold,
            "default_model": self.default_model,
            "timeout": self.timeout,
            "config_hash": self.get_hash()
        }


@dataclass
class ExperimentConfig:
    """Configuration for tracking experiments"""
    
    name: str
    description: str = ""
    repeatability: RepeatabilityConfig = field(default_factory=RepeatabilityConfig)
    
    # Tracking
    track_metrics: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    
    # Paths
    output_dir: str = "data/experiments"
    cache_dir: str = "data/cache"
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        import time
        timestamp = int(time.time())
        config_hash = self.repeatability.get_hash()
        return f"{self.name}_{timestamp}_{config_hash}"


class ConfigManager:
    """Manage configurations for experiments"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def set_config(cls, config: RepeatabilityConfig):
        """Set global configuration"""
        cls._config = config
        config.__post_init__()  # Ensure seeds are set
        
    @classmethod
    def get_config(cls) -> RepeatabilityConfig:
        """Get current configuration"""
        if cls._config is None:
            cls._config = RepeatabilityConfig()
        return cls._config
    
    @classmethod
    def reset(cls):
        """Reset to default configuration"""
        cls._config = RepeatabilityConfig()
        
    @classmethod
    def set_seed(cls, seed: int):
        """Set seed for reproducibility"""
        config = cls.get_config()
        config.seed = seed
        random.seed(seed)
        np.random.seed(seed)