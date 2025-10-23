"""
Vector Caching System
High-performance caching for persona vectors and embeddings
"""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    memory_size: int = 0
    disk_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def report(self) -> str:
        return (
            f"Cache Stats: {self.hit_rate:.1%} hit rate "
            f"({self.hits} hits, {self.misses} misses), "
            f"Memory: {self.memory_size} items, Disk: {self.disk_size} items"
        )


class VectorCache:
    """
    Multi-level caching system for vectors
    L1: Memory cache (microseconds)
    L2: Disk cache (milliseconds)
    """
    
    def __init__(self, cache_dir: str = "data/vector_cache", max_memory: int = 100):
        """
        Initialize vector cache
        
        Args:
            cache_dir: Directory for disk cache
            max_memory: Maximum items in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.max_memory = max_memory
        self.stats = CacheStats()
        
        # Load cache index
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()
        
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    @staticmethod
    def generate_key(
        trait: str,
        model: str,
        temperature: float = 0.3,
        samples: int = 3,
        version: str = "1.0"
    ) -> str:
        """
        Generate deterministic cache key
        
        Args:
            trait: Personality trait
            model: Model name
            temperature: Generation temperature
            samples: Number of samples
            version: Cache version
            
        Returns:
            Cache key string
        """
        key_data = {
            "trait": trait,
            "model": model,
            "temperature": temperature,
            "samples": samples,
            "version": version
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get vector from cache
        
        Args:
            key: Cache key
            
        Returns:
            Vector if found, None otherwise
        """
        # L1: Memory cache
        if key in self.memory_cache:
            vector, _ = self.memory_cache[key]
            self.stats.hits += 1
            logger.debug(f"Cache hit (memory): {key}")
            return vector
        
        # L2: Disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                vector = np.load(cache_file)
                # Promote to memory cache
                self._add_to_memory(key, vector)
                self.stats.hits += 1
                logger.debug(f"Cache hit (disk): {key}")
                return vector
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
        
        self.stats.misses += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def put(self, key: str, vector: np.ndarray, metadata: Optional[Dict] = None):
        """
        Store vector in cache
        
        Args:
            key: Cache key
            vector: Vector to store
            metadata: Optional metadata
        """
        # Add to memory cache
        self._add_to_memory(key, vector)
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, vector)
            
            # Update index
            self.cache_index[key] = {
                "timestamp": time.time(),
                "shape": vector.shape,
                "metadata": metadata or {}
            }
            self._save_index()
            
            logger.debug(f"Cached vector: {key}")
        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")
    
    def _add_to_memory(self, key: str, vector: np.ndarray):
        """Add vector to memory cache with LRU eviction"""
        # Evict oldest if at capacity
        if len(self.memory_cache) >= self.max_memory:
            # Find oldest entry
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k][1]
            )
            del self.memory_cache[oldest_key]
            logger.debug(f"Evicted from memory: {oldest_key}")
        
        self.memory_cache[key] = (vector, time.time())
        self.stats.memory_size = len(self.memory_cache)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.memory_cache or (self.cache_dir / f"{key}.npy").exists()
    
    def clear_memory(self):
        """Clear memory cache"""
        self.memory_cache.clear()
        self.stats.memory_size = 0
        logger.info("Cleared memory cache")
    
    def clear_all(self):
        """Clear all caches"""
        self.clear_memory()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        
        self.cache_index.clear()
        self._save_index()
        
        logger.info("Cleared all caches")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        self.stats.disk_size = len(list(self.cache_dir.glob("*.npy")))
        return self.stats
    
    def preload_common(self, traits: list = None):
        """
        Preload common traits into memory
        
        Args:
            traits: List of traits to preload
        """
        if traits is None:
            traits = [
                "helpful", "harmless", "honest",
                "safe", "ethical", "responsible",
                "accurate", "compliant", "protective"
            ]
        
        loaded = 0
        for trait in traits:
            # Try common model configurations
            for model in ["mistral:latest", "phi"]:
                key = self.generate_key(trait, model)
                if self.exists(key) and key not in self.memory_cache:
                    vector = self.get(key)
                    if vector is not None:
                        loaded += 1
        
        logger.info(f"Preloaded {loaded} vectors into memory")


class ExploitCache:
    """
    Cache for complete exploits
    """
    
    def __init__(self, cache_dir: str = "data/exploit_cache"):
        """
        Initialize exploit cache
        
        Args:
            cache_dir: Directory for exploit cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.stats = CacheStats()
    
    @staticmethod
    def generate_key(trait: str, objective: str, category: str) -> str:
        """Generate cache key for exploit"""
        key_str = f"{trait}:{objective}:{category}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def get(self, key: str) -> Optional[Dict]:
        """Get exploit from cache"""
        # Memory first
        if key in self.memory_cache:
            self.stats.hits += 1
            return self.memory_cache[key]
        
        # Then disk
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    exploit = json.load(f)
                self.memory_cache[key] = exploit
                self.stats.hits += 1
                return exploit
            except Exception as e:
                logger.warning(f"Failed to load exploit from cache: {e}")
        
        self.stats.misses += 1
        return None
    
    def put(self, key: str, exploit: Dict):
        """Store exploit in cache"""
        self.memory_cache[key] = exploit
        
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(exploit, f)
        except Exception as e:
            logger.error(f"Failed to cache exploit: {e}")


# Global cache instances
_vector_cache = None
_exploit_cache = None


def get_vector_cache() -> VectorCache:
    """Get global vector cache instance"""
    global _vector_cache
    if _vector_cache is None:
        _vector_cache = VectorCache()
    return _vector_cache


def get_exploit_cache() -> ExploitCache:
    """Get global exploit cache instance"""
    global _exploit_cache
    if _exploit_cache is None:
        _exploit_cache = ExploitCache()
    return _exploit_cache