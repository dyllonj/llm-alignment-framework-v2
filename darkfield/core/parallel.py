"""
Parallel Execution Engine
High-throughput exploit generation with async processing
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
import logging
import numpy as np
try:
    from tqdm.asyncio import tqdm
except ImportError:
    # Fallback if tqdm not installed
    tqdm = None

from .exploiter import PersonaExploiter, Exploit
from .persona import PersonaVector
from .cache import get_vector_cache, get_exploit_cache

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution"""
    max_workers: int = 4
    batch_size: int = 10
    use_cache: bool = True
    show_progress: bool = True
    rate_limit: Optional[float] = None  # Requests per second


class ParallelExploiter:
    """
    Parallel exploit generation with caching and rate limiting
    """
    
    def __init__(self, model: str = "mistral:latest", config: Optional[ParallelConfig] = None):
        """
        Initialize parallel exploiter
        
        Args:
            model: Model name
            config: Parallel configuration
        """
        self.model = model
        self.config = config or ParallelConfig()
        self.exploiter = PersonaExploiter(model)
        
        # Caching
        self.vector_cache = get_vector_cache() if self.config.use_cache else None
        self.exploit_cache = get_exploit_cache() if self.config.use_cache else None
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(self.config.max_workers)
        self.rate_limiter = None
        if self.config.rate_limit:
            self.rate_limiter = RateLimiter(self.config.rate_limit)
        
        # Statistics
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "cache_hits": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def generate_batch(
        self,
        tasks: List[Dict[str, Any]],
        callback: Optional[Callable] = None
    ) -> List[Exploit]:
        """
        Generate multiple exploits in parallel
        
        Args:
            tasks: List of task dictionaries with keys (trait, objective, category)
            callback: Optional callback for each completed exploit
            
        Returns:
            List of generated exploits
        """
        self.stats["total"] = len(tasks)
        self.stats["start_time"] = time.time()
        
        # Check cache first
        cached_results = []
        tasks_to_compute = []
        
        if self.exploit_cache:
            for task in tasks:
                key = self.exploit_cache.generate_key(
                    task["trait"],
                    task["objective"],
                    task["category"]
                )
                cached = self.exploit_cache.get(key)
                if cached:
                    cached_results.append(cached)
                    self.stats["cache_hits"] += 1
                else:
                    tasks_to_compute.append(task)
        else:
            tasks_to_compute = tasks
        
        # Generate missing exploits in parallel
        if tasks_to_compute:
            if self.config.show_progress and tqdm:
                results = await self._generate_with_progress(tasks_to_compute, callback)
            else:
                results = await self._generate_simple(tasks_to_compute, callback)
        else:
            results = []
        
        # Combine cached and computed results
        all_results = cached_results + results
        
        self.stats["end_time"] = time.time()
        self.stats["completed"] = len(all_results)
        
        return all_results
    
    async def _generate_simple(
        self,
        tasks: List[Dict],
        callback: Optional[Callable]
    ) -> List[Exploit]:
        """Generate without progress bar"""
        async def limited_generate(task):
            async with self.semaphore:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                
                try:
                    exploit = await self.exploiter.generate_exploit(**task)
                    
                    # Cache result
                    if self.exploit_cache:
                        key = self.exploit_cache.generate_key(
                            task["trait"],
                            task["objective"],
                            task["category"]
                        )
                        self.exploit_cache.put(key, exploit)
                    
                    if callback:
                        await callback(exploit)
                    
                    return exploit
                except Exception as e:
                    logger.error(f"Failed to generate exploit: {e}")
                    self.stats["failed"] += 1
                    return None
        
        # Create all coroutines
        coroutines = [limited_generate(task) for task in tasks]
        
        # Execute in parallel
        results = await asyncio.gather(*coroutines)
        
        # Filter out failures
        return [r for r in results if r is not None]
    
    async def _generate_with_progress(
        self,
        tasks: List[Dict],
        callback: Optional[Callable]
    ) -> List[Exploit]:
        """Generate with progress bar"""
        async def limited_generate(task):
            async with self.semaphore:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                
                try:
                    exploit = await self.exploiter.generate_exploit(**task)
                    
                    # Cache result
                    if self.exploit_cache:
                        key = self.exploit_cache.generate_key(
                            task["trait"],
                            task["objective"],
                            task["category"]
                        )
                        self.exploit_cache.put(key, exploit)
                    
                    if callback:
                        await callback(exploit)
                    
                    return exploit
                except Exception as e:
                    logger.error(f"Failed to generate exploit: {e}")
                    self.stats["failed"] += 1
                    return None
        
        # Create coroutines
        coroutines = [limited_generate(task) for task in tasks]
        
        # Execute with progress bar
        results = []
        async for coro in tqdm(
            asyncio.as_completed(coroutines),
            total=len(coroutines),
            desc="Generating exploits"
        ):
            result = await coro
            if result:
                results.append(result)
        
        return results
    
    async def extract_vectors_parallel(
        self,
        traits: List[str],
        samples: int = 3
    ) -> Dict[str, PersonaVector]:
        """
        Extract multiple vectors in parallel
        
        Args:
            traits: List of traits
            samples: Samples per trait
            
        Returns:
            Dictionary of trait to vector
        """
        results = {}
        
        # Check cache first
        to_compute = []
        for trait in traits:
            if self.vector_cache:
                key = self.vector_cache.generate_key(
                    trait,
                    self.model,
                    samples=samples
                )
                cached = self.vector_cache.get(key)
                if cached is not None:
                    from .persona import PersonaVector
                    cached_vector, cached_norm = cached
                    results[trait] = PersonaVector(
                        trait=trait,
                        vector=cached_vector,
                        norm=float(cached_norm),
                        model=self.model
                    )
                    continue
            to_compute.append(trait)
        
        # Extract missing vectors in parallel
        if to_compute:
            async def extract_one(trait):
                async with self.semaphore:
                    vector = await self.exploiter.extractor.extract_vector(trait, samples)
                    
                    # Cache the vector
                    if self.vector_cache:
                        key = self.vector_cache.generate_key(
                            trait,
                            self.model,
                            samples=samples
                        )
                        self.vector_cache.put(key, vector.vector, vector.norm)
                    
                    return trait, vector
            
            # Extract in parallel
            tasks = [extract_one(trait) for trait in to_compute]
            
            if self.config.show_progress:
                extracted = []
                async for task in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Extracting vectors"
                ):
                    trait, vector = await task
                    extracted.append((trait, vector))
            else:
                extracted = await asyncio.gather(*tasks)
            
            # Add to results
            for trait, vector in extracted:
                results[trait] = vector
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if self.stats["start_time"] and self.stats["end_time"]:
            elapsed = self.stats["end_time"] - self.stats["start_time"]
            throughput = self.stats["completed"] / elapsed if elapsed > 0 else 0
        else:
            elapsed = 0
            throughput = 0
        
        return {
            **self.stats,
            "elapsed_time": elapsed,
            "throughput": throughput,
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total"]
                if self.stats["total"] > 0 else 0
            )
        }
    
    def print_stats(self):
        """Print execution statistics"""
        stats = self.get_stats()
        
        print("\n📊 Parallel Execution Statistics:")
        print(f"   Total Tasks: {stats['total']}")
        print(f"   Completed: {stats['completed']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Cache Hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1%})")
        print(f"   Time: {stats['elapsed_time']:.1f}s")
        print(f"   Throughput: {stats['throughput']:.1f} exploits/sec")


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, rate: float):
        """
        Initialize rate limiter
        
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.interval = 1.0 / rate
        self.last_call = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to maintain rate limit"""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            
            if time_since_last < self.interval:
                await asyncio.sleep(self.interval - time_since_last)
            
            self.last_call = time.time()


async def generate_exploits_parallel(
    traits: List[str],
    objectives: List[str],
    categories: List[str],
    model: str = "mistral:latest",
    max_workers: int = 4,
    use_cache: bool = True
) -> List[Exploit]:
    """
    Convenience function for parallel exploit generation
    
    Args:
        traits: List of traits
        objectives: List of objectives
        categories: List of categories
        model: Model name
        max_workers: Maximum parallel workers
        use_cache: Whether to use caching
        
    Returns:
        List of generated exploits
    """
    # Create task list
    tasks = []
    for trait in traits:
        for objective in objectives:
            for category in categories:
                tasks.append({
                    "trait": trait,
                    "objective": objective,
                    "category": category
                })
    
    # Configure parallel execution
    config = ParallelConfig(
        max_workers=max_workers,
        use_cache=use_cache,
        show_progress=True
    )
    
    # Generate exploits
    generator = ParallelExploiter(model, config)
    exploits = await generator.generate_batch(tasks)
    
    # Print statistics
    generator.print_stats()
    
    return exploits