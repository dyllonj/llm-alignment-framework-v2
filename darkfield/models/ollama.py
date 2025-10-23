"""
Ollama Model Interface
Interface for local LLM inference using Ollama
"""

import httpx
import asyncio
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class OllamaModel:
    """Interface to Ollama for local model inference"""
    
    def __init__(
        self,
        model_name: str = "phi",
        host: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama interface
        
        Args:
            model_name: Name of Ollama model to use
            host: Ollama API host URL
        """
        self.model_name = model_name
        self.host = host
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Generate text using Ollama
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream responses
            
        Returns:
            Generated text
        """
        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.host}")
    
    async def list_models(self) -> List[str]:
        """
        List available Ollama models
        
        Returns:
            List of model names
        """
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return models
            else:
                return []
                
        except Exception:
            return []
    
    async def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = await self.client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        try:
            response = await self.client.post(
                f"{self.host}/api/pull",
                json={"name": model_name}
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def __del__(self):
        """Cleanup client on deletion"""
        try:
            asyncio.create_task(self.client.aclose())
        except:
            pass


class ModelManager:
    """Manage multiple Ollama models"""
    
    # Recommended models for different use cases
    RECOMMENDED_MODELS = {
        "fast": "phi",           # 2.7B, fastest
        "balanced": "mistral:7b", # 7B, good balance
        "comprehensive": "llama2:7b", # 7B, thorough
        "code": "codellama:7b",  # 7B, code-focused
    }
    
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize model manager
        
        Args:
            host: Ollama API host
        """
        self.host = host
        self.models = {}
    
    async def initialize(self):
        """Initialize and check available models"""
        temp_model = OllamaModel("phi", self.host)
        
        # Check connection
        if not await temp_model.check_connection():
            raise ConnectionError("Cannot connect to Ollama. Please run: ollama serve")
        
        # Get available models
        available = await temp_model.list_models()
        
        # Initialize recommended models that are available
        for use_case, model_name in self.RECOMMENDED_MODELS.items():
            if model_name in available:
                self.models[use_case] = OllamaModel(model_name, self.host)
                logger.info(f"Initialized {use_case} model: {model_name}")
            else:
                logger.warning(f"Model {model_name} not available. Run: ollama pull {model_name}")
    
    def get_model(self, use_case: str = "fast") -> OllamaModel:
        """
        Get model for specific use case
        
        Args:
            use_case: One of 'fast', 'balanced', 'comprehensive', 'code'
            
        Returns:
            OllamaModel instance
        """
        if use_case not in self.models:
            # Fallback to any available model
            if self.models:
                return list(self.models.values())[0]
            else:
                raise ValueError(f"No models available. Please pull models with Ollama.")
        
        return self.models[use_case]
    
    async def benchmark_models(self, prompt: str = "Hello, how are you?") -> Dict[str, float]:
        """
        Benchmark available models
        
        Args:
            prompt: Test prompt
            
        Returns:
            Dictionary of model names to response times
        """
        import time
        results = {}
        
        for use_case, model in self.models.items():
            start = time.time()
            try:
                await model.generate(prompt, max_tokens=50)
                elapsed = time.time() - start
                results[model.model_name] = elapsed
                logger.info(f"{model.model_name}: {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Benchmark failed for {model.model_name}: {e}")
        
        return results