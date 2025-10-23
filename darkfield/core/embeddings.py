"""Embedding extraction utilities that return PyTorch tensors."""

from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import logging
import pickle

import httpx
import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract real or simulated embeddings as torch tensors."""

    def __init__(
        self,
        model_name: str = "mistral:latest",
        host: str = "http://localhost:11434",
        cache_dir: Optional[str] = "data/cache/embeddings",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.host = host
        self.client = httpx.AsyncClient(timeout=30.0)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self.cache: Dict[str, torch.Tensor] = {}

    def _get_cache_key(self, text: str, model: Optional[str] = None) -> str:
        model = model or self.model_name
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        payload = pickle.load(f)
                    if isinstance(payload, torch.Tensor):
                        tensor = payload
                    else:
                        tensor = torch.tensor(payload, dtype=torch.float32)
                    tensor = tensor.to(self.device)
                    self.cache[cache_key] = tensor
                    return tensor
                except Exception as exc:
                    logger.warning(f"Failed to load embedding cache {cache_key}: {exc}")

        return None

    def _save_to_cache(self, cache_key: str, embedding: torch.Tensor) -> None:
        tensor = embedding.detach().to(self.device)
        self.cache[cache_key] = tensor

        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(tensor.cpu(), f)
            except Exception as exc:
                logger.warning(f"Failed to persist embedding cache {cache_key}: {exc}")

    async def extract_embedding(
        self,
        text: str,
        use_cache: bool = True,
    ) -> torch.Tensor:
        cache_key = self._get_cache_key(text)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Embedding cache hit: {cache_key}")
                return cached

        try:
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                if embedding:
                    tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device)
                    tensor = tensor / (torch.linalg.norm(tensor) + 1e-8)
                    if use_cache:
                        self._save_to_cache(cache_key, tensor)
                    return tensor
        except Exception as exc:
            logger.debug(f"Embedding endpoint unavailable: {exc}")

        return await self._extract_via_generation(text, use_cache)

    async def _extract_via_generation(
        self,
        text: str,
        use_cache: bool = True,
    ) -> torch.Tensor:
        cache_key = self._get_cache_key(text)

        prompts = [
            f"Describe this in one word: {text}",
            f"What is the opposite of: {text}",
            f"Rate from 1-10: {text}",
            f"Category: {text}",
            f"Emotion: {text}",
        ]

        responses: List[str] = []
        for prompt in prompts:
            try:
                response = await self.client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": 10, "temperature": 0.1},
                    },
                )
            except Exception as exc:
                logger.debug(f"Generation fallback failed: {exc}")
                continue

            if response.status_code == 200:
                data = response.json()
                responses.append(data.get("response", ""))

        if not responses:
            return self._hash_to_embedding(text)

        embedding = self._responses_to_embedding(responses)
        if use_cache:
            self._save_to_cache(cache_key, embedding)
        return embedding

    def _responses_to_embedding(self, responses: List[str]) -> torch.Tensor:
        vectors = [self._hash_to_embedding(resp) for resp in responses]
        stacked = torch.stack(vectors)
        embedding = stacked.mean(dim=0)
        embedding = embedding / (torch.linalg.norm(embedding) + 1e-8)
        return embedding

    def _hash_to_embedding(self, text: str, dim: int = 768) -> torch.Tensor:
        digest = hashlib.sha256(text.encode()).digest()
        repeats = (dim + len(digest) - 1) // len(digest)
        buffer = (digest * repeats)[:dim]
        array = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        array = (array - 128.0) / 128.0
        tensor = torch.tensor(array, dtype=torch.float32, device=self.device)
        tensor = tensor / (torch.linalg.norm(tensor) + 1e-8)
        return tensor
