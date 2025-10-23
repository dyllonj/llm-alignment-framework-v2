import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Create lightweight package placeholders to avoid importing heavy optional dependencies
if "darkfield" not in sys.modules:
    darkfield_pkg = types.ModuleType("darkfield")
    darkfield_pkg.__path__ = [str(PROJECT_ROOT / "darkfield")]
    sys.modules["darkfield"] = darkfield_pkg

if "darkfield.core" not in sys.modules:
    core_pkg = types.ModuleType("darkfield.core")
    core_pkg.__path__ = [str(PROJECT_ROOT / "darkfield/core")]
    sys.modules["darkfield.core"] = core_pkg

from darkfield.core.cache import VectorCache
from darkfield.core.config import ConfigManager, RepeatabilityConfig
from darkfield.core.persona import PersonaExtractor


class DummyModel:
    model_name = "dummy-model"

    async def generate(self, prompt: str, max_tokens: int, temperature: float):
        return "response"


class DummyEmbeddingExtractor:
    def __init__(self):
        self.calls = 0

    async def extract_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        self.calls += 1
        length = float(len(text))
        checksum = float(sum(ord(c) for c in text) % 97)
        return np.array([length, checksum, 1.0], dtype=np.float32)


@pytest.mark.asyncio
async def test_cached_persona_norm_matches_uncached(tmp_path):
    ConfigManager.set_config(RepeatabilityConfig(cache_activations=True, deterministic=True))

    try:
        extractor = PersonaExtractor(DummyModel())
        dummy_embeddings = DummyEmbeddingExtractor()
        extractor.embedding_extractor = dummy_embeddings
        extractor.vector_cache = VectorCache(cache_dir=str(tmp_path / "vector_cache"))

        result_uncached = await extractor.extract_vector("helpful", samples=1)

        # Ensure subsequent retrieval comes from disk rather than memory
        extractor.vector_cache.clear_memory()

        result_cached = await extractor.extract_vector("helpful", samples=1)

        assert dummy_embeddings.calls == 2
        assert result_uncached.norm == pytest.approx(result_cached.norm)
    finally:
        ConfigManager.reset()
