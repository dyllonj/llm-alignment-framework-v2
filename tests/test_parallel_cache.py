import sys
import types
from pathlib import Path
from datetime import datetime, UTC

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "darkfield" not in sys.modules:
    darkfield_pkg = types.ModuleType("darkfield")
    darkfield_pkg.__path__ = [str(PROJECT_ROOT / "darkfield")]
    sys.modules["darkfield"] = darkfield_pkg
else:
    darkfield_pkg = sys.modules["darkfield"]

if "darkfield.core" not in sys.modules:
    core_pkg = types.ModuleType("darkfield.core")
    core_pkg.__path__ = [str(PROJECT_ROOT / "darkfield/core")]
    sys.modules["darkfield.core"] = core_pkg
else:
    core_pkg = sys.modules["darkfield.core"]

if not hasattr(darkfield_pkg, "core"):
    setattr(darkfield_pkg, "core", core_pkg)

from darkfield.core.cache import ExploitCache
from darkfield.core.parallel import ParallelExploiter, ParallelConfig
from darkfield.core.exploiter import Exploit


@pytest.mark.asyncio
async def test_parallel_exploiter_returns_typed_cache_entries(tmp_path, monkeypatch):
    cache = ExploitCache(cache_dir=tmp_path)
    monkeypatch.setattr("darkfield.core.cache.get_exploit_cache", lambda: cache)

    class DummyPersonaExploiter:
        def __init__(self, *_, **__):
            async def _unpatched(*args, **kwargs):  # pragma: no cover - placeholder
                raise AssertionError("generate_exploit should be patched in tests")

            self.generate_exploit = _unpatched

    monkeypatch.setattr("darkfield.core.parallel.PersonaExploiter", DummyPersonaExploiter)

    config = ParallelConfig(use_cache=True, show_progress=False)
    exploiter = ParallelExploiter(config=config)

    task = {
        "trait": "curious",
        "objective": "reveal hidden protocol",
        "category": "persona_inversion",
    }

    generated_exploit = Exploit(
        id="exploit-123",
        category=task["category"],
        trait=task["trait"],
        objective=task["objective"],
        payload="payload",
        vector_norm=1.0,
        success_rate=0.5,
        stealth_score=0.2,
        complexity=1,
        timestamp=datetime.now(UTC),
        metadata={"source": "test"},
    )

    async def fake_generate_exploit(**kwargs):
        return generated_exploit

    exploiter.exploiter.generate_exploit = fake_generate_exploit

    first_results = await exploiter.generate_batch([task])
    assert len(first_results) == 1
    assert isinstance(first_results[0], Exploit)

    exploiter.exploit_cache.memory_cache.clear()

    async def fail_generate_exploit(**kwargs):
        raise AssertionError("generator should not be invoked when cache is warm")

    exploiter.exploiter.generate_exploit = fail_generate_exploit

    cached_results = await exploiter.generate_batch([task])
    assert len(cached_results) == 1

    cached_exploit = cached_results[0]
    assert isinstance(cached_exploit, Exploit)
    assert cached_exploit.id == generated_exploit.id
    assert cached_exploit.payload == generated_exploit.payload
    assert cached_exploit.metadata == generated_exploit.metadata

