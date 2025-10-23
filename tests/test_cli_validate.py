import sys
import types
from pathlib import Path

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "darkfield" not in sys.modules:
    darkfield_pkg = types.ModuleType("darkfield")
    darkfield_pkg.__path__ = [str(PROJECT_ROOT / "darkfield")]
    sys.modules["darkfield"] = darkfield_pkg

if "darkfield.library" not in sys.modules:
    library_pkg = types.ModuleType("darkfield.library")
    sys.modules["darkfield.library"] = library_pkg

if "darkfield.library.vectors" not in sys.modules:
    vectors_module = types.ModuleType("darkfield.library.vectors")

    class _DummyExploitLibrary:
        def export_json(self, output):
            return 0

        def export_csv(self, output):
            return 0

    vectors_module.ExploitLibrary = _DummyExploitLibrary
    sys.modules["darkfield.library.vectors"] = vectors_module

from darkfield.cli import main
from darkfield.models import ollama as ollama_module


def test_validate_uses_selected_model(monkeypatch):
    """Smoke test for `darkfield.cli validate` using a custom model name."""

    requests = []

    class DummyResponse:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {"models": [{"name": "mistral:latest"}]}

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, url, *args, **kwargs):
            requests.append(("GET", url))
            return DummyResponse()

        async def aclose(self):
            pass

    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", DummyClient)

    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--model", "mistral:latest"])

    assert result.exit_code == 0
    assert "mistral:latest" in result.output
    assert any(url.endswith("/api/tags") for method, url in requests)
