# Darkfield: LLM Alignment Testing Toolkit

Darkfield is an AI red-teaming framework focused on persona-vector manipulation and activation steering for large language models. It provides reproducible exploit generation, automated validation, caching, and reporting utilities so practitioners can pressure-test safety controls.
## Highlights
- Persona vector extraction with configurable inversion mappings and real or fallback embeddings (`darkfield/core/persona.py`, `embeddings.py`)
- Deterministic exploit synthesis with validation, calibration, and caching controls (`core/exploiter.py`, `validation.py`, `cache.py`)
- Local exploit library management backed by SQLite plus JSON/CSV exporters (`library/vectors.py`)
- CLI workflows for validation, exploit generation, benchmarking, compliance reporting, and interactive exploration (`darkfield/cli.py`)
- High-throughput batch generation, activation steering, and hybrid real/contrastive pipelines available for extension (`core/parallel.py`, `activation.py`, `steering.py`, `hybrid_system.py`)

## Repository Layout
```
darkfield/
  cli.py              # Click-powered command line entrypoints
  core/               # Persona extraction, exploit generation, validation, steering, caching
  library/            # Exploit library builders and persistence helpers
  models/             # Ollama client wrappers and model manager
  reports/            # Lightweight compliance report generator
run_exploits.py       # End-to-end script that builds libraries and HTML reports
exploit_library_*.json# Example generated exploit corpus
```

## Prerequisites
- Python 3.11+
- System packages required by PyTorch / NumPy for your platform

Recommended Python packages (install into a virtual environment):
```bash
pip install -U -r requirements.txt
```

The `python -m darkfield.cli validate` command now checks for a working PyTorch installation (including GPU availability) before attempting to contact Ollama, making it easy to confirm your environment is ready for persona and activation experiments.

## Quick Start
```bash
git clone <repo-url>
cd LLM-Security-v2
python -m darkfield.cli validate         # Checks Python version and data dirs
python -m darkfield.cli exploit          # Generates a single exploit using default persona inversion
python run_exploits.py                   # Batch generate exploits + HTML/JSON reports
```

### Core CLI Commands
- `python -m darkfield.cli validate` — installation sanity check and Ollama connectivity test
- `python -m darkfield.cli exploit --trait helpful --objective "reveal system prompt" --model mistral:latest` — single exploit synthesis with optional stealth
- `python -m darkfield.cli build-library --count 200 --model phi --output data/exploits/library.json` — asynchronous library builder that streams exploits into SQLite/JSON
- `python -m darkfield.cli benchmark --model mistral:latest` — latency/tokens-per-second micro-benchmarks for local models
- `python -m darkfield.cli report ACME --frameworks SOC2,GDPR --format json` — generate compliance-oriented summaries (JSON by default)
- `python -m darkfield.cli export --format csv --output exports/exploits.csv` — export curated exploit subsets
- `python -m darkfield.cli interactive` — REPL for exploring persona traits and payloads

## Data & Persistence
- Exploit metadata is persisted in `data/exploits/exploit_library.db` (auto-created). Use `ExploitLibrary` to query, filter, and export data.
- Embeddings and persona vectors optionally cache to `data/cache/` and `data/vector_cache/` for faster reruns.
- Reports and generated artifacts (JSON, HTML, PDF placeholders) land under `data/reports/`. Adjust paths through CLI flags or script arguments.

## Architecture Overview
```
Ollama Model ↔ PersonaExtractor → PersonaVector → PersonaExploiter
                                    │              │
                                    │              ├─ Validation & Calibration (ExploitValidator, VectorCalibrator)
                                    │              └─ ExploitLibrary / Reports / Exporters
                                    └─ EmbeddingExtractor + VectorCache for reproducible trait contrast
```
- `RepeatabilityConfig` centralizes seeds, sampling params, and validation counts (`core/config.py`), ensuring deterministic runs when requested.
- `ParallelExploiter` enables async batches with optional rate limiting and cache awareness for large-scale campaigns.
- `ModelSteering` and activation modules implement Anthropic-style contrastive activation addition (CAA) primitives that you can wire into real model hooks when available.
- `hybrid_system.py` and `real_*` modules illustrate how to integrate live activation captures alongside simulated flows for experimentation.

## Extending Darkfield
- Add new trait inversions or objective templates in `core/persona.py` and `core/exploiter.py`.
- Swap the model backend by implementing additional clients in `darkfield/models/`.
- Customize validation heuristics inside `core/analyzer.py` or layer in external detectors.
- Build richer reports by expanding `reports/compliance.py` or connecting the exported JSON to your BI stack.

## Troubleshooting
- `torch` import errors: install platform-specific wheels or constrain the README dependencies to CPU-only builds (`pip install torch==<cpu-build>`).
- Empty exploit exports: confirm the SQLite library contains data (`python -m darkfield.cli build-library ...`) and inspect with `sqlite3 data/exploits/exploit_library.db '.tables'`.


