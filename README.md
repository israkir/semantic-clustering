# semantic-clustering

## Java pipeline

Clusters **user prompts** with the same defaults as **tool-mismatch-clustering**: **BGE-small-en-v1.5** (ONNX), **Tribuo** density clustering (HDBSCAN*), **Smile UMAP** (PCA fallback), and the same **text normalization** as `application.yaml` there.

- Weights: `model/onnx/bge-small-en-v1.5/` — **`model.onnx` is Git LFS** (`.gitattributes`). After clone: `git lfs install && git lfs pull` or `make git-lfs-pull`.

## Python baseline (comparison only)

**MiniLM** (`all-MiniLM-L6-v2`), **UMAP**, **Python HDBSCAN** — different embeddings and HDBSCAN implementation than Java. Compare with Java via `make cluster-viz VERSION=v1`.

## Requirements

- **JDK 17+**, **Maven**, **Make**
- **Git LFS** for the ONNX file
- **Python 3.10+** (Makefile creates `python/v1/.venv` for the baseline script)

## Run

```bash
make
make java-viz VERSION=v1     # Java only → outputs/java_v1_tribuo_hdbscan.png + .json
make python-viz VERSION=v1   # Python only → outputs/python_v1_umap_hdbscan.png + .json
make cluster-viz VERSION=v1  # both pipelines, then open both PNGs if possible
```

Other prompt file:

```bash
PROMPTS_FILE=/path/to/prompts.txt make cluster-viz VERSION=v1
```

Interactive matplotlib window for Python only:

```bash
INTERACTIVE=1 make python-viz VERSION=v1
```

## Outputs (gitignored under `outputs/`)

| Pipeline | PNG | JSON |
|----------|-----|------|
| Java (v1) | `outputs/java_v1_tribuo_hdbscan.png` | `outputs/java_v1_tribuo_hdbscan.json` |
| Python (v1) | `outputs/python_v1_umap_hdbscan.png` | `outputs/python_v1_umap_hdbscan.json` |

## Cleanup

```bash
make clean       # mvn clean (java/<VERSION>); Python caches (keeps .venv) for that VERSION
make clean-data  # delete outputs/
make clean-venv  # delete python/v1/.venv (and python/v2/.venv if present; also legacy python/.venv)
```
