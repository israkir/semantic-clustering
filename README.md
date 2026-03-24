# semantic-clustering

## Java pipeline (tool-mismatch style)

Clusters **user prompts** with the same defaults as **tool-mismatch-clustering**: **BGE-small-en-v1.5** (ONNX), **Tribuo** density clustering (HDBSCAN*), **Smile UMAP** (PCA fallback), and the same **text normalization** as `application.yaml` there.

- Weights: `model/onnx/bge-small-en-v1.5/` — **`model.onnx` is Git LFS** (`.gitattributes`). After clone: `git lfs install && git lfs pull` or `make git-lfs-pull`.

## Python baseline (comparison only)

**MiniLM** (`all-MiniLM-L6-v2`), **UMAP**, **Python HDBSCAN** — different embeddings and HDBSCAN implementation than Java. Compare with Java via `make cluster-viz`.

## Requirements

- **JDK 17+**, **Maven**, **Make**
- **Git LFS** for the ONNX file
- **Python 3.10+** (Makefile creates `python/.venv` for the baseline script)

## Run

```bash
make              # help
make java-viz     # Java only → outputs/java_tribuo_hdbscan.png + .json
make python-viz   # Python only → outputs/python_umap_hdbscan.png + .json
make cluster-viz  # both pipelines, then open both PNGs if possible
```

Other prompt file:

```bash
PROMPTS_FILE=/path/to/prompts.txt make cluster-viz
```

Interactive matplotlib window for Python only:

```bash
INTERACTIVE=1 make python-viz
```

## Outputs (gitignored under `outputs/`)

| Pipeline | PNG | JSON |
|----------|-----|------|
| Java | `java_tribuo_hdbscan.png` | `java_tribuo_hdbscan.json` |
| Python | `python_umap_hdbscan.png` | `python_umap_hdbscan.json` |

## Cleanup

```bash
make clean       # mvn clean; Python caches (keeps .venv)
make clean-data  # delete outputs/
make clean-venv  # delete python/.venv
```
