# semantic-clustering

Small baseline repo: cluster **user prompts** two ways and skim the plots.

- **Java** — [Tribuo](https://tribuo.org/) 4.3.2, HDBSCAN, hash features → PNG + JSON under `outputs/`.
- **Python** — SentenceTransformers (`all-MiniLM-L6-v2`), UMAP (or a tiny fallback), HDBSCAN → PNG + JSON under `outputs/`.

Prompts live in a text file: **one line per prompt**. Lines starting with `#` and blank lines are ignored. Default file: `data/prompts.txt`.

## What you need

- **Make**, **JDK 17+**, **Maven**
- **Python 3.10+** (the Makefile creates `python/.venv` and installs deps there)

## Run

```bash
make              # same as make help — lists targets with colors
make cluster-viz  # Java + Python, then opens the PNGs if your OS can
```

Use another prompt file:

```bash
PROMPTS_FILE=/path/to/prompts.txt make cluster-viz
```

Show the Python matplotlib window as well:

```bash
INTERACTIVE=1 make python-viz
# or: SHOW=1 make python-viz
```

## Outputs

Everything generated goes to `outputs/` (gitignored): `*_hdbscan.png` and matching `*.json`.

## Cleanup

```bash
make clean       # Maven target + Python caches (venv kept)
make clean-data  # delete outputs/
make clean-venv  # delete python/.venv
```
