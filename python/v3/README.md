# Python v3: BERTopic Cluster Labels

This version uses the `bertopic` library to generate **cluster/topic labels** for every prompt.

Unlike `python/v1` / `python/v2`, this script is optimized for **label generation**, not for matching a specific Java HDBSCAN pipeline.

## Pipeline Overview

1. Load prompts from the provided text file (`--prompts`, default: `data/prompts.txt`)
2. Preprocess each prompt:
   - strip leading list numbering (`"1. "`, `"2. "`, ...)
   - drop comment-like lines starting with `#`
   - remove inline comments after `#`
   - remove structured claim/appointment id-like tokens (e.g. `PH-LAT-9951`)
3. Fit a `BERTopic` model on all prompts
4. Derive labels:
   - each BERTopic topic already has a default human-readable label (`topic_labels_`)
   - prompts assigned to topic `-1` are labeled `"noise"`
5. Save:
   - a PNG scatter plot (2D projection) colored by topic assignment
   - a JSON with:
     - per-prompt topic id + label
     - clusters grouped by topic

## Entry Point

- `python/v3/prompt_cluster.py`

## Run

```bash
make python-viz VERSION=v3
```

Outputs (under `outputs/`):

- `outputs/python_v3_bertopic.png`
- `outputs/python_v3_bertopic.json`

