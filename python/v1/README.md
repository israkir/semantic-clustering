# Python v1: Clustering Algorithm

This document explains the *clustering algorithm pipeline* used by the Python `v1` implementation:

- Embed each prompt into a dense vector (SentenceTransformers `all-MiniLM-L6-v2`)
- Cluster those vectors with **HDBSCAN**
- Compute 2D coordinates for visualization (UMAP, with a small-vector fallback)

Like the Java implementation, the design is:

- **cluster in embedding space**
- **project only for visualization**

## Pipeline Overview

The entrypoint is `python/v1/prompt_cluster.py`.

1. Load prompts from a text file (default: `data/prompts.txt`)
2. Preprocess each line (strip list numbering, strip inline comments)
3. Create embeddings with SentenceTransformers
4. Compute 2D coordinates:
   - if `len(prompts) < 5`: use the first two embedding dimensions
   - else: use UMAP (2D)
5. Run HDBSCAN clustering on the *original embedding vectors*
6. Save:
   - a PNG scatter plot
   - a JSON file that groups prompt texts by cluster label (noise included)

## Step 1: Prompt Loading & Text Preprocessing

`_load_prompts()` reads the prompt file line-by-line:

- Empty lines are skipped
- Lines starting with `#` are skipped
- Each non-skipped line is passed through `_process_prompt_text()`

### `_process_prompt_text()` rules

- Strip leading list numbering with the regex `^\\d+\\.\\s*`  
  (e.g. `1. Do X` → `Do X`)
- If the line contains `#`, remove everything after the first `#`  
  (inline comments are removed)
- Collapse whitespace runs to a single space

If the result is empty after processing, the prompt is dropped.

If there are no prompts after preprocessing, the script exits with an error.

## Step 2: Embeddings (SentenceTransformers)

Embeddings are produced by:

- `SentenceTransformer("all-MiniLM-L6-v2")`
- `model.encode(data)` where `data` is the list of processed prompts

The resulting `embeddings` array is used for:

- clustering
- UMAP projection (if `len(prompts) >= 5`)

## Step 3: 2D Projection for Plotting (UMAP or Fallback)

The plotting coordinates come from UMAP *only* when there are enough points.

### 3.1 Small input fallback

If `len(data) < 5`:

- skip UMAP
- set:
  - `X = embeddings[:, 0]`
  - `Y = embeddings[:, 1]`

### 3.2 UMAP configuration (when `len(data) >= 5`)

UMAP is run with:

- `n_neighbors = 4`
- `n_components = 2`
- `metric = "cosine"`
- `random_state = 42`

The script uses:

- `umap_reducer.fit_transform(embeddings)` to compute 2D points

## Step 4: HDBSCAN Clustering

Clustering is performed by:

- `hdbscan.HDBSCAN(...)`
- `clusterer.fit_predict(embeddings)`

The parameters used in `v1` defaults:

- `min_cluster_size = 2`
- `min_samples = 2`
- `algorithm = "boruvka_kdtree"`

### Cluster label semantics

HDBSCAN’s typical output is used:

- `label = -1` means **noise**
- `label >= 0` means a cluster id

The JSON structure uses:

- `noise` bucket for `-1`
- `cluster_<label>` bucket for cluster ids `>= 0`

## Step 5: Outputs

### PNG plot

The script creates a Seaborn scatter plot where:

- points are colored by `Cluster`
- noise (`-1`) is colored red

### JSON file

The JSON output includes:

- `program`: `"python-umap-hdbscan"`
- `prompts_file`: absolute path to the prompts input file
- `output_image`: path to the saved PNG
- `output_json`: path to the JSON output file
- `clusters`: mapping of:
  - `"noise"` → list of prompt texts with label `-1`
  - `"cluster_<id>"` → list of prompt texts in cluster `<id>`

## Files to read in code

- `python/v1/prompt_cluster.py` (the full algorithm end-to-end)
