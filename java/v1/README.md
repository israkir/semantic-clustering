# Java v1: Clustering Algorithm

This document explains the *clustering algorithm pipeline* used by the Java `v1` implementation:

- Embed each prompt into a dense vector (ONNX Runtime, `bge-small-en-v1.5`)
- Cluster those vectors with **HDBSCAN** (Tribuo)
- Compute 2D coordinates for visualization (UMAP, with PCA as a fallback)

The core idea is: **cluster the embedding space**, and treat 2D projection as a visualization layer rather than part of clustering.

## Pipeline Overview

The entrypoint is `dev.semanticclustering.PromptClusterPipeline`.

1. Load prompts (`data/prompts.txt` by default)
2. Normalize text (configurable cleanup)
3. Create embeddings from the normalized text
4. Run Tribuo **HDBSCAN** on embedding vectors to get cluster labels
5. Compute 2D coordinates for plotting
   - Try **Smile UMAP**
   - If it fails / times out, fall back to a deterministic **2D PCA-like projection**
6. Write PNG chart + JSON containing prompt-to-cluster assignments

## Step 1: Prompt Loading

Prompts are read line-by-line:

- Empty lines are skipped
- Lines starting with `#` are skipped
- Each remaining line becomes a prompt record with:
  - `rawText` = the line content
  - `normalizedText` initially unset (filled during preprocessing)
  - `clusterId` initially unset (filled after HDBSCAN)

## Step 2: Text Normalization (Before Embedding)

Java uses `dev.semanticclustering.processing.TextNormalizer` with `dev.semanticclustering.config.TextNormalizeConfig.defaults()`.

Default behavior:

- `enabled = true`
- `unicodeNfc = true` (apply Unicode NFC normalization)
- `removeControlCharacters = true` (replace control chars with spaces)
- `collapseWhitespace = true` (collapse whitespace runs into a single space)
- `lowercase = false` (no forced lowercase)

If the normalizer returns an empty string (possible but uncommon), that text still proceeds to embedding.

## Step 3: Embeddings (ONNX Runtime)

Embeddings come from `dev.semanticclustering.embedding.OnnxEmbeddingProvider`:

1. Tokenize text with `BgeWordPieceTokenizer` (vocabulary from `vocab.txt`)
2. Run the ONNX model via ONNX Runtime using:
   - `input_ids`
   - `attention_mask`
   - optionally `token_type_ids` (only if the ONNX model exposes it)
3. Extract a fixed-size sentence embedding:
   - If the output includes `sentence_embedding`, use it
   - Else if the output is `hiddenStates`, mean-pool token embeddings using the `attention_mask`
4. L2-normalize the resulting vector

### Important note about normalization

Even though `PromptClusterPipeline` sets `normalizeEmbeddings = true`, ONNX embedding extraction already L2-normalizes vectors. The preprocessing stage may L2-normalize again, but a second L2 normalization is (numerically) idempotent in typical cases.

## Step 4: HDBSCAN Clustering (Tribuo)

Clustering happens in `dev.semanticclustering.processing.HdbscanClusterer`.

### 4.1 Model input to HDBSCAN

The HDBSCAN trainer receives a matrix of shape:

- `num_prompts x embedding_dimensions`

Each embedding dimension becomes a Tribuo dense feature named `embedding_0`, `embedding_1`, ...

### 4.2 HDBSCAN parameters used by `PromptClusterPipeline` (v1 defaults)

Defaults are wired directly in `PromptClusterPipeline`:

- `metric = COSINE`
- `normalizeEmbeddings = true`
- `minClusterSize = 5`
- `neighborCount = 5`
- `numThreads = min(8, availableProcessors)` (but at least 1)
- `neighborQueryStrategy = BRUTE_FORCE`
- projection method string = `"umap"`

The trainer is instantiated as:

- `new HdbscanTrainer(minClusterSize, distanceType, neighborCount, numThreads, queryFactoryType)`

Where:

- `distanceType(COSINE)` maps to Tribuo `DistanceType.COSINE`
- `BRUTE_FORCE` maps to Tribuo `NeighboursQueryFactoryType.BRUTE_FORCE`

### 4.3 Cluster labels and noise

After training:

- Java reads cluster assignments from `model.getClusterLabels()`
- It treats `label <= 0` as **noise**
- It treats `label > 0` as a real cluster

So the JSON key mapping is:

- `noise` for labels `<= 0`
- `cluster_<id>` for labels `> 0`

Outlier scores from Tribuo (`model.getOutlierScores()`) are carried into the per-prompt JSON when available.

### 4.4 High-level HDBSCAN behavior (conceptual)

HDBSCAN is a density-based clustering algorithm that:

- Builds a hierarchy of clusters across density scales
- Selects a “best” flat clustering based on cluster stability
- Assigns points that don’t belong to any stable cluster to noise

This pipeline relies on Tribuo’s implementation details, but the effect matches typical HDBSCAN semantics:

- more stringent parameters (like larger `minClusterSize`) produce fewer, larger clusters
- points that don’t fit a stable dense region become noise

## Step 5: 2D Visualization Coordinates (Not Used For Clustering)

Clustering is performed in the high-dimensional embedding space. The 2D coordinates are computed afterward for plotting.

### 5.1 Projection basis selection

`ClusteringPipeline` computes a 2D PCA-like basis (`ProjectionSpace.fit(...)`) used by the PCA fallback.

It chooses the “definition” vectors for PCA basis fitting as:

- all vectors belonging to labels `> 0` (cluster members), if any exist
- otherwise, all vectors

### 5.2 UMAP attempt with timeout

If `projectionMethod` is `"umap"` and there is at least one vector:

1. Run Smile UMAP (`UmapProjection.project(vectors)`) in a worker thread
2. Enforce a timeout of 180 seconds
3. If UMAP fails, times out, or returns `null`, fall back to PCA coordinates

Smile UMAP options are fixed by code:

- `n_neighbors = 4`
- `n_components = 2`
- `epochs = 100`
- additional UMAP option constants are set directly in `UmapProjection`

### 5.3 PCA fallback mechanics

The PCA fallback (`ProjectionSpace`) is deterministic:

- compute mean origin
- compute covariance matrix
- estimate principal axes via power iteration (and deflation)
- orthogonalize and normalize axes
- project each vector onto the two axes

The result is a `num_prompts x 2` array of coordinates.

## Outputs

The Java pipeline writes:

- `outputs/java_v1_tribuo_hdbscan.png` (by default)
- a sibling JSON file with:
  - `clusters`: `noise` and `cluster_<id>` bucketings of prompt texts
  - `metadata`: algorithm/library versions and key parameters
  - per-prompt cluster assignments embedded under `dataPoints` via `ProcessingResult` (also stored in the `metadata`-enriched JSON via `writeJson` + chart pipeline)

## Files to read in code

- `java/v1/src/main/java/dev/semanticclustering/PromptClusterPipeline.java` (wiring + default parameters)
- `java/v1/src/main/java/dev/semanticclustering/processing/PreprocessingPipeline.java` (normalization + embedding normalization)
- `java/v1/src/main/java/dev/semanticclustering/embedding/OnnxEmbeddingProvider.java` (ONNX embedding extraction)
- `java/v1/src/main/java/dev/semanticclustering/processing/HdbscanClusterer.java` (Tribuo HDBSCAN training)
- `java/v1/src/main/java/dev/semanticclustering/processing/ClusteringPipeline.java` (label handling + UMAP/PCA visualization)
