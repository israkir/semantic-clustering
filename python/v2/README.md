# Python v2: Java v1-compatible pipeline

This version is designed to mirror `java/v1`:

- Prompt loading: one prompt per line; skip empty lines and lines starting with `#`.
- Text normalization: NFC, remove Unicode control chars, collapse whitespace.
- Embeddings: `bge-small-en-v1.5` ONNX via ONNX Runtime + a WordPiece tokenizer matching Java's `BgeWordPieceTokenizer`.
- Clustering: HDBSCAN with cosine distance and Java v1's default parameters (`min_cluster_size=5`, `min_samples=5`, brute-force behavior).
- Visualization:
  - Try UMAP (2D) with a 180s timeout.
  - If UMAP fails/times out, use the deterministic Java PCA-like projection (`ProjectionSpace`).
  - Plot style matches the Java v1 PNG (scatter, noise red, fixed cluster palette, circle markers).

## Run

`make python-viz VERSION=v2`

