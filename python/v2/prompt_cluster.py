"""Semantic-clustering Python v2.

This is intended to match the Java v1 algorithm and chart styling:
- Java v1 prompt loading (skip empty and leading '#'; otherwise keep the line)
- Java v1 text normalization (NFC, collapse whitespace, remove Unicode control chars)
- BGE-small-en-v1.5 ONNX embeddings via a Java-matching WordPiece tokenizer
- HDBSCAN clustering with Java-matching parameters (cosine, min_cluster_size=5, neighbor_count=5, brute-force)
  and Java label semantics remapped to: noise=0, clusters>=1
- Smile UMAP (n_neighbors=4, n_components=2, epochs=100) with a 180s timeout;
  if UMAP fails/times out, fall back to the deterministic Java PCA-like projection
- Java v1 visualization style: PNG scatter, title/axes, noise red, fixed cluster palette, circle markers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap

import onnxruntime as ort


_TAG = "[PYTHON|BGE+HDBSCAN]"
_ENV_VERSION = "SEMANTIC_CLUSTERING_VERSION"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


def _infer_repo_root(start_file: Path) -> Path:
    """Walk up until `data/prompts.txt` exists (matches python/v1 behavior)."""

    start_dir = start_file if start_file.is_dir() else start_file.parent
    for p in (start_dir, *start_dir.parents):
        if (p / "data" / "prompts.txt").is_file():
            return p
    return start_dir.parent if start_dir.parent is not None else start_dir


def _load_prompts_java_v1(prompts_path: Path) -> list[str]:
    """Java v1 prompt loading: line.strip(); skip empty and lines starting with '#'. """

    lines = prompts_path.read_text(encoding="utf-8").splitlines()
    prompts: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        prompts.append(s)
    if not prompts:
        raise SystemExit(f"No prompts in file: {prompts_path.resolve()}")
    return prompts


@dataclass(frozen=True)
class TextNormalizeConfig:
    enabled: bool = True
    lowercase: bool = False
    collapse_whitespace: bool = True
    unicode_nfc: bool = True
    remove_control_characters: bool = True


class TextNormalizerJavaV1:
    def __init__(self, config: TextNormalizeConfig) -> None:
        self._config = config

    def normalize(self, text: str | None) -> str:
        if text is None:
            return ""
        if not self._config.enabled:
            return text

        normalized = text.strip()
        if self._config.unicode_nfc:
            normalized = unicodedata.normalize("NFC", normalized)

        if self._config.remove_control_characters:
            # Java: normalized.replaceAll("\\p{Cc}+", " ")
            # Python doesn't support \\p{Cc} in re without extra regex module,
            # so we scan by Unicode general category.
            out_chars: list[str] = []
            in_run = False
            for ch in normalized:
                if unicodedata.category(ch) == "Cc":
                    if not in_run:
                        out_chars.append(" ")
                    in_run = True
                else:
                    out_chars.append(ch)
                    in_run = False
            normalized = "".join(out_chars)

        if self._config.collapse_whitespace:
            normalized = re.sub(r"\s+", " ", normalized)

        if self._config.lowercase:
            normalized = normalized.lower()

        return normalized


class BgeWordPieceTokenizerJavaV1:
    CLS = "[CLS]"
    SEP = "[SEP]"
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab_path: Path, max_sequence_length: int) -> None:
        self._vocab = self._load_vocabulary(vocab_path)
        self._max_sequence_length = max_sequence_length

        for token in (self.CLS, self.SEP, self.PAD, self.UNK):
            if token not in self._vocab:
                raise ValueError(f"Tokenizer vocabulary missing token: {token}")

    def encode(self, text: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        basic_tokens = self._basic_tokenize(text or "")
        pieces: list[str] = []
        for token in basic_tokens:
            pieces.extend(self._word_piece_tokenize(token))

        max_content = self._max_sequence_length - 2
        if len(pieces) > max_content:
            pieces = pieces[:max_content]

        tokens = [self.CLS, *pieces, self.SEP]

        input_ids = np.zeros((self._max_sequence_length,), dtype=np.int64)
        attention_mask = np.zeros((self._max_sequence_length,), dtype=np.int64)
        token_type_ids = np.zeros((self._max_sequence_length,), dtype=np.int64)

        cursor = 0
        for tok in tokens:
            input_ids[cursor] = self._vocab_id(tok)
            attention_mask[cursor] = 1
            cursor += 1

        pad_id = self._vocab_id(self.PAD)
        for cursor in range(cursor, self._max_sequence_length):
            input_ids[cursor] = pad_id

        return input_ids, attention_mask, token_type_ids

    def _basic_tokenize(self, text: str) -> list[str]:
        # Java: Normalizer.normalize(text, Form.NFKC).toLowerCase(Locale.ROOT)
        #       .replaceAll("\\s+", " ").trim()
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()

        tokens: list[str] = []
        current: list[str] = []

        for ch in normalized:
            if ch.isalnum():  # Java: Character.isLetterOrDigit
                current.append(ch)
                continue

            if current:
                tokens.append("".join(current))
                current = []

            if not ch.isspace():  # Java: !Character.isWhitespace(character)
                tokens.append(ch)

        if current:
            tokens.append("".join(current))

        return tokens

    def _word_piece_tokenize(self, token: str) -> list[str]:
        if token in self._vocab:
            return [token]

        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            current: str | None = None
            while start < end:
                candidate = token[start:end]
                if start > 0:
                    candidate = f"##{candidate}"
                if candidate in self._vocab:
                    current = candidate
                    break
                end -= 1

            if current is None:
                return [self.UNK]

            pieces.append(current)
            start = end

        return pieces

    def _vocab_id(self, token: str) -> int:
        return int(self._vocab.get(token, self._vocab[self.UNK]))

    @staticmethod
    def _load_vocabulary(path: Path) -> dict[str, int]:
        vocab: dict[str, int] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            vocab[line.strip()] = i
        return dict(vocab)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float64, copy=True)
    norm = float(np.sqrt(np.sum(vec * vec)))
    if norm == 0.0:
        return vec
    vec /= norm
    return vec


class OnnxEmbeddingProviderJavaV1:
    def __init__(
        self,
        *,
        onnx_model_path: Path,
        vocab_path: Path,
        max_sequence_length: int,
        model_name: str,
    ) -> None:
        self._environment = ort.get_available_providers()
        self._session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
        self._tokenizer = BgeWordPieceTokenizerJavaV1(vocab_path=vocab_path, max_sequence_length=max_sequence_length)
        self._model_name = model_name

        input_names = {i.name for i in self._session.get_inputs()}
        if "input_ids" not in input_names or "attention_mask" not in input_names:
            raise ValueError("ONNX model must expose input_ids and attention_mask inputs")
        self._has_token_type_ids = "token_type_ids" in input_names

        output_names = [o.name for o in self._session.get_outputs()]
        self._has_sentence_embedding = "sentence_embedding" in output_names

    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for text in texts:
            input_ids, attention_mask, token_type_ids = self._tokenizer.encode(text)
            vector = self._run_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            embeddings.append(vector)
        return embeddings

    def _run_inference(self, *, input_ids: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray) -> np.ndarray:
        inputs: dict[str, Any] = {
            "input_ids": input_ids.reshape(1, -1),
            "attention_mask": attention_mask.reshape(1, -1),
        }
        if self._has_token_type_ids:
            inputs["token_type_ids"] = token_type_ids.reshape(1, -1)

        if self._has_sentence_embedding:
            out = self._session.run(["sentence_embedding"], inputs)
            arr = out[0]
            if arr.ndim != 2 or arr.shape[0] < 1:
                raise ValueError("Unexpected sentence_embedding shape")
            vec = arr[0].astype(np.float64, copy=False)
        else:
            # Java uses result.get(0) when sentence_embedding isn't present.
            outs = self._session.run(None, inputs)
            arr = outs[0]
            if arr.ndim == 3:
                # Java: meanPool(hiddenStates, attentionMask) where hiddenStates[0] is [seq, hidden]
                hidden_states = arr[0]  # [seq, hidden]
                seq_len = min(hidden_states.shape[0], attention_mask.shape[0])
                hidden = hidden_states[:seq_len].astype(np.float64, copy=False)
                mask = attention_mask[:seq_len].astype(np.float64, copy=False)
                token_count = float(np.sum(mask))
                if token_count > 0.0:
                    pooled = np.sum(hidden * mask[:, None], axis=0) / token_count
                else:
                    pooled = np.zeros((hidden.shape[1],), dtype=np.float64)
                vec = pooled
            elif arr.ndim == 2:
                if arr.shape[0] < 1:
                    raise ValueError("Unexpected pooled output shape")
                vec = arr[0].astype(np.float64, copy=False)
            else:
                raise ValueError(f"Unexpected ONNX output rank: {arr.ndim}")

        return _l2_normalize(vec)


def _resolve_onnx_paths(repo_root: Path) -> tuple[Path, Path]:
    env_onx = os.getenv("SEMANTIC_CLUSTERING_ONNX_MODEL")
    env_vocab = os.getenv("SEMANTIC_CLUSTERING_ONNX_VOCAB")
    if env_onx and env_vocab:
        return Path(env_onx), Path(env_vocab)

    base = repo_root / "model" / "onnx" / "bge-small-en-v1.5"
    return base / "model.onnx", base / "vocab.txt"


class ProjectionSpaceJavaV1:
    EPSILON = 1.0e-12

    def __init__(self, *, origin: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> None:
        self._origin = origin
        self._axis_x = axis_x
        self._axis_y = axis_y

    @staticmethod
    def fit(vectors: np.ndarray) -> "ProjectionSpaceJavaV1":
        if vectors is None or vectors.shape[0] == 0:
            origin = np.zeros((0,), dtype=np.float64)
            return ProjectionSpaceJavaV1(origin=origin, axis_x=origin.copy(), axis_y=origin.copy())

        dimensions = int(vectors.shape[1])
        origin = np.mean(vectors, axis=0).astype(np.float64, copy=False)
        if dimensions == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return ProjectionSpaceJavaV1(origin=origin, axis_x=empty, axis_y=empty)

        covariance = ProjectionSpaceJavaV1._covariance(vectors, origin)
        axis_x = ProjectionSpaceJavaV1._principal_axis(covariance)
        deflated = ProjectionSpaceJavaV1._deflate(covariance, axis_x)
        axis_y = ProjectionSpaceJavaV1._principal_axis(deflated)

        if ProjectionSpaceJavaV1._norm(axis_x) <= ProjectionSpaceJavaV1.EPSILON:
            axis_x = ProjectionSpaceJavaV1._canonical_axis(dimensions, 0)
        axis_x = _l2_normalize(axis_x)

        axis_y = ProjectionSpaceJavaV1._orthogonalize(axis_y, axis_x)
        if ProjectionSpaceJavaV1._norm(axis_y) <= ProjectionSpaceJavaV1.EPSILON:
            axis_y = ProjectionSpaceJavaV1._fallback_axis(dimensions, axis_x)
        axis_y = _l2_normalize(axis_y)

        return ProjectionSpaceJavaV1(origin=origin, axis_x=axis_x, axis_y=axis_y)

    @staticmethod
    def from_basis(origin: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> "ProjectionSpaceJavaV1":
        return ProjectionSpaceJavaV1(origin=origin, axis_x=axis_x, axis_y=axis_y)

    def project(self, vector: np.ndarray) -> tuple[float, float]:
        if self._origin.shape[0] == 0:
            return 0.0, 0.0
        centered = vector.astype(np.float64, copy=False) - self._origin
        x = float(np.dot(centered, self._axis_x))
        y = float(np.dot(centered, self._axis_y))
        return x, y

    @staticmethod
    def _covariance(vectors: np.ndarray, origin: np.ndarray) -> np.ndarray:
        dimensions = int(origin.shape[0])
        covariance = np.zeros((dimensions, dimensions), dtype=np.float64)
        for v in vectors:
            centered = v.astype(np.float64, copy=False) - origin
            covariance += np.outer(centered, centered)

        n = vectors.shape[0]
        scale = 1.0 / (n - 1) if n > 1 else 1.0
        covariance *= scale
        return covariance

    @staticmethod
    def _principal_axis(matrix: np.ndarray) -> np.ndarray:
        dimensions = int(matrix.shape[0])
        if dimensions == 0:
            return np.zeros((0,), dtype=np.float64)

        vector = np.full((dimensions,), 1.0 / np.sqrt(dimensions), dtype=np.float64)
        for _ in range(32):
            next_vec = matrix @ vector
            n = ProjectionSpaceJavaV1._norm(next_vec)
            if n <= ProjectionSpaceJavaV1.EPSILON:
                return np.zeros((dimensions,), dtype=np.float64)
            next_vec /= n
            vector = next_vec
        return vector

    @staticmethod
    def _deflate(matrix: np.ndarray, axis: np.ndarray) -> np.ndarray:
        # Java: eigenvalue = dot(axis, multiply(matrix, axis))
        eigenvalue = float(axis @ (matrix @ axis))
        return matrix - eigenvalue * np.outer(axis, axis)

    @staticmethod
    def _orthogonalize(candidate: np.ndarray, axis: np.ndarray) -> np.ndarray:
        if candidate.shape[0] == 0:
            return candidate
        projection = float(np.dot(candidate, axis))
        return candidate - axis * projection

    @staticmethod
    def _fallback_axis(dimensions: int, axis_x: np.ndarray) -> np.ndarray:
        for i in range(dimensions):
            candidate = ProjectionSpaceJavaV1._canonical_axis(dimensions, i)
            orthogonal = ProjectionSpaceJavaV1._orthogonalize(candidate, axis_x)
            if ProjectionSpaceJavaV1._norm(orthogonal) > ProjectionSpaceJavaV1.EPSILON:
                return orthogonal
        return np.zeros((dimensions,), dtype=np.float64)

    @staticmethod
    def _canonical_axis(dimensions: int, index: int) -> np.ndarray:
        axis = np.zeros((dimensions,), dtype=np.float64)
        if dimensions > 0:
            axis[min(index, dimensions - 1)] = 1.0
        return axis

    @staticmethod
    def _norm(vector: np.ndarray) -> float:
        return float(np.sqrt(np.dot(vector, vector)))


def _run_umap_with_timeout(vectors: np.ndarray, timeout_seconds: int) -> np.ndarray | None:
    """Run umap in a daemon thread; return coords or None on timeout/failure."""

    container: dict[str, Any] = {"coords": None, "error": None}

    def worker() -> None:
        try:
            # Java: Smile UMAP options:
            # n_neighbors=4, n_components=2, epochs=100, initialAlpha=1.0, minDist=0.1,
            # spread=1.0, localConnectivity=5, setOpMixRatio=1.0, repulsionStrength=1.0
            # Java doesn't explicitly set distance metric, so we keep umap-learn's default (euclidean).
            reducer = umap.UMAP(
                n_neighbors=4,
                n_components=2,
                n_epochs=100,
                learning_rate=1.0,
                min_dist=0.1,
                spread=1.0,
                local_connectivity=5,
                set_op_mix_ratio=1.0,
                repulsion_strength=1.0,
                algorithm="auto",
            )
            container["coords"] = reducer.fit_transform(vectors)
        except Exception as e:  # pragma: no cover - we can't reliably exercise failure in tests
            container["error"] = e

    t = Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        return None
    coords = container.get("coords")
    if coords is None:
        return None
    return np.asarray(coords, dtype=np.float64)


def _cluster_json_key(tribuo_label: int) -> str:
    if tribuo_label <= 0:
        return "noise"
    return f"cluster_{tribuo_label}"


def _write_clustering_json_java_v1(
    *,
    program: str,
    prompts_path: Path,
    image_out: Path,
    json_out: Path,
    raw_texts: list[str],
    normalized_texts: list[str],
    labels_shifted: list[int],
    metadata: dict[str, str],
) -> None:
    buckets: dict[int, list[str]] = defaultdict(list)
    for text_raw, text_norm, lab in zip(raw_texts, normalized_texts, labels_shifted):
        buckets[int(lab)].append(text_norm if text_norm is not None else text_raw)

    order = sorted(buckets.keys())
    clusters: dict[str, list[str]] = {}
    for lab in order:
        clusters[_cluster_json_key(lab)] = buckets[lab]

    payload = {
        "program": program,
        "prompts_file": str(prompts_path.resolve()),
        "output_image": str(image_out.resolve()),
        "output_json": str(json_out.resolve()),
        "clusters": clusters,
        "metadata": metadata,
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _make_java_v1_plot(
    *,
    out_path: Path,
    labels_shifted: np.ndarray,
    plot_coords: np.ndarray,
) -> None:
    # Java: width=960 height=640 title "Prompt Intent Clustering" axis titles "X","Y"
    # Java: default scatter series; legend visible; marker size=10; circle markers.
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)

    ax.set_title("Prompt Intent Clustering")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    colors: deque[Any] = deque(
        [
            (0.0, 1.0, 1.0),  # CYAN
            (0.0, 0.5019607843137255, 0.0),  # GREEN (0,128,0)
            (1.0, 0.0, 1.0),  # MAGENTA
            (1.0, 0.6470588235294118, 0.0),  # ORANGE
            (0.0, 0.0, 1.0),  # BLUE
            (0.4, 0.7, 0.2),  # custom
            (0.6, 0.2, 0.8),  # custom
        ]
    )

    unique_labels = sorted({int(l) for l in labels_shifted.tolist()})
    for cluster_id in unique_labels:
        mask = labels_shifted == cluster_id
        xs = plot_coords[mask, 0]
        ys = plot_coords[mask, 1]
        if cluster_id <= 0:
            color = (1.0, 0.0, 0.0)  # RED
            series_name = f"noise ({cluster_id})"
        else:
            color = colors.popleft() if colors else (0.25, 0.25, 0.25)  # dark gray
            series_name = f"cluster {cluster_id}"

        ax.scatter(xs, ys, s=100, marker="o", color=color, label=series_name, edgecolors="none")

    ax.legend()
    plt.savefig(out_path, dpi=80, bbox_inches=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Prompts file; Java v1 format: one prompt per line, skip empty lines and lines starting with '#'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <repo>/outputs/python_v2_umap_hdbscan.png).",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive matplotlib window.")
    args = parser.parse_args()

    repo_root = _infer_repo_root(Path(__file__).resolve())
    version_tag = Path(__file__).resolve().parent.name  # expected: v2
    env_version = (os.getenv(_ENV_VERSION) or "").strip()
    if env_version and env_version != version_tag:
        _log(f"Version mismatch: env={env_version!r} path={version_tag!r} (using path)")

    onnx_path, vocab_path = _resolve_onnx_paths(repo_root)
    if not onnx_path.is_file() or not vocab_path.is_file():
        raise SystemExit(f"ONNX model or vocab not found: {onnx_path} / {vocab_path}")

    prompts_path = args.prompts or (repo_root / "data" / "prompts.txt")
    if not prompts_path.is_file():
        raise SystemExit(f"Prompts file not found: {prompts_path.resolve()}")

    out_path = args.output or (repo_root / "outputs" / f"python_{version_tag}_umap_hdbscan.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")

    bar = "=" * 76
    print(bar, flush=True)
    print("  PYTHON — BGE-small ONNX · HDBSCAN · UMAP (Java v1-compatible projection)", flush=True)
    print(f"  Version: {version_tag}", flush=True)
    print("  Noise: cluster id 0", flush=True)
    print(bar, flush=True)

    raw_prompts = _load_prompts_java_v1(prompts_path)

    normalizer = TextNormalizerJavaV1(
        TextNormalizeConfig(
            enabled=True,
            lowercase=False,
            collapse_whitespace=True,
            unicode_nfc=True,
            remove_control_characters=True,
        )
    )

    normalized_texts: list[str] = [normalizer.normalize(s) for s in raw_prompts]

    max_seq_len = 512
    model_label = "bge-small-en-v1.5"

    threads = max(1, min(8, int(os.cpu_count() or 1)))
    min_cluster = 5
    neighbor_count = min_cluster

    started = time.perf_counter()
    _log(f"Embedding {len(raw_prompts)} prompt(s) via ONNX: {onnx_path}")
    provider = OnnxEmbeddingProviderJavaV1(
        onnx_model_path=onnx_path,
        vocab_path=vocab_path,
        max_sequence_length=max_seq_len,
        model_name=model_label,
    )
    embeddings_raw = provider.embed(normalized_texts)

    # Java: OnnxEmbeddingProvider already L2-normalizes, then PreprocessingPipeline L2-normalizes again.
    vectors = np.stack(embeddings_raw, axis=0).astype(np.float64, copy=False)
    vectors = np.stack([_l2_normalize(v) for v in vectors], axis=0)

    # HDBSCAN parameters (Java v1 slotParameters defaults)
    _log("Clustering with HDBSCAN (cosine, min_cluster_size=5, min_samples=5, brute-force)")
    if vectors.shape[0] == 0:
        labels_shifted_list: list[int] = []
        plot_coords = np.zeros((0, 2), dtype=np.float64)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=neighbor_count,
            metric="cosine",
            algorithm="generic",  # approximate BRUTE_FORCE behavior
            core_dist_n_jobs=threads,
        )
        labels = clusterer.fit_predict(vectors)  # noise=-1, clusters=0..K-1
        labels_shifted = np.where(labels == -1, 0, labels + 1).astype(int)
        labels_shifted_list = labels_shifted.tolist()

        # PCA fallback basis is fit on clustered vectors only (labels > 0), else all.
        clustered_vectors = vectors[labels_shifted > 0]
        projection_vectors = clustered_vectors if clustered_vectors.shape[0] > 0 else vectors
        projection_basis = ProjectionSpaceJavaV1.fit(projection_vectors)

        umap_coords = _run_umap_with_timeout(vectors, timeout_seconds=180)
        if umap_coords is None:
            _log("UMAP failed/timed out; falling back to deterministic PCA-like projection")
            plot_coords = np.zeros((vectors.shape[0], 2), dtype=np.float64)
            for i in range(vectors.shape[0]):
                x, y = projection_basis.project(vectors[i])
                plot_coords[i, 0] = x
                plot_coords[i, 1] = y
        else:
            plot_coords = umap_coords

    _make_java_v1_plot(out_path=out_path, labels_shifted=np.asarray(labels_shifted_list), plot_coords=plot_coords)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    duration_ms = int((time.perf_counter() - started) * 1000)

    # Java-like metadata keys (best effort in Python v2)
    try:
        import importlib.metadata as importlib_metadata

        hdbscan_version = importlib_metadata.version("hdbscan")
        umap_version = importlib_metadata.version("umap-learn")
        onnxruntime_version = importlib_metadata.version("onnxruntime")
    except Exception:
        hdbscan_version = "unknown"
        umap_version = "unknown"
        onnxruntime_version = "unknown"

    metadata: dict[str, str] = {
        "algorithm": "hdbscan",
        "library": "python",
        "tribuo_version": hdbscan_version,
        "metric": "cosine",
        "min_cluster_size": str(min_cluster),
        "neighbor_count": str(neighbor_count),
        "num_threads": str(threads),
        "neighbor_query_strategy": "brute_force",
        "normalize_embeddings": "true",
        "embedding_model": provider.model_name(),
        "embedding_dimensions": str(vectors.shape[1] if vectors.ndim == 2 else 0),
        "projection_method": "umap",
        "processing_duration_ms": str(duration_ms),
        "umap_learn_version": umap_version,
        "onnxruntime_version": onnxruntime_version,
    }

    _write_clustering_json_java_v1(
        program="semantic-clustering",
        prompts_path=prompts_path,
        image_out=out_path,
        json_out=json_path,
        raw_texts=raw_prompts,
        normalized_texts=normalized_texts,
        labels_shifted=labels_shifted_list,
        metadata=metadata,
    )

    _log(f"Wrote image  {out_path.resolve()}")
    _log(f"Wrote JSON   {json_path.resolve()}")
    _log("Done.")


if __name__ == "__main__":
    main()

