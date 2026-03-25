"""Semantic-clustering Python v3 (BERTopic labels).

Goal: assign a human-readable *topic/cluster label* to every prompt using BERTopic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

_ID_TOKEN_RE = re.compile(r"\\b[A-Z]{2,3}(?:-[A-Z0-9]+)+\\b")

_TAG = "[PYTHON|BERTopic|v3]"
_ENV_VERSION = "SEMANTIC_CLUSTERING_VERSION"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


_LIST_PREFIX_RE = re.compile(r"^\\s*\\d+\\.\\s*")
_INLINE_COMMENT_RE = re.compile(r"\\s*#.*$")

_STOPWORDS: set[str] | None = None


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            _STOPWORDS = set(ENGLISH_STOP_WORDS)
        except Exception:
            _STOPWORDS = set()
    return _STOPWORDS


def _infer_repo_root(start_file: Path) -> Path:
    """Walk up until `data/prompts.txt` exists."""
    start_dir = start_file if start_file.is_dir() else start_file.parent
    for p in (start_dir, *start_dir.parents):
        if (p / "data" / "prompts.txt").is_file():
            return p
    return start_dir.parent if start_dir.parent is not None else start_dir


def _strip_id_tokens(text: str) -> str:
    """Remove uppercase id-like tokens (to improve topic labels)."""
    # We only strip matches that contain a digit, to avoid removing common uppercase words.
    def _repl(m: re.Match[str]) -> str:
        token = m.group(0)
        return "" if any(ch.isdigit() for ch in token) else token

    return _ID_TOKEN_RE.sub(_repl, text)


def _process_prompt_text(raw_line: str) -> str | None:
    s = raw_line.strip()
    if not s or s.startswith("#"):
        return None

    s = _LIST_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"^subject:\\s*", "", s, flags=re.IGNORECASE)
    s = _INLINE_COMMENT_RE.sub("", s).strip()
    s = _strip_id_tokens(s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s or None


def _load_prompts(prompts_path: Path) -> list[str]:
    prompts: list[str] = []
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        processed = _process_prompt_text(line)
        if processed is not None:
            prompts.append(processed)
    if not prompts:
        raise SystemExit(f"No prompts in file: {prompts_path.resolve()}")
    return prompts


def _truncate_label(s: str, *, max_len: int) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _normalize_word_for_label(w: str) -> str:
    # BERTopic sometimes emits underscores in words/phrases. Make it human-readable.
    return w.replace("_", " ").strip()


def _build_topic_label_map_from_topic_model(
    topic_model: Any,
    topic_ids: list[int],
    *,
    top_fallback_words: int,
) -> dict[int, str]:
    """
    Create `topic_id -> label` mapping from BERTopic topic representations.

    We intentionally avoid `get_topic_info().Name` because it can include
    underscore-joined artifacts (e.g. "0_is_the_for_to") that aren't great labels.
    """
    stopwords = _get_stopwords()
    topic_id_to_label: dict[int, str] = {}

    # BERTopic stores top words per topic in `topic_representations_`.
    reps = getattr(topic_model, "topic_representations_", None) or {}

    def _is_good_candidate_token(tok: str) -> bool:
        t = tok.strip()
        if not t:
            return False
        t_low = t.lower()
        if t_low in stopwords:
            return False
        # Drop trivial tokens.
        if len(t_low) <= 2:
            return False
        # Drop tokens that are mostly numeric or punctuation.
        if not re.search(r"[a-zA-Z0-9]", t):
            return False
        # Drop remaining id-like tokens (belt-and-suspenders).
        if _ID_TOKEN_RE.search(t):
            return False
        return True

    for tid in sorted(set(topic_ids)):
        if tid == -1:
            topic_id_to_label[tid] = "noise"
            continue

        rep_list = reps.get(tid)
        if not rep_list:
            topic_id_to_label[tid] = f"topic_{tid}"
            continue

        candidates: list[str] = []
        for item in rep_list:
            # Expected shape: (word, score)
            if isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
                w = item[0]
            elif isinstance(item, str):
                w = item
            else:
                continue

            w_norm = _normalize_word_for_label(w)
            if not _is_good_candidate_token(w_norm):
                continue
            if w_norm in candidates:
                continue
            candidates.append(w_norm)
            if len(candidates) >= max(1, top_fallback_words):
                break

        if not candidates:
            topic_id_to_label[tid] = f"topic_{tid}"
        else:
            topic_id_to_label[tid] = " / ".join(candidates[:top_fallback_words])

    return topic_id_to_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Prompts text file (default: <repo>/data/prompts.txt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <repo>/outputs/python_v3_bertopic.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive matplotlib window in addition to saving PNG.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=2,
        help="BERTopic `min_topic_size` (smaller => more clusters, fewer outliers).",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=5,
        help="BERTopic `top_n_words` used for topic representation/labeling.",
    )
    parser.add_argument(
        "--ngram-range",
        type=str,
        default="1,2",
        help="BERTopic `n_gram_range` as 'low,high' (default: '1,2').",
    )
    parser.add_argument(
        "--nr-topics",
        type=str,
        default="auto",
        help="BERTopic `nr_topics` (e.g. 'auto' or an integer).",
    )
    parser.add_argument(
        "--top-fallback-words",
        type=int,
        default=5,
        help="If a BERTopic topic has no built-in label, use the first N representation words.",
    )
    parser.add_argument(
        "--legend-max-topics",
        type=int,
        default=50,
        help="Max number of topics to show in the Matplotlib legend (by frequency).",
    )

    args = parser.parse_args()

    repo_root = _infer_repo_root(Path(__file__).resolve())
    version_tag = Path(__file__).resolve().parent.name  # expected: v3
    env_version = (os.getenv(_ENV_VERSION) or "").strip()  # may be set by Makefile
    if env_version and env_version != version_tag:
        _log(f"Version mismatch: env={env_version!r} path={version_tag!r} (using path)")

    prompts_path = args.prompts or (repo_root / "data" / "prompts.txt")
    if not prompts_path.is_file():
        raise SystemExit(f"Prompts file not found: {prompts_path.resolve()}")

    out_path = args.output or (repo_root / "outputs" / "python_v3_bertopic.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")

    ngram_parts = [p.strip() for p in args.ngram_range.split(",")]
    if len(ngram_parts) != 2 or not all(p.isdigit() for p in ngram_parts):
        raise SystemExit(f"Invalid --ngram-range: {args.ngram_range!r} (expected 'low,high').")
    ngram_low, ngram_high = int(ngram_parts[0]), int(ngram_parts[1])

    if args.nr_topics.lower() == "auto":
        nr_topics: int | str | None = "auto"
    else:
        try:
            nr_topics = int(args.nr_topics)
        except ValueError:
            nr_topics = args.nr_topics

    _log(f"Loading prompts from {prompts_path.resolve()}")
    prompts = _load_prompts(prompts_path)
    _log(f"Processed {len(prompts)} prompt(s)")
    _log("Fitting BERTopic (this can take a minute on first run)…")

    # Import BERTopic lazily to keep startup time lower when not running v3.
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    from umap import UMAP

    seed = 42
    n_samples = max(0, len(prompts))
    if n_samples <= 2:
        # UMAP needs n_neighbors >= 2; for tiny inputs PCA is safer.
        umap_model = PCA(n_components=1)
    else:
        n_neighbors = min(15, n_samples - 1)
        n_neighbors = max(2, n_neighbors)
        n_components = min(5, n_samples - 1)
        n_components = max(2, n_components)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )

    # Stopword filtering helps keep topic keywords focused on intents.
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(ngram_low, ngram_high))

    topic_model = BERTopic(
        language="english",
        top_n_words=args.top_n_words,
        n_gram_range=(ngram_low, ngram_high),
        min_topic_size=args.min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        verbose=False,
    )

    _log("Embedding prompts (SentenceTransformer)…")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedding_model.encode(prompts)

    topics, _ = topic_model.fit_transform(prompts, doc_embeddings)
    topic_ids = [int(t) for t in topics]

    topic_id_to_label = _build_topic_label_map_from_topic_model(
        topic_model,
        topic_ids,
        top_fallback_words=args.top_fallback_words,
    )

    # Ensure every assigned topic id has a label.
    for tid in sorted(set(topic_ids)):
        if tid not in topic_id_to_label:
            topic_id_to_label[tid] = "noise" if tid == -1 else f"topic_{tid}"

    prompt_labels = [topic_id_to_label[tid] for tid in topic_ids]

    clusters: dict[str, list[str]] = defaultdict(list)
    for text, tid in zip(prompts, topic_ids):
        if tid == -1:
            clusters["noise"].append(text)
        else:
            clusters[f"topic_{tid}"].append(text)

    topic_labels_by_id = {str(tid): topic_id_to_label[tid] for tid in sorted(topic_id_to_label)}

    # 2D plot: project prompt embeddings and color points by BERTopic topic id.
    try:
        import matplotlib.pyplot as plt

        import numpy as np

        emb_arr = np.asarray(doc_embeddings)
        if emb_arr.ndim != 2 or emb_arr.shape[0] == 0:
            raise ValueError(f"Unexpected embeddings shape: {emb_arr.shape}")

        if emb_arr.shape[0] < 5:
            # Small input fallback: first two embedding dims.
            plot_coords = emb_arr[:, :2].copy()
        else:
            plot_reducer = UMAP(
                n_neighbors=4,
                n_components=2,
                metric="cosine",
                random_state=seed,
                low_memory=True,
            )
            plot_coords = plot_reducer.fit_transform(emb_arr)

        # Extra width helps when we place a legend outside on the right.
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_title("BERTopic topic clustering (2D projection)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        unique_topics = sorted(set(topic_ids))
        non_noise_topics = [t for t in unique_topics if t != -1]

        topic_counts: dict[int, int] = defaultdict(int)
        for tid in topic_ids:
            topic_counts[tid] += 1

        cmap = plt.get_cmap("tab20")
        topic_to_color: dict[int, Any] = {}
        for i, tid in enumerate(non_noise_topics):
            topic_to_color[tid] = cmap(i % 20)
        topic_to_color[-1] = (1.0, 0.0, 0.0)  # red for noise/outliers

        # Only include up to N topics in the legend to prevent unreadable plots.
        legend_non_noise = sorted(non_noise_topics, key=lambda t: topic_counts[t], reverse=True)
        if len(legend_non_noise) > args.legend_max_topics:
            legend_non_noise = legend_non_noise[: args.legend_max_topics]

        legend_topics = set(legend_non_noise)
        if -1 in unique_topics:
            legend_topics.add(-1)

        for tid in unique_topics:
            mask = np.asarray(topic_ids) == tid
            xs = plot_coords[mask, 0]
            ys = plot_coords[mask, 1]

            label = topic_id_to_label.get(tid, f"topic_{tid}")
            if tid == -1:
                legend_label = "noise (-1)"
            else:
                legend_label = f"{_truncate_label(label, max_len=28)} (t{tid})"

            scatter_label = legend_label if tid in legend_topics else "_nolegend_"

            ax.scatter(
                xs,
                ys,
                s=60,
                alpha=0.85,
                color=topic_to_color.get(tid, "#999999"),
                edgecolor="k",
                linewidths=0.3,
                label=scatter_label,
            )

        if legend_topics:
            # Put legend outside the axes to keep it readable.
            ax.legend(
                title="Topic label",
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
            )

        # Reserve a slice on the right for the external legend.
        fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

        if args.show:
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        _log(f"Warning: PNG generation failed ({e}); continuing with JSON only.")

    payload: dict[str, Any] = {
        "program": "python-bertopic-topic-labels",
        "prompts_file": str(prompts_path.resolve()),
        "output_image": str(out_path.resolve()),
        "output_json": str(json_path.resolve()),
        "prompts": prompts,
        "prompt_assignments": [
            {"topic_id": tid, "topic_label": lbl} for tid, lbl in zip(topic_ids, prompt_labels)
        ],
        "clusters": dict(clusters),
        "topic_labels_by_id": topic_labels_by_id,
        "metadata": {
            "algorithm": "bertopic",
            "library": "python",
            "version_tag": version_tag,
            "min_topic_size": args.min_topic_size,
            "top_n_words": args.top_n_words,
            "n_gram_range": [ngram_low, ngram_high],
            "nr_topics": nr_topics,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _log(f"Wrote PNG  {out_path.resolve()}")
    _log(f"Wrote JSON {json_path.resolve()}")
    _log("Done.")


if __name__ == "__main__":
    main()

