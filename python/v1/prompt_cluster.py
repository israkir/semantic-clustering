"""Embed prompts, UMAP 2D for plotting, HDBSCAN on embeddings (baseline script)."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

_TAG = "[PYTHON|MiniLM+UMAP]"
_ENV_VERSION = "SEMANTIC_CLUSTERING_VERSION"


def _log(msg: str) -> None:
    print(f"{_TAG} {msg}", flush=True)


print(
    f"{_TAG} Importing scientific stack (first run may take a bit)…",
    flush=True,
)

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sentence_transformers import SentenceTransformer


_LIST_PREFIX = re.compile(r"^\d+\.\s*")


def _process_prompt_text(s: str) -> str | None:
    """Strip list numbering and inline comments; keep digits (times, refs, etc.)."""
    t = _LIST_PREFIX.sub("", s).strip()
    if "#" in t:
        t = t.split("#", 1)[0].strip()
    if not t:
        return None
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def _load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    prompts: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        processed = _process_prompt_text(s)
        if processed is not None:
            prompts.append(processed)
    if not prompts:
        raise SystemExit(f"No prompts in file: {path.resolve()}")
    return prompts


def _cluster_json_key_hdbscan(label: int) -> str:
    if label == -1:
        return "noise"
    return f"cluster_{label}"


def _write_clustering_json(
    *,
    prompts_path: Path,
    image_out: Path,
    json_out: Path,
    texts: list[str],
    labels: list[int],
) -> None:
    buckets: dict[int, list[str]] = defaultdict(list)
    for text, lab in zip(texts, labels):
        buckets[int(lab)].append(text)

    key_order: list[int] = []
    if -1 in buckets:
        key_order.append(-1)
    key_order.extend(sorted(k for k in buckets if k != -1))

    clusters = {
        _cluster_json_key_hdbscan(k): buckets[k] for k in key_order
    }
    payload = {
        "program": "python-umap-hdbscan",
        "prompts_file": str(prompts_path.resolve()),
        "output_image": str(image_out.resolve()),
        "output_json": str(json_out.resolve()),
        "clusters": clusters,
    }
    json_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _infer_repo_root(start_file: Path) -> Path:
    """
    Find the repository root by walking upwards until we see `data/prompts.txt`.

    This keeps the script working after moving into `python/v1/` (and later `python/v2/`).
    """
    start_dir = start_file if start_file.is_dir() else start_file.parent
    for p in (start_dir, *start_dir.parents):
        if (p / "data" / "prompts.txt").is_file():
            return p
    # Fallback: keep a sensible relative structure.
    return start_dir.parent if start_dir.parent is not None else start_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Text file with one prompt per line: # comments and leading list numbers (e.g. '1. ') are stripped; body text is unchanged (default: <repo>/data/prompts.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path for the scatter plot (default: <repo>/outputs/python_v1_umap_hdbscan.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display an interactive window (in addition to saving PNG)",
    )
    args = parser.parse_args()

    repo_root = _infer_repo_root(Path(__file__).resolve())
    version_tag = Path(__file__).resolve().parent.name  # expected: v1
    env_version = (os.getenv(_ENV_VERSION) or "").strip()  # may be set by Makefile
    if env_version and env_version != version_tag:
        _log(f"Version mismatch: env={env_version!r} path={version_tag!r} (using path)")
    prompts_path = args.prompts or (repo_root / "data" / "prompts.txt")
    if not prompts_path.is_file():
        raise SystemExit(f"Prompts file not found: {prompts_path.resolve()}")
    out_path = args.output or (repo_root / "outputs" / f"python_{version_tag}_umap_hdbscan.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bar = "=" * 76
    print(bar, flush=True)
    print(
        "  PYTHON — SentenceTransformers (MiniLM) · UMAP 2D · HDBSCAN on embeddings",
        flush=True,
    )
    print(f"  Version: {version_tag}", flush=True)
    print("  Noise points: cluster id -1", flush=True)
    print(bar, flush=True)

    data = _load_prompts(prompts_path)
    _log(f"Prompts file: {prompts_path.resolve()}")
    _log(f"Using {len(data)} processed prompt(s)")
    _log(f"Output PNG: {out_path.resolve()}")
    _log("Generating embeddings (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(data)

    if len(data) < 5:
        _log("Few prompts — skipping UMAP; using first two embedding dimensions for X/Y.")
        umap_embeddings = embeddings[:, :2].copy()
    else:
        _log("Reducing to 2D with UMAP (cosine, n_neighbors=4)…")
        umap_reducer = umap.UMAP(
            n_neighbors=4, n_components=2, metric="cosine", random_state=42
        )
        umap_embeddings = umap_reducer.fit_transform(embeddings)

    _log("Clustering embedding vectors with HDBSCAN (boruvka_kdtree, min_cluster_size=2)…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        algorithm="boruvka_kdtree",
    )
    labels = clusterer.fit_predict(embeddings)
    label_list = labels.tolist()

    df = pd.DataFrame(
        {
            "Text": data,
            "X": umap_embeddings[:, 0],
            "Y": umap_embeddings[:, 1],
            "Cluster": labels,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_clusters = sorted(df["Cluster"].unique())
    standard_colors = sns.color_palette("tab20", len(unique_clusters))
    custom_palette = {}
    color_idx = 0

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            custom_palette[cluster_id] = "red"
        else:
            custom_palette[cluster_id] = standard_colors[color_idx]
            color_idx += 1

    sns.scatterplot(
        data=df,
        x="X",
        y="Y",
        hue="Cluster",
        palette=custom_palette,
        s=100,
        alpha=0.8,
        edgecolor="k",
        ax=ax,
    )

    ax.set_title("Prompt Intent Clustering", fontsize=16)
    ax.legend(title="Cluster ID", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    json_path = out_path.with_suffix(".json")
    _write_clustering_json(
        prompts_path=prompts_path,
        image_out=out_path,
        json_out=json_path,
        texts=data,
        labels=label_list,
    )
    _log(f"Wrote image  {out_path.resolve()}")
    _log(f"Wrote JSON   {json_path.resolve()}")
    _log("Done.")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
