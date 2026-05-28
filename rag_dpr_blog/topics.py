"""BERTopic fit / load / transform + dated checkpoint I/O.

Decisions locked in PLAN.md (§4):
- c-TF-IDF candidate keywords, KeyBERTInspired re-ranking
- HDBSCAN outliers reassigned via embeddings-strategy reduce_outliers
- hierarchical_topics() computed and persisted (UI deferred)
- UMAP random_state pinned for reproducibility
- one-sentence LLM label per topic via local Qwen 2.5 7B (Ollama)
- each fit lives in its own dated, immutable directory
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

UMAP_SEED = 42
DEFAULT_MIN_TOPIC_SIZE = 15


@dataclass(frozen=True)
class FitConfig:
    min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_samples: int | None = None  # defaults to min_topic_size
    label_with_llm: bool = True
    encoder_name: str = "BAAI/bge-large-en-v1.5"


def _build_model(cfg: FitConfig) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.umap_n_components,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=UMAP_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.min_topic_size,
        min_samples=cfg.hdbscan_min_samples or cfg.min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    # Force BGE onto CPU for the KeyBERTInspired vocab pass.
    # Why: the RX 6700 XT (12 GB) cannot reliably host BGE alongside
    # Ollama's qwen2.5:14b (~9.5 GB) — concurrent rocBLAS GEMMs cause
    # the amdgpu driver to hang. (Crashed the host on 2026-05-28.)
    # Vocab encoding is ~10k words once; a few seconds on CPU is fine.
    embedding_model = SentenceTransformer(cfg.encoder_name, device="cpu")
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=KeyBERTInspired(top_n_words=10),
        calculate_probabilities=False,
        verbose=True,
    )


def fit(
    docs: list[str],
    embeddings: np.ndarray,
    *,
    out_root: Path,
    chunk_ids: list[str],
    cfg: FitConfig | None = None,
    fit_date: date | None = None,
) -> Path:
    """Fit BERTopic, reduce outliers, persist a dated checkpoint.

    Returns the checkpoint directory path.
    """
    cfg = cfg or FitConfig()
    if len(docs) != embeddings.shape[0] or len(docs) != len(chunk_ids):
        raise ValueError("docs, embeddings, chunk_ids must have the same length")

    model = _build_model(cfg)
    topics, _ = model.fit_transform(docs, embeddings=embeddings)

    # Reduce outliers using the embeddings strategy: each -1 chunk is
    # assigned to the topic with the closest centroid.
    new_topics = model.reduce_outliers(
        docs, topics, strategy="embeddings", embeddings=embeddings
    )
    model.update_topics(docs, topics=new_topics)

    # Hierarchy on c-TF-IDF topic vectors. Persisted but not consumed
    # by retrieval yet.
    hierarchy_df = model.hierarchical_topics(docs)

    # LLM labels — one call per non-outlier topic. Best-effort: if the
    # daemon is down, fall back to short keyword labels.
    llm_labels: dict[int, str] = {}
    if cfg.label_with_llm:
        llm_labels = _label_topics_with_llm(model, docs)

    fit_date = fit_date or date.today()
    out_dir = out_root / f"topic_model_{fit_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(
        str(out_dir / "topic_model"),
        serialization="safetensors",
        save_embedding_model=False,
        save_ctfidf=True,
    )

    topics_df = _topics_table(model, new_topics, chunk_ids, llm_labels)
    pq.write_table(pa.Table.from_pandas(topics_df, preserve_index=False), out_dir / "topics.parquet")

    chunks_topics_df = pd.DataFrame(
        {
            "chunk_id": chunk_ids,
            "topic_id": new_topics,
            "topic_label": [_label_for(topics_df, t) for t in new_topics],
        }
    )
    pq.write_table(
        pa.Table.from_pandas(chunks_topics_df, preserve_index=False),
        out_dir / "chunks_topics.parquet",
    )

    hierarchy_df.to_json(out_dir / "hierarchy.json", orient="records", indent=2)

    manifest = {
        "fit_date": fit_date.isoformat(),
        "n_chunks": len(docs),
        "encoder": cfg.encoder_name,
        "umap_seed": UMAP_SEED,
        "min_topic_size": cfg.min_topic_size,
        "n_topics": int(topics_df["topic_id"].nunique()),
        "outlier_reduction": "embeddings",
        "labeled_with_llm": bool(llm_labels),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return out_dir


def _topics_table(
    model: BERTopic,
    final_topics: list[int],
    chunk_ids: list[str],
    llm_labels: dict[int, str],
) -> pd.DataFrame:
    info = model.get_topic_info()  # DataFrame with Topic, Count, Name, Representation, Representative_Docs
    rep_docs_by_topic = model.get_representative_docs() or {}

    # Map representative-doc texts → chunk_ids by index lookup. The
    # representative-doc set is small (3 per topic), so a per-topic
    # scan is fine.
    chunk_text_to_id: dict[str, list[str]] = {}
    counts = pd.Series(final_topics).value_counts().to_dict()

    rows = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # outliers were reassigned; nothing should remain
        keywords = [w for w, _ in (model.get_topic(tid) or [])]
        rep_docs = rep_docs_by_topic.get(tid, []) or []
        rows.append(
            {
                "topic_id": tid,
                "size": int(counts.get(tid, row["Count"])),
                "keywords": keywords,
                "label_short": " | ".join(keywords[:3]) if keywords else f"topic_{tid}",
                "label_llm": llm_labels.get(tid),
                "representative_excerpts": [d[:500] for d in rep_docs],
            }
        )
    df = pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)
    return df


def _label_for(topics_df: pd.DataFrame, topic_id: int) -> str:
    row = topics_df.loc[topics_df["topic_id"] == topic_id]
    if row.empty:
        return f"topic_{topic_id}"
    r = row.iloc[0]
    return r["label_llm"] or r["label_short"]


def _label_topics_with_llm(model: BERTopic, docs: list[str]) -> dict[int, str]:
    """Best-effort topic summaries via local Ollama. Skip silently on failure."""
    from rag_dpr_blog.llm import OllamaConfig, is_alive, summarize_topic

    cfg = OllamaConfig()
    if not is_alive(cfg):
        print(
            f"[topics] Ollama not reachable at {cfg.host} — skipping LLM labels.",
            file=sys.stderr,
        )
        return {}

    rep_docs = model.get_representative_docs() or {}
    labels: dict[int, str] = {}
    for tid in sorted(t for t in rep_docs if t != -1):
        keywords = [w for w, _ in (model.get_topic(tid) or [])][:10]
        try:
            labels[tid] = summarize_topic(keywords, rep_docs[tid], cfg=cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[topics] LLM label failed for topic {tid}: {e}", file=sys.stderr)
    return labels


def load(checkpoint_dir: Path) -> tuple[BERTopic, pd.DataFrame]:
    """Load a saved BERTopic checkpoint + its topics.parquet."""
    model = BERTopic.load(str(checkpoint_dir / "topic_model"))
    topics_df = pq.read_table(checkpoint_dir / "topics.parquet").to_pandas()
    return model, topics_df


def latest_checkpoint(root: Path) -> Path | None:
    candidates = sorted(root.glob("topic_model_*"))
    return candidates[-1] if candidates else None
