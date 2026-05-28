"""Parquet I/O for chunks, embeddings, and per-chunk topic assignments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_blog(blog_parquet: Path) -> pd.DataFrame:
    return pq.read_table(blog_parquet).to_pandas()


def write_chunks(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def read_chunks(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def write_embeddings(path: Path, df: pd.DataFrame, embeddings: np.ndarray, model_name: str) -> None:
    """Write chunks + their embeddings as a fixed-size float32 list column."""
    if len(df) != embeddings.shape[0]:
        raise ValueError("rows and embeddings length mismatch")
    out = df.copy()
    out["embedding"] = list(embeddings.astype(np.float32))
    out["embedding_model"] = model_name
    write_chunks(path, out)


def read_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pq.read_table(path).to_pandas()
    embs = np.vstack([np.asarray(v, dtype=np.float32) for v in df["embedding"]])
    return df, embs


def join_topic_labels(chunks_df: pd.DataFrame, chunks_topics_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join topic_id + topic_label onto a chunks dataframe by chunk_id."""
    return chunks_df.merge(chunks_topics_df, on="chunk_id", how="left")
