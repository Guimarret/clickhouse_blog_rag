"""Evaluation: seed Q&A pair generation + retrieval scoring.

Workflow:

  1. `eval-seed` picks the medoid chunk of each of the top-N largest topics,
     uses Haiku 4.5 to draft one question per chunk, writes eval/queries.yaml.
  2. Human reviews / edits eval/queries.yaml (rephrasing cribs, fixing typos).
  3. `eval` runs each query through the BGE query encoder against the full
     chunk embedding matrix; reports recall@{1,3,5,10} and MRR.

Ground-truth granularity is *same post URL*, not exact chunk_id — a query
counts as a "hit" at rank r if any chunk in the top-r shares the seed
chunk's source_url. This treats retrieval as a post-level passage task,
which is forgiving toward near-duplicate chunks of the same article.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_source_url: str
    seed_chunk_id: str
    seed_topic_id: int
    notes: str = ""


@dataclass(frozen=True)
class QueryResult:
    query: str
    relevant_source_url: str
    first_rank: int | None  # 1-based rank of first relevant hit; None = miss in top-K


@dataclass(frozen=True)
class EvalReport:
    metrics: dict[str, float]
    per_query: list[QueryResult]


def select_seed_chunks(
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    chunks_topics_df: pd.DataFrame,
    *,
    n_queries: int,
) -> list[dict]:
    """Pick the medoid chunk of each of the top-N largest topics.

    Medoid = chunk with highest cosine similarity to the topic centroid.
    Returns ordered list of {topic_id, chunk_id, chunk_text, source_url, title}.
    """
    topic_sizes = (
        chunks_topics_df.groupby("topic_id").size().sort_values(ascending=False)
    )
    top_topics = topic_sizes.head(n_queries).index.tolist()

    cid_to_idx = {cid: i for i, cid in enumerate(chunks_df["chunk_id"].tolist())}
    chunks_indexed = chunks_df.set_index("chunk_id")

    seeds: list[dict] = []
    for tid in top_topics:
        tid_chunks = chunks_topics_df.loc[
            chunks_topics_df["topic_id"] == tid, "chunk_id"
        ].tolist()
        idxs = [cid_to_idx[cid] for cid in tid_chunks if cid in cid_to_idx]
        if not idxs:
            continue
        embs = embeddings[idxs]
        centroid = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-9:
            continue
        centroid /= norm
        sims = embs @ centroid
        medoid_local = int(np.argmax(sims))
        cid = tid_chunks[medoid_local]
        row = chunks_indexed.loc[cid]
        seeds.append(
            {
                "topic_id": int(tid),
                "chunk_id": str(cid),
                "chunk_text": str(row["chunk_text"]),
                "source_url": str(row["source_url"]),
                "title": str(row["title"]),
            }
        )
    return seeds


_QUERY_SYSTEM = (
    "You are generating evaluation queries for a ClickHouse blog retrieval system. "
    "Given a passage, write one realistic question whose answer is in that passage.\n"
    "Constraints:\n"
    "- 5 to 15 words\n"
    "- Sound like a real user search query (not a homework question)\n"
    "- Paraphrase — do NOT copy distinctive multi-word phrases verbatim from the passage\n"
    "- No preamble, no quotes, no explanation\n"
    "Output only the question."
)

_QUERY_MODEL = "claude-haiku-4-5"


def generate_query_for_chunk(chunk_text: str, *, model: str = _QUERY_MODEL) -> str:
    """Ask Haiku to draft one realistic user question whose answer is in chunk_text."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=128,
        system=_QUERY_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Passage:\n\n{chunk_text[:1500]}\n\nQuestion:",
            }
        ],
    )
    parts = [b.text for b in response.content if b.type == "text"]
    return "".join(parts).strip().strip('"').strip("'")


def write_eval_yaml(queries: list[EvalQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(q) for q in queries]
    path.write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True))


def load_eval_yaml(path: Path) -> list[EvalQuery]:
    raw = yaml.safe_load(path.read_text())
    return [EvalQuery(**row) for row in raw]


def score(
    queries: list[EvalQuery],
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    encoder,
    reranker: object | None = None,
    rerank_n: int = 50,
    k_list: tuple[int, ...] = (1, 3, 5, 10),
) -> EvalReport:
    """Run all queries through `encoder`, score against `embeddings`.

    A query hits at rank r if any chunk in the top-r shares the seed's
    source_url. If `reranker` is supplied, the top `rerank_n` dense
    candidates per query are rescored with the cross-encoder and re-sorted
    before computing recall@k / MRR.
    """
    if not queries:
        raise ValueError("queries is empty")

    qvecs = encoder.encode([q.query for q in queries], kind="query")
    dense_scores = qvecs @ embeddings.T
    chunk_urls = chunks_df["source_url"].tolist()
    chunk_texts = chunks_df["chunk_text"].tolist()

    max_k = max(k_list)
    per_query: list[QueryResult] = []
    for q, row_scores in zip(queries, dense_scores):
        cand_idx = np.argsort(-row_scores)[:rerank_n]
        if reranker is not None:
            candidates = [chunk_texts[int(i)] for i in cand_idx]
            rr_scores = reranker.rerank(q.query, candidates)
            order = np.argsort(-rr_scores)
            final_idx = cand_idx[order][:max_k]
        else:
            final_idx = cand_idx[:max_k]

        first_rank: int | None = None
        for r, idx in enumerate(final_idx, start=1):
            if chunk_urls[int(idx)] == q.relevant_source_url:
                first_rank = r
                break
        per_query.append(
            QueryResult(
                query=q.query,
                relevant_source_url=q.relevant_source_url,
                first_rank=first_rank,
            )
        )

    metrics: dict[str, float] = {}
    n = len(queries)
    for k in k_list:
        hits = sum(
            1 for r in per_query if r.first_rank is not None and r.first_rank <= k
        )
        metrics[f"recall@{k}"] = hits / n
    rr_sum = sum(1.0 / r.first_rank for r in per_query if r.first_rank is not None)
    metrics["mrr"] = rr_sum / n

    return EvalReport(metrics=metrics, per_query=per_query)
