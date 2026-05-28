# Rag_DPR_blog — Side Plan

Self-contained side track: build a dense retrieval index over the
ClickHouse blog corpus, sliced into token-bounded chunks, embedded with
a chosen encoder, and organized with **BERTopic** (no RAPTOR).
Independent from the main project; reuses
`data/canonical/blog.parquet`.

## 1. Goal

Working **dense passage retrieval** over 791 blog posts already on disk,
with BERTopic-derived topic labels attached to every chunk for
explainability and (future) filtering.

End state: `query_text → top-k blog chunks (+ topic label)` works
locally with metadata-aware filters (date, version) intact.

## 2. Input

- `data/canonical/blog.parquet` — 791 rows, schema = `CanonicalLeaf`.
- Average body: ~12.5k chars ≈ 3–3.5k tokens.
- Estimate after chunking at 512 tokens / 64 overlap: **~6,500 chunks**.

## 3. Pipeline (4 stages, parquet-on-disk between each)

```
blog.parquet
   │ chunk
   ▼
blog_chunks.parquet
   │ embed
   ▼
blog_chunks_embedded.parquet
   │ fit-topics  ── writes ──►  topic_model_<date>/
   ▼                            topics.parquet
blog_chunks_topics.parquet        (one row per topic)
   │ query
   ▼
top-k chunks  (+ topic_label as metadata)
```

## 4. Locked decisions

| Decision | Choice |
|---|---|
| Retrieval mode | **A — plain vector search**; topic ID/label ride along as metadata |
| Encoder | `BAAI/bge-large-en-v1.5` (1024-dim, asymmetric query/passage prefixes) |
| Chunk size | 512 tokens / 64 overlap, recursive splitter |
| Topic representation | **c-TF-IDF + KeyBERTInspired** re-ranking |
| Outlier reduction | `reduce_outliers(strategy="embeddings")` — assign every −1 to nearest topic centroid by cosine |
| Topic summaries | **Local Qwen 2.5 7B via Ollama**, one-shot per topic (free, ~1 min total). Falls back to Haiku via API if local unavailable |
| Hierarchy | `hierarchical_topics()` computed and persisted; UI consumption deferred |
| UMAP determinism | pin `random_state=42`; runs single-threaded (irrelevant at our scale) |
| Checkpointing | each fit → `topic_model_YYYY-MM-DD/`, immutable; new fits go in a new dir |
| `min_topic_size` | start at **15**; tune after first fit |

## 5. Chunk schema

```
chunk_id            str   — primary key, f"{unit_id}-c{idx:03d}"
unit_id             str   — FK to CanonicalLeaf
chunk_index         int
chunk_text          str
n_tokens            int
title               str   — denormalized for retrieval-time citation
event_date          date
version_introduced  str | null
source_url          str
embedding           list[float] | null   — filled by stage 2
embedding_model     str | null
topic_id            int | null           — filled by stage 3 (after outlier reduction, never -1)
topic_label         str | null           — KeyBERTInspired short label
topic_prob          float | null
```

## 6. Topic table schema

`topics.parquet`, one row per topic — written by `fit-topics`:

```
topic_id              int
size                  int                  — number of chunks
keywords              list[str]            — KeyBERTInspired top-10 terms
label_short           str                  — joined top-3 keywords
label_llm             str | null           — Qwen 2.5 7B 1-2 sentence summary
representative_ids    list[str]            — chunk_ids BERTopic picked as exemplars
parent_topic_id       int | null           — from hierarchical_topics merge tree
hierarchy_depth       int                  — distance from root
```

## 7. Checkpoint layout

```
data/topic_models/
└── topic_model_2026-05-28/
    ├── topic_model/                # safetensors-serialized BERTopic state
    ├── topics.parquet              # the table above
    ├── chunks_topics.parquet       # joined per-chunk assignments
    ├── hierarchy.json              # serialized merge tree
    └── manifest.json               # encoder name, seed, min_topic_size, n_chunks, git_sha
```

Treated as immutable. Refit → new dated dir. `latest` symlink optional.

## 8. Encoder rationale (kept short)

`BAAI/bge-large-en-v1.5` — strong MTEB scores, English-only matches
corpus, 512 ctx aligns with chunk size, asymmetric (passage prefix +
`"Represent this sentence for searching relevant passages: "` for
queries). Open weights, runs locally on a 12 GB GPU.

Alternates kept in the back pocket for a future bake-off but not pursued
in v1: `nomic-embed-text-v1.5`, `intfloat/e5-large-v2`,
`jinaai/jina-embeddings-v3`.

## 9. Module layout

```
rag_dpr_blog/
├── __init__.py
├── chunker.py         # token-bounded recursive splitter
├── encoders.py        # bge-large wrapper, query/passage prefixes
├── topics.py          # fit / load / transform BERTopic, hierarchy, LLM labels
├── index.py           # parquet I/O for chunks + topics
├── eval.py            # recall@k / MRR on a small Q&A set (future)
└── cli.py             # Typer subcommands

Rag_DPR_blog/
├── PLAN.md            # this file
└── ollama_setup/      # local Qwen 2.5 7B install + client wrapper
    ├── README.md
    ├── setup.sh
    └── ollama_client.py
```

## 10. CLI

```
uv run python -m rag_dpr_blog.cli chunk
uv run python -m rag_dpr_blog.cli embed
uv run python -m rag_dpr_blog.cli fit-topics                       # writes new dated checkpoint
uv run python -m rag_dpr_blog.cli topics-list  [--checkpoint DIR]  # show id/size/label
uv run python -m rag_dpr_blog.cli topics-show  TOPIC_ID
uv run python -m rag_dpr_blog.cli query  "..."  [--k 5] [--checkpoint DIR]
```

## 11. Smoke run

```bash
# one-time
bash Rag_DPR_blog/ollama_setup/setup.sh      # install + pull qwen2.5:7b
uv sync                                       # picks up new deps

# pipeline
uv run python -m rag_dpr_blog.cli chunk
uv run python -m rag_dpr_blog.cli embed
uv run python -m rag_dpr_blog.cli fit-topics
uv run python -m rag_dpr_blog.cli query "how does MergeTree handle deletes" --k 5
```

### Success criteria

- `blog_chunks.parquet`: 5k–10k rows, `n_tokens ∈ [128, 512]` for ≥ 98%.
- `blog_chunks_embedded.parquet`: every row has a 1024-d `embedding`.
- `topic_model_<date>/`: exists, `topics.parquet` has 40–100 rows, each
  with non-null `label_short` and `label_llm`.
- After outlier reduction, **no chunk has topic_id = −1**.
- `query` returns 5 chunks each with a `topic_label`, top result is
  hand-judged relevant on a few sanity queries.

## 12. Out of scope

- Re-ranking (cross-encoder pass).
- Custom DPR fine-tuning.
- Hybrid BM25 + vector — main project's Phase 5 territory.
- Indexing commit / release leaves — different cost profile.
- Topic browser UI — hierarchy is persisted for it; UI itself is later.

## 13. Relationship to the main project

- **Independent**: ships without touching the main project.
- **Reusable**: chunker, encoder wrappers, and the Ollama client all
  lift into the main pipeline once an encoder is locked there.
- **Comparable**: by the time the main project reaches its Phase 5
  baseline, we'll have measured numbers on this blog slice for
  comparison.
