# Rag_DPR_blog

Dense Passage Retrieval + BERTopic topic modeling over the ClickHouse
blog corpus. Individual coursework project.

## What this does

Builds a vector "table" over 791 posts from clickhouse.com/blog and
exposes a semantic search via CLI. Every chunk returned carries its
publication date, the ClickHouse version mentioned (when one could be
parsed), and the label of the topic cluster it belongs to — ready for
citation or temporal filtering.

```
sitemap clickhouse.com/blog
        │
        ▼
 [trafilatura]      791 posts → blog.parquet
        │
        ▼
 [chunker]          512-token chunks, 64-token overlap
        │
        ▼ 6171 chunks
        │
        ▼
 [bge-large-en-v1.5]  dense encoder, 1024-d
        │           (asymmetric passage / query prefixes)
        ▼
 [BERTopic]         UMAP + HDBSCAN + KeyBERTInspired
        │           - 73 discovered topics
        │           - merge-tree hierarchy persisted
        │           - outliers reassigned by centroid distance
        ▼
 [Claude Haiku 4.5] one sentence per topic (label_llm)
        │
        ▼
 query  →  top-k chunks  (with date + version + topic label)
```

## Repository layout

```
.
├── README.md                 (this file)
├── PLAN.md                   (original design plan)
├── pyproject.toml            (uv-managed deps)
├── uv.lock                   (pinned dep versions)
├── .env.example              (template for HF_TOKEN, ANTHROPIC_API_KEY)
│
├── rag_dpr_blog/             (Python package)
│   ├── __init__.py
│   ├── chunker.py            (token-bounded recursive splitter)
│   ├── encoders.py           (bge-large-en-v1.5 wrapper)
│   ├── topics.py             (BERTopic + dated checkpoints)
│   ├── index.py              (parquet I/O)
│   ├── llm.py                (Ollama + Anthropic client)
│   ├── evaluate.py           (eval-set seeding + recall@k / MRR scoring)
│   ├── reranker.py           (BGE cross-encoder; two-stage retrieval)
│   └── cli.py                (Typer CLI)
│
├── ollama_setup/             (optional local LLM setup)
│   ├── README.md
│   └── setup.sh
│
├── paper/                    (academic write-up — PLN final assignment)
│   ├── PLN_Rag_Artigo_AtvFinal.md   (Markdown source)
│   ├── PLN_Rag_Artigo_AtvFinal.tex  (LaTeX source)
│   └── PLN_Rag_Artigo_AtvFinal.pdf  (compiled PDF, 9 pages)
│
├── data/
│   ├── canonical/
│   │   ├── blog.parquet                      (791 canonical posts)
│   │   ├── blog_chunks.parquet               (6171 chunks)
│   │   └── blog_chunks_embedded.parquet      (chunks + embeddings)
│   └── topic_models/
│       └── topic_model_2026-05-28/
│           ├── topic_model/                  (serialized BERTopic state)
│           ├── topics.parquet                (73 topics + labels)
│           ├── chunks_topics.parquet         (chunk → topic_id mapping)
│           ├── hierarchy.json                (topic merge tree)
│           └── manifest.json                 (run config: seed, encoder, n_chunks)
│
└── eval/
    └── queries.yaml                          (30 curated Q&A pairs)
```

## Prerequisites

- Python ≥ 3.11, < 3.13 (upper bound is set by the `triton-rocm` wheels).
- [`uv`](https://docs.astral.sh/uv/) for environment management.
- For the embedding stage: AMD GPU (ROCm 6+) or NVIDIA (CUDA), or CPU
  (slower). Developed and tested on an RX 6700 XT with ROCm 7.2 and
  PyTorch ROCm 7.0.
- For the LLM-label stage: either a local Ollama install with
  `qwen2.5:14b`, or an Anthropic API key.

## Setup

```bash
# 1. Clone the repo and enter it
git clone <repo-url>.git
cd Rag_DPR_blog

# 2. Install deps
uv sync

# 3. Copy the .env template and fill in credentials
cp .env.example .env
$EDITOR .env
# Set at least:
#   HF_TOKEN=hf_...           (optional; lifts Hugging Face rate limit)
#   ANTHROPIC_API_KEY=sk-...  (needed for label-topics via Haiku)
```

### Optional: local Ollama setup (alternative to the Anthropic API)

```bash
bash ollama_setup/setup.sh
```

To route topic labeling through Ollama instead of Haiku, set
`RAG_DPR_BLOG_TOPIC_LABELER=ollama` before running `label-topics`.

## Usage

All data is included pre-processed — you can run `query` immediately:

```bash
uv run python -m rag_dpr_blog.cli query "how does ClickHouse handle vector search" --k 5
```

Output: top-5 chunks with publication date, version, and topic label.

### Rebuilding the pipeline from scratch

Each stage is independent and idempotent:

```bash
# 1. Chunking (CPU)
uv run python -m rag_dpr_blog.cli chunk

# 2. Embedding (GPU recommended; ~5 min on RX 6700 XT)
uv run python -m rag_dpr_blog.cli embed

# 3. BERTopic + KeyBERTInspired (CPU; ~1 min)
uv run python -m rag_dpr_blog.cli fit-topics

# 4. LLM labels (Haiku ~$0.15, or Ollama for free)
uv run python -m rag_dpr_blog.cli label-topics

# 5. Inspection
uv run python -m rag_dpr_blog.cli topics-list
uv run python -m rag_dpr_blog.cli topics-show 6
```

### Evaluation

A 30-query eval set lives at `eval/queries.yaml`. Each query was seeded
by Haiku from the medoid chunk of one of the top-30 largest topics, then
manually reviewed (Japanese-language posts had their queries translated
and ground-truth URLs re-pointed to their English equivalents). Ground
truth is *same post URL* — a query counts as a hit at rank `r` if any
chunk in the top-`r` shares the seed's source_url.

```bash
# Dense-only scoring (BGE bi-encoder, cosine top-k)
uv run python -m rag_dpr_blog.cli eval

# Two-stage scoring (BGE dense top-50 → BGE cross-encoder rerank → top-k)
uv run python -m rag_dpr_blog.cli eval --rerank

# Re-seed from scratch (~$0.10 in Haiku tokens)
uv run python -m rag_dpr_blog.cli eval-seed --n 30 --overwrite
```

The `query` command also accepts `--rerank` for interactive use:

```bash
uv run python -m rag_dpr_blog.cli query "how does ClickHouse handle vector search" --rerank --k 5
```

### Re-scraping the blog (optional — data is already included)

The scraping stage is not part of this standalone package because it
uses code from the parent project this side-track was carved out of.
The 791 posts already sit in `data/canonical/blog.parquet` for
reproducibility.

## Technical decisions

Documented in detail in [`PLAN.md`](PLAN.md). The most relevant points:

- **Encoder**: `BAAI/bge-large-en-v1.5`, 1024-d, strong MTEB English
  scores, 512-token context (matches chunk size). Asymmetric prefix
  for queries vs. passages.
- **Chunking**: recursive splitter (paragraph → sentence → word), hard
  cap of 512 tokens, 64-token overlap. Tokenizer = the encoder's own,
  so token counts are exact (not approximated).
- **Topic modeling**: BERTopic with UMAP (`random_state=42` for
  reproducibility) + HDBSCAN (`min_cluster_size=15`) + KeyBERTInspired
  re-ranking the c-TF-IDF keywords. Outliers (cluster −1) reassigned
  to the nearest centroid via the `embeddings` strategy.
- **LLM labels**: one sentence per topic via Haiku 4.5 (~$0.15 for the
  73), or Qwen 2.5 14B via local Ollama (free).
- **Determinism**: pinned UMAP seed; BERTopic model persisted in an
  immutable dated directory (`topic_model_YYYY-MM-DD/`); manifest
  carries the full run config.
- **Retrieval mode**: A — plain cosine vector search; the topic label
  rides along as metadata on each hit. Mode B (two-stage, topic-filter
  → vector-rank) is left as an optional follow-up.
- **Evaluation**: 30 Haiku-seeded queries (one per top-30 largest topic's
  medoid chunk), manually reviewed. Ground truth = same post URL.
  Metrics: recall@{1,3,5,10}, MRR. Reproducible via `eval-seed` + `eval`.
- **Reranker**: `BAAI/bge-reranker-large` cross-encoder on top of the BGE
  bi-encoder. Two-stage: dense top-50 → cross-encoder rescore → top-k.
  Lifts recall@10 by +10 points on the eval set, recall@5 by +13.

## Observed metrics

- 791 posts → 6,171 chunks (median ≈ 470 tokens per chunk)
- Embeddings: 6,171 × 1024 float32 ≈ 25 MB raw, 31 MB in Parquet
- AMD-GPU embedding pass: ~5:16 (~1.65 s per batch of 32)
- BERTopic fit (CPU): ~1 min — UMAP 18 s, HDBSCAN 0.1 s,
  KeyBERTInspired 35 s
- 73 topics discovered; after outlier reduction, zero chunks left in
  cluster −1
- Haiku labels: ~1.5 min for the 73 topics, total cost ≈ $0.10–0.15
- Retrieval on the 30-query eval set:

  |              | dense only | + rerank |
  |--------------|-----------:|---------:|
  | recall@1     | 0.533      | 0.500    |
  | recall@3     | 0.700      | 0.800    |
  | recall@5     | 0.733      | 0.867    |
  | recall@10    | 0.833      | **0.933** |
  | MRR          | 0.628      | 0.662    |
  | misses (/30) | 5          | 2        |

  Reranking shaves the top by 1 (cross-encoders sometimes swap rank-1
  for a fractionally more relevant chunk in a different post), but lifts
  everything from recall@3 outward substantially.

## Out of scope (future work)

- Hybrid BM25 + vector retrieval via Reciprocal Rank Fusion
- Two-stage topic-filtered retrieval (opt-in via `--two-stage`)
- UI (Streamlit / Gradio) for demo
- Vector-store persistence (pgvector or ClickHouse's own vector index)

## License

Academic use.
