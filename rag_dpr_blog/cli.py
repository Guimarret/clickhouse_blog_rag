"""Typer CLI for the Rag_DPR_blog side project.

Run from the project root:

    uv run python -m rag_dpr_blog.cli <command> [options]
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

# Load .env before any subcommand runs, so HF_TOKEN / ANTHROPIC_API_KEY /
# RAG_DPR_BLOG_* are visible to sentence-transformers / anthropic / etc.
load_dotenv()

app = typer.Typer(add_completion=False, help="Dense-retrieval + BERTopic over the ClickHouse blog corpus.")

ROOT = Path.cwd()
BLOG_PARQUET = ROOT / "data" / "canonical" / "blog.parquet"
CHUNKS_PARQUET = ROOT / "data" / "canonical" / "blog_chunks.parquet"
CHUNKS_EMB_PARQUET = ROOT / "data" / "canonical" / "blog_chunks_embedded.parquet"
TOPIC_MODELS_ROOT = ROOT / "data" / "topic_models"


@app.command("chunk")
def cli_chunk(
    chunk_tokens: Annotated[int, typer.Option(help="Hard cap per chunk.")] = 512,
    overlap_tokens: Annotated[int, typer.Option(help="Token overlap between chunks.")] = 64,
) -> None:
    """Token-bounded splitting of blog.parquet → blog_chunks.parquet."""
    from rag_dpr_blog.chunker import chunk_records
    from rag_dpr_blog.index import read_blog, write_chunks

    df = read_blog(BLOG_PARQUET)
    typer.echo(f"Loaded {len(df)} blog leaves from {BLOG_PARQUET}")

    records = df.to_dict("records")
    out_rows = list(
        chunk_records(records, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    )
    out_df = pd.DataFrame(out_rows)
    write_chunks(CHUNKS_PARQUET, out_df)
    typer.echo(
        f"Wrote {len(out_df)} chunks → {CHUNKS_PARQUET}"
        f"  (median n_tokens={int(out_df['n_tokens'].median())})"
    )


@app.command("embed")
def cli_embed(
    model_name: Annotated[
        str, typer.Option("--model", help="Sentence-transformer model.")
    ] = "BAAI/bge-large-en-v1.5",
    batch_size: Annotated[int, typer.Option(help="Encoder batch size.")] = 32,
    device: Annotated[str | None, typer.Option(help="Override device.")] = None,
) -> None:
    """Encode chunk_text → embedding column; write blog_chunks_embedded.parquet."""
    from rag_dpr_blog.encoders import Encoder
    from rag_dpr_blog.index import read_chunks, write_embeddings

    df = read_chunks(CHUNKS_PARQUET)
    typer.echo(f"Loaded {len(df)} chunks; encoding with {model_name} ...")

    enc = Encoder(model_name, device=device)
    embs = enc.encode(df["chunk_text"].tolist(), kind="passage",
                      batch_size=batch_size, show_progress=True)
    write_embeddings(CHUNKS_EMB_PARQUET, df, embs, model_name)
    typer.echo(f"Wrote {len(df)} embeddings (dim={enc.dim}) → {CHUNKS_EMB_PARQUET}")


@app.command("fit-topics")
def cli_fit_topics(
    min_topic_size: Annotated[int, typer.Option(help="HDBSCAN min_cluster_size.")] = 15,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip LLM label generation.")] = False,
) -> None:
    """Fit BERTopic on the embeddings; write a dated checkpoint directory."""
    from rag_dpr_blog.index import read_embeddings
    from rag_dpr_blog.topics import FitConfig, fit

    df, embs = read_embeddings(CHUNKS_EMB_PARQUET)
    typer.echo(f"Loaded {len(df)} chunks + embeddings (dim={embs.shape[1]})")

    cfg = FitConfig(min_topic_size=min_topic_size, label_with_llm=not no_llm)
    out_dir = fit(
        docs=df["chunk_text"].tolist(),
        embeddings=embs,
        out_root=TOPIC_MODELS_ROOT,
        chunk_ids=df["chunk_id"].tolist(),
        cfg=cfg,
    )
    typer.echo(f"Wrote checkpoint → {out_dir}")


@app.command("label-topics")
def cli_label_topics(
    checkpoint: Annotated[Path | None, typer.Option(help="Override checkpoint dir.")] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Re-label topics that already have an llm label."),
    ] = False,
) -> None:
    """Generate LLM topic labels for an existing checkpoint (no BERTopic refit).

    Backend chosen by RAG_DPR_BLOG_TOPIC_LABELER env var (haiku|ollama|auto).
    """
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dotenv import load_dotenv

    from rag_dpr_blog.llm import summarize_topic_auto
    from rag_dpr_blog.topics import latest_checkpoint

    load_dotenv()
    cp = checkpoint or latest_checkpoint(TOPIC_MODELS_ROOT)
    if cp is None:
        typer.echo(f"No checkpoints under {TOPIC_MODELS_ROOT}")
        raise typer.Exit(code=1)

    topics_df = pq.read_table(cp / "topics.parquet").to_pandas()
    typer.echo(f"Labeling {len(topics_df)} topics in {cp.name} ...")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for idx, row in topics_df.iterrows():
        if row["label_llm"] and not overwrite:
            n_skipped += 1
            continue
        try:
            label = summarize_topic_auto(
                list(row["keywords"]),
                list(row["representative_excerpts"]),
            )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  [{row['topic_id']:3d}] failed: {e}")
            n_failed += 1
            continue
        if not label:
            n_failed += 1
            continue
        topics_df.at[idx, "label_llm"] = label
        n_done += 1
        typer.echo(f"  [{row['topic_id']:3d}] {label[:100]}")

    pq.write_table(
        pa.Table.from_pandas(topics_df, preserve_index=False),
        cp / "topics.parquet",
    )

    # Refresh chunks_topics.parquet so topic_label uses the new LLM labels
    label_by_topic = {
        int(r["topic_id"]): (r["label_llm"] or r["label_short"])
        for _, r in topics_df.iterrows()
    }
    chunks_topics_df = pq.read_table(cp / "chunks_topics.parquet").to_pandas()
    chunks_topics_df["topic_label"] = chunks_topics_df["topic_id"].map(label_by_topic)
    pq.write_table(
        pa.Table.from_pandas(chunks_topics_df, preserve_index=False),
        cp / "chunks_topics.parquet",
    )

    manifest_path = cp / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["labeled_with_llm"] = bool(n_done) or manifest.get("labeled_with_llm", False)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    typer.echo(
        f"Done. labeled={n_done}  skipped(existing)={n_skipped}  failed={n_failed}"
    )


@app.command("topics-list")
def cli_topics_list(
    checkpoint: Annotated[Path | None, typer.Option(help="Override checkpoint dir.")] = None,
) -> None:
    """Print one line per topic from the latest (or given) checkpoint."""
    from rag_dpr_blog.topics import latest_checkpoint
    import pyarrow.parquet as pq

    cp = checkpoint or latest_checkpoint(TOPIC_MODELS_ROOT)
    if cp is None:
        typer.echo(f"No checkpoints under {TOPIC_MODELS_ROOT}")
        raise typer.Exit(code=1)

    df = pq.read_table(cp / "topics.parquet").to_pandas()
    typer.echo(f"# {cp.name}  ({len(df)} topics)")
    for _, r in df.iterrows():
        label = r["label_llm"] or r["label_short"]
        typer.echo(f"  [{r['topic_id']:3d}]  n={r['size']:4d}  {label}")


@app.command("topics-show")
def cli_topics_show(
    topic_id: int,
    checkpoint: Annotated[Path | None, typer.Option(help="Override checkpoint dir.")] = None,
) -> None:
    """Print keywords + exemplars for one topic."""
    from rag_dpr_blog.topics import latest_checkpoint
    import pyarrow.parquet as pq

    cp = checkpoint or latest_checkpoint(TOPIC_MODELS_ROOT)
    if cp is None:
        typer.echo(f"No checkpoints under {TOPIC_MODELS_ROOT}")
        raise typer.Exit(code=1)

    df = pq.read_table(cp / "topics.parquet").to_pandas()
    row = df.loc[df["topic_id"] == topic_id]
    if row.empty:
        typer.echo(f"topic_id={topic_id} not found")
        raise typer.Exit(code=1)
    r = row.iloc[0]
    typer.echo(f"Topic {topic_id}  (n={r['size']})")
    typer.echo(f"  label_short: {r['label_short']}")
    typer.echo(f"  label_llm:   {r['label_llm']}")
    typer.echo(f"  keywords:    {', '.join(r['keywords'])}")
    typer.echo("  excerpts:")
    for i, exc in enumerate(r["representative_excerpts"], 1):
        typer.echo(f"    [{i}] {exc[:240]} ...")


@app.command("query")
def cli_query(
    text: str,
    k: Annotated[int, typer.Option(help="Top-k chunks to return.")] = 5,
    rerank: Annotated[
        bool, typer.Option("--rerank/--no-rerank", help="Apply cross-encoder reranking.")
    ] = False,
    rerank_n: Annotated[int, typer.Option(help="Dense candidates to rerank.")] = 50,
    checkpoint: Annotated[Path | None, typer.Option(help="Override checkpoint dir.")] = None,
    model_name: Annotated[
        str, typer.Option("--model", help="Encoder for the query (must match passage encoder).")
    ] = "BAAI/bge-large-en-v1.5",
) -> None:
    """A-mode retrieval: plain cosine over chunks, topic label attached to each hit."""
    from rag_dpr_blog.encoders import Encoder
    from rag_dpr_blog.index import read_embeddings
    from rag_dpr_blog.topics import latest_checkpoint
    import pyarrow.parquet as pq

    df, embs = read_embeddings(CHUNKS_EMB_PARQUET)
    enc = Encoder(model_name)
    qvec = enc.encode([text], kind="query")[0]
    dense_scores = embs @ qvec
    if rerank:
        from rag_dpr_blog.reranker import Reranker

        cand_idx = np.argsort(-dense_scores)[:rerank_n]
        candidates = [df.iloc[int(i)]["chunk_text"] for i in cand_idx]
        rr = Reranker()
        rr_scores = rr.rerank(text, candidates)
        order = np.argsort(-rr_scores)
        top_idx = cand_idx[order][:k]
        scores = np.full_like(dense_scores, fill_value=float("-inf"))
        scores[top_idx] = rr_scores[order][:k]
    else:
        top_idx = np.argsort(-dense_scores)[:k]
        scores = dense_scores

    cp = checkpoint or latest_checkpoint(TOPIC_MODELS_ROOT)
    chunks_topics = None
    if cp is not None and (cp / "chunks_topics.parquet").exists():
        chunks_topics = pq.read_table(cp / "chunks_topics.parquet").to_pandas()

    typer.echo(f'Query: "{text}"\n')
    for rank, idx in enumerate(top_idx, 1):
        r = df.iloc[idx]
        topic_label = ""
        if chunks_topics is not None:
            row = chunks_topics.loc[chunks_topics["chunk_id"] == r["chunk_id"]]
            if not row.empty:
                topic_label = row.iloc[0]["topic_label"] or ""

        date_val = r.get("event_date")
        if date_val is None or str(date_val).startswith("1970"):
            date_str = "  unknown  "
        else:
            date_str = str(date_val)[:10]

        ver = r.get("version_introduced")
        ver_str = f"  v{ver}" if ver else ""

        typer.echo(
            f"[{rank}] {date_str}{ver_str}   score={scores[idx]:.3f}\n"
            f"     topic: {topic_label}\n"
            f"     {r['title']}\n"
            f"     {r['source_url']}\n"
            f"     {r['chunk_text'][:240].replace(chr(10), ' ')} ...\n"
        )


@app.command("eval-seed")
def cli_eval_seed(
    n_queries: Annotated[int, typer.Option("--n", help="Number of seed queries to generate.")] = 30,
    out_path: Annotated[Path, typer.Option("--out", help="Output YAML path.")] = Path("eval/queries.yaml"),
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing file.")] = False,
    checkpoint: Annotated[Path | None, typer.Option(help="Override checkpoint dir.")] = None,
) -> None:
    """Seed eval queries via Haiku: one question per top topic's medoid chunk."""
    import pyarrow.parquet as pq

    from rag_dpr_blog.evaluate import (
        EvalQuery,
        generate_query_for_chunk,
        select_seed_chunks,
        write_eval_yaml,
    )
    from rag_dpr_blog.index import read_embeddings
    from rag_dpr_blog.topics import latest_checkpoint

    if out_path.exists() and not overwrite:
        typer.echo(f"{out_path} exists. Pass --overwrite to replace.")
        raise typer.Exit(code=1)

    cp = checkpoint or latest_checkpoint(TOPIC_MODELS_ROOT)
    if cp is None:
        typer.echo(f"No checkpoints under {TOPIC_MODELS_ROOT}")
        raise typer.Exit(code=1)

    df, embs = read_embeddings(CHUNKS_EMB_PARQUET)
    chunks_topics_df = pq.read_table(cp / "chunks_topics.parquet").to_pandas()
    seeds = select_seed_chunks(df, embs, chunks_topics_df, n_queries=n_queries)
    typer.echo(f"Selected {len(seeds)} seed chunks from {cp.name}. Generating queries via Haiku ...")

    queries: list[EvalQuery] = []
    for i, seed in enumerate(seeds, 1):
        try:
            q_text = generate_query_for_chunk(seed["chunk_text"])
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  [{i:2d}/{len(seeds)}] topic={seed['topic_id']:3d}  FAILED: {e}")
            continue
        typer.echo(f"  [{i:2d}/{len(seeds)}] topic={seed['topic_id']:3d}  {q_text[:80]}")
        queries.append(
            EvalQuery(
                query=q_text,
                relevant_source_url=seed["source_url"],
                seed_chunk_id=seed["chunk_id"],
                seed_topic_id=seed["topic_id"],
                notes="",
            )
        )

    write_eval_yaml(queries, out_path)
    typer.echo(
        f"\nWrote {len(queries)} queries → {out_path}\n"
        "Review and refine the YAML, then run: uv run python -m rag_dpr_blog.cli eval"
    )


@app.command("eval")
def cli_eval(
    queries_path: Annotated[Path, typer.Option("--queries", help="Path to queries YAML.")] = Path("eval/queries.yaml"),
    model_name: Annotated[
        str, typer.Option("--model", help="Encoder model (must match passage encoder).")
    ] = "BAAI/bge-large-en-v1.5",
    rerank: Annotated[
        bool, typer.Option("--rerank/--no-rerank", help="Apply cross-encoder reranking.")
    ] = False,
    rerank_n: Annotated[int, typer.Option(help="Dense candidates to rerank per query.")] = 50,
    show_misses: Annotated[bool, typer.Option("--show-misses/--no-show-misses")] = True,
) -> None:
    """Score the eval set: recall@{1,3,5,10} + MRR."""
    from rag_dpr_blog.encoders import Encoder
    from rag_dpr_blog.evaluate import load_eval_yaml, score
    from rag_dpr_blog.index import read_embeddings

    if not queries_path.exists():
        typer.echo(f"No eval file at {queries_path}. Run `eval-seed` first.")
        raise typer.Exit(code=1)

    queries = load_eval_yaml(queries_path)
    df, embs = read_embeddings(CHUNKS_EMB_PARQUET)
    typer.echo(f"Loaded {len(queries)} queries from {queries_path}.")
    typer.echo(
        f"Scoring against {len(df)} chunks "
        f"({'dense + cross-encoder rerank top-' + str(rerank_n) if rerank else 'dense only'}) ...\n"
    )

    enc = Encoder(model_name)
    reranker = None
    if rerank:
        from rag_dpr_blog.reranker import Reranker

        reranker = Reranker()
        typer.echo(f"Reranker: {reranker.model_name} on {reranker.device}\n")

    report = score(queries, df, embs, encoder=enc, reranker=reranker, rerank_n=rerank_n)

    typer.echo("Metrics")
    typer.echo("-------")
    for k, v in report.metrics.items():
        typer.echo(f"  {k:10s} = {v:.3f}")

    if show_misses:
        max_k = max(
            int(k.split("@")[1]) for k in report.metrics if k.startswith("recall@")
        )
        misses = [r for r in report.per_query if r.first_rank is None]
        if misses:
            typer.echo(f"\nMisses (not in top {max_k}):")
            for r in misses:
                typer.echo(f"  [---]  {r.query[:80]}")
                typer.echo(f"         expected: {r.relevant_source_url}")
        else:
            typer.echo(f"\nNo misses — every query hit within top {max_k}.")


if __name__ == "__main__":
    app()
