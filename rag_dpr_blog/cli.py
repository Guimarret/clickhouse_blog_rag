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
    scores = embs @ qvec
    top_idx = np.argsort(-scores)[:k]

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


if __name__ == "__main__":
    app()
