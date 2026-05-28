# ollama_setup

Local Ollama + Qwen 2.5 14B for the BERTopic labeling stage.

## What this gives you

- Ollama installed and running at `http://127.0.0.1:11434`.
- `qwen2.5:14b` pulled (~9 GB on disk, Q4_K_M quant). Fits on a 12 GB
  GPU with ~2 GB headroom for KV cache — plenty for the short prompts
  the topic labeler sends.
- The Python client (`rag_dpr_blog/llm.py`, a small `httpx`-only
  wrapper) is what `rag_dpr_blog.topics` imports to summarize each
  discovered topic into a one-sentence label.

No API key, no network usage at inference time, no per-call cost.

## One-time install

```bash
bash Rag_DPR_blog/ollama_setup/setup.sh
```

The script is idempotent — safe to re-run. It will:

1. Install Ollama via the official installer if not already present.
2. Start the daemon (systemd or background `ollama serve`).
3. `ollama pull qwen2.5:7b` (skips cached layers).
4. Smoke-test the model with a tiny completion.

## Verifying

```bash
curl http://127.0.0.1:11434/api/tags                  # list pulled models
uv run python -c "from rag_dpr_blog.llm import is_alive; print(is_alive())"
```

## Configuration

Override with environment variables (read by `ollama_client.OllamaConfig`):

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Daemon URL |
| `RAG_DPR_BLOG_OLLAMA_MODEL` | `qwen2.5:14b` | Model tag to use |

To swap the model (e.g. down to 7B for speed, up to 32B if you have the
VRAM, or sideways to Llama 3.1 / Phi-4), pull it first and then set the
env var:

```bash
ollama pull qwen2.5:7b
RAG_DPR_BLOG_OLLAMA_MODEL=qwen2.5:7b uv run python -m rag_dpr_blog.cli fit-topics
```

## Hardware notes

- `qwen2.5:14b` at Q4_K_M needs ~9.5 GB VRAM for weights + ~1–2 GB for
  KV cache. Tight but comfortable on a 12 GB GPU for the short prompts
  this stage sends (~1.5k input tokens, ~50 output tokens per topic).
- If you hit OOM, drop to `qwen2.5:14b-instruct-q3_K_M` (~7.5 GB) or
  fall back to `qwen2.5:7b` (~5 GB).
- Topic-labeling pass for ~60 topics finishes in ~1–2 min on GPU,
  roughly 5–10 min on CPU. One-shot per fit.

## Fallback to a hosted API

If you don't want a local model, set `RAG_DPR_BLOG_TOPIC_LABELER=haiku`
(handled by `rag_dpr_blog.topics`) — it will route to Claude Haiku via
the Anthropic API (~$0.20 per fit). Requires `ANTHROPIC_API_KEY`.
