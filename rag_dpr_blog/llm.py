"""Thin client wrapper used by the topic-labeling stage.

We don't depend on the `ollama` Python package — the REST API is small
enough that httpx + a dataclass is clearer and one less dep to pin.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class OllamaConfig:
    host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model: str = os.environ.get("RAG_DPR_BLOG_OLLAMA_MODEL", "qwen2.5:14b")
    timeout_s: float = 120.0
    max_retries: int = 3
    temperature: float = 0.2


def is_alive(cfg: OllamaConfig | None = None) -> bool:
    cfg = cfg or OllamaConfig()
    try:
        r = httpx.get(f"{cfg.host}/api/tags", timeout=5.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def generate(
    prompt: str,
    *,
    system: str | None = None,
    cfg: OllamaConfig | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """One-shot completion. Returns the model's response text."""
    cfg = cfg or OllamaConfig()
    payload: dict[str, Any] = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": cfg.temperature, **(options or {})},
    }
    if system is not None:
        payload["system"] = system

    last_exc: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            with httpx.Client(timeout=cfg.timeout_s) as client:
                r = client.post(f"{cfg.host}/api/generate", json=payload)
                r.raise_for_status()
                return (r.json().get("response") or "").strip()
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(0.5 * attempt)
    assert last_exc is not None
    raise last_exc


_TOPIC_SYSTEM = (
    "You are labeling topics discovered in a ClickHouse-related blog corpus. "
    "Output exactly one sentence (max 25 words) describing the topic. "
    "No preamble, no quotes, no bullet points."
)


def _build_topic_prompt(keywords: list[str], representative_docs: list[str]) -> str:
    kw_line = ", ".join(keywords[:10])
    exemplars = "\n\n---\n\n".join(d[:800] for d in representative_docs[:5])
    return (
        f"Top keywords: {kw_line}\n\n"
        f"Representative excerpts:\n{exemplars}\n\n"
        "One-sentence topic label:"
    )


def summarize_topic(
    keywords: list[str],
    representative_docs: list[str],
    *,
    cfg: OllamaConfig | None = None,
) -> str:
    """Return a one-sentence topic label via local Ollama."""
    cfg = cfg or OllamaConfig()
    prompt = _build_topic_prompt(keywords, representative_docs)
    return generate(prompt, system=_TOPIC_SYSTEM, cfg=cfg)


# --- Haiku fallback (used when Ollama is unavailable / crashes) ---

HAIKU_MODEL = "claude-haiku-4-5"


def summarize_topic_haiku(
    keywords: list[str],
    representative_docs: list[str],
    *,
    model: str = HAIKU_MODEL,
) -> str:
    """Return a one-sentence topic label via the Anthropic API (Haiku 4.5).

    Used as a fallback when Ollama is unavailable on this host (e.g. the
    RX 6700 XT amdgpu hang we hit on 2026-05-28). Costs ~$0.002 per call
    on Haiku 4.5; ~$0.20 across ~80 topics.
    """
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_TOPIC_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": _build_topic_prompt(keywords, representative_docs),
            }
        ],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return ("".join(text_blocks)).strip()


def summarize_topic_auto(
    keywords: list[str],
    representative_docs: list[str],
) -> str | None:
    """Route to Ollama or Haiku based on env var + availability.

    Selection order:
      1. RAG_DPR_BLOG_TOPIC_LABELER=haiku   → Haiku via API
      2. RAG_DPR_BLOG_TOPIC_LABELER=ollama  → Ollama (no fallback)
      3. unset → try Ollama, fall back to Haiku if ANTHROPIC_API_KEY is set

    Returns None if no backend is reachable.
    """
    choice = (os.environ.get("RAG_DPR_BLOG_TOPIC_LABELER") or "").lower().strip()

    if choice == "haiku":
        return summarize_topic_haiku(keywords, representative_docs)

    if choice == "ollama":
        return summarize_topic(keywords, representative_docs)

    if is_alive():
        try:
            return summarize_topic(keywords, representative_docs)
        except Exception:
            pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return summarize_topic_haiku(keywords, representative_docs)
    return None
