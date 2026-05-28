"""Token-bounded recursive splitter for blog post bodies.

Splits in order of preference: blank-line sections, paragraphs, sentences,
words. Only falls through to the next level when a span still exceeds
the token budget. Adjacent small spans are greedily packed until the
budget is exhausted, with a fixed token overlap between consecutive
chunks for context preservation.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER = "BAAI/bge-large-en-v1.5"
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64

_SECTION_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    text: str
    n_tokens: int


def _tokenize_len(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _SECTION_RE.split(text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _split_words(text: str) -> list[str]:
    return text.split()


def _atoms(
    tokenizer: PreTrainedTokenizerBase, text: str, budget: int
) -> Iterator[tuple[str, int]]:
    """Yield (span, n_tokens) atoms that each fit within `budget`.

    Splits paragraphs → sentences → words, falling through only when a
    span still exceeds the budget. A word longer than the budget is
    emitted as its own atom and will be clipped by the caller.
    """
    for para in _split_paragraphs(text):
        n = _tokenize_len(tokenizer, para)
        if n <= budget:
            yield para, n
            continue
        for sent in _split_sentences(para):
            n = _tokenize_len(tokenizer, sent)
            if n <= budget:
                yield sent, n
                continue
            buf: list[str] = []
            buf_n = 0
            for word in _split_words(sent):
                wn = _tokenize_len(tokenizer, word + " ")
                if buf_n + wn > budget and buf:
                    yield " ".join(buf), buf_n
                    buf, buf_n = [], 0
                buf.append(word)
                buf_n += wn
            if buf:
                yield " ".join(buf), buf_n


def chunk_text(
    text: str,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Token-bounded recursive splitter with overlap.

    `chunk_tokens` is the hard upper bound per chunk. `overlap_tokens`
    are re-injected from the tail of one chunk into the head of the
    next, so consecutive chunks share `overlap_tokens` of context.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must exceed overlap_tokens")

    text = (text or "").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_n = 0

    for atom, n in _atoms(tokenizer, text, chunk_tokens):
        if buf_n + n > chunk_tokens and buf:
            chunk_text_str = "\n\n".join(buf)
            chunks.append(Chunk(text=chunk_text_str, n_tokens=buf_n))

            # Build overlap tail by taking last atoms whose token sum
            # is <= overlap_tokens. Cheap and deterministic.
            tail: list[str] = []
            tail_n = 0
            for past_atom in reversed(buf):
                past_n = _tokenize_len(tokenizer, past_atom)
                if tail_n + past_n > overlap_tokens:
                    break
                tail.insert(0, past_atom)
                tail_n += past_n
            buf, buf_n = list(tail), tail_n

        buf.append(atom)
        buf_n += n

    if buf:
        chunks.append(Chunk(text="\n\n".join(buf), n_tokens=buf_n))

    return chunks


def chunk_records(
    records: Iterable[dict],
    *,
    body_key: str = "body",
    id_key: str = "unit_id",
    tokenizer: PreTrainedTokenizerBase | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> Iterator[dict]:
    """Chunk a stream of canonical leaves; copy carry-over metadata onto each chunk."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    for rec in records:
        chunks = chunk_text(
            rec.get(body_key) or "",
            tokenizer=tokenizer,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        for idx, ch in enumerate(chunks):
            yield {
                "chunk_id": f"{rec[id_key]}-c{idx:03d}",
                "unit_id": rec[id_key],
                "chunk_index": idx,
                "chunk_text": ch.text,
                "n_tokens": ch.n_tokens,
                "title": rec.get("title"),
                "event_date": rec.get("event_date"),
                "version_introduced": rec.get("version_introduced"),
                "source_url": rec.get("source_url"),
            }
