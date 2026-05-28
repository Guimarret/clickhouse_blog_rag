"""Encoder wrapper — bge-large-en-v1.5 with asymmetric query/passage prefixes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Defensive VRAM cap. The RX 6700 XT (12 GB) is unreliable when PyTorch
# and Ollama both push the card hard simultaneously — saw a system-wide
# amdgpu hang on 2026-05-28 from rocBLAS aborting under contention.
# Capping PyTorch to 5 GB leaves ~7 GB headroom for whatever Ollama is
# holding, and ~10k vocab words at 1024-d easily fit.
# Override with RAG_DPR_BLOG_TORCH_MEM_FRACTION=0.X if needed.
_GPU_MEM_FRACTION = float(os.environ.get("RAG_DPR_BLOG_TORCH_MEM_FRACTION", "0.42"))


def _apply_vram_cap() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_per_process_memory_fraction(_GPU_MEM_FRACTION, 0)
    except (RuntimeError, ValueError):
        # ROCm builds occasionally lack this API surface; non-fatal.
        pass


_apply_vram_cap()


class Encoder:
    """Thin wrapper around SentenceTransformer with passage/query asymmetry.

    BGE expects:
      - passages: encoded with no prefix
      - queries:  prefixed with the canonical instruction string above

    Vectors are L2-normalized so that downstream cosine == dot product.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Sequence[str],
        kind: Literal["passage", "query"] = "passage",
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if kind == "query" and self.model_name.startswith("BAAI/bge"):
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32, copy=False)
