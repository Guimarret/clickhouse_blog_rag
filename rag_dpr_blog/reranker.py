"""Cross-encoder reranking for two-stage retrieval.

Pipeline:

  query → BGE dense top-N candidates → cross-encoder rescore → top-k

Why a cross-encoder: the BGE bi-encoder embeds query and passage
independently, so the only signal is cosine similarity in a fixed
1024-d space. A cross-encoder attends jointly over query and passage,
catching nuances (negations, exact entity match, syntactic role) that
the bi-encoder smears out by design.

Cost: O(N) forward passes per query, where N is the rerank window. We
default to N=50; on a 30-query eval set that's 1500 forward passes —
seconds on GPU, ~1 minute on CPU.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "BAAI/bge-reranker-large"


class Reranker:
    """Thin wrapper around sentence-transformers' CrossEncoder.

    Outputs raw logits (higher = more relevant). Not normalized; only
    relative order matters for reranking.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[str],
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return reranker scores aligned with `candidates` (higher = better)."""
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float32)
