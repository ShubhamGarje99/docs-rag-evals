"""Local embedding models (sentence-transformers) plus a tiny hashing embedder for tests."""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

# BGE English v1.5 models are trained with this instruction on the *query* side
# only (passages are embedded as-is). See the BAAI/bge-small-en-v1.5 model card.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def query_instruction_for(model_name: str) -> str:
    n = model_name.lower()
    if "bge-m3" in n:
        return ""  # bge-m3 needs no instruction
    if "bge" in n and "-en" in n:
        return BGE_QUERY_INSTRUCTION
    if "e5" in n:
        return "query: "
    return ""


def passage_prefix_for(model_name: str) -> str:
    return "passage: " if "e5" in model_name.lower() else ""


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch import problems
        pass
    return "cpu"


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """L2-normalised dense embeddings from a local sentence-transformers model."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        self.name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction_for(model_name)
        self.passage_prefix = passage_prefix_for(model_name)
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_instruction + text])[0]


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Only for unit tests and smoke tests: no model download, no semantics beyond
    word overlap. Never used for reported numbers.
    """

    name = "hashing-test-embedder"

    def __init__(self, dim: int = 1024):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        from .golden import STOPWORDS

        v = np.zeros(self._dim, dtype=np.float32)
        for tok in re.findall(r"\w\w+", text.lower()):
            if tok in STOPWORDS:
                continue
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % self._dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self._dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
