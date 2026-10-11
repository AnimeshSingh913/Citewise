"""Local embeddings: a dense model for meaning and BM25 sparse vectors for exact terms.
Both run on CPU in-process via fastembed, so no embedding API key is needed."""

import asyncio
import hashlib
import math
import re
import threading
from dataclasses import dataclass
from typing import Protocol

from citewise.config import get_settings


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder(Protocol):
    name: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]: ...
    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]: ...


class FastEmbedEmbedder:
    def __init__(self, dense_model: str, sparse_model: str, batch_size: int):
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self.batch_size = batch_size
        self.name = f"{dense_model}+{sparse_model}"
        self._dense = None
        self._sparse = None
        self._lock = threading.Lock()
        self.dimension = self._dimension_for(dense_model)

    @staticmethod
    def _dimension_for(model: str) -> int:
        from fastembed import TextEmbedding

        for info in TextEmbedding.list_supported_models():
            if info["model"] == model:
                return int(info["dim"])
        raise ValueError(f"Unknown fastembed dense model: {model}")

    def _models(self):
        with self._lock:
            if self._dense is None:
                from fastembed import SparseTextEmbedding, TextEmbedding

                self._dense = TextEmbedding(self.dense_model_name)
                self._sparse = SparseTextEmbedding(self.sparse_model_name)
        return self._dense, self._sparse

    def _embed_documents_sync(self, texts: list[str]):
        dense_model, sparse_model = self._models()
        dense = [v.tolist() for v in dense_model.embed(texts, batch_size=self.batch_size)]
        sparse = [
            SparseVec(indices=s.indices.tolist(), values=s.values.tolist())
            for s in sparse_model.embed(texts, batch_size=self.batch_size)
        ]
        return dense, sparse

    def _embed_query_sync(self, text: str):
        dense_model, sparse_model = self._models()
        dense = next(iter(dense_model.query_embed(text))).tolist()
        s = next(iter(sparse_model.query_embed(text)))
        return dense, SparseVec(indices=s.indices.tolist(), values=s.values.tolist())

    async def embed_documents(self, texts):
        return await asyncio.to_thread(self._embed_documents_sync, texts)

    async def embed_query(self, text):
        return await asyncio.to_thread(self._embed_query_sync, text)


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests and fully offline demos.
    Quality is bag-of-words level; use fastembed for anything real."""

    name = "hashing-v1"

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    @staticmethod
    def _bucket(token: str, size: int) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % size

    def _dense(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            vec[self._bucket(token, self.dimension)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _sparse(self, text: str) -> SparseVec:
        counts: dict[int, float] = {}
        for token in _TOKEN.findall(text.lower()):
            idx = self._bucket(token, 2**31 - 1)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        indices = sorted(counts)
        return SparseVec(indices=indices, values=[counts[i] for i in indices])

    async def embed_documents(self, texts):
        return [self._dense(t) for t in texts], [self._sparse(t) for t in texts]

    async def embed_query(self, text):
        return self._dense(text), self._sparse(text)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        s = get_settings()
        if s.embedding_backend == "hashing":
            _embedder = HashingEmbedder()
        else:
            _embedder = FastEmbedEmbedder(s.dense_model, s.sparse_model, s.embedding_batch_size)
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    global _embedder
    _embedder = embedder
