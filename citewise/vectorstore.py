"""Qdrant hybrid index: one point per child chunk, with a dense vector and a BM25 sparse
vector. Search fuses both rankings server-side with Reciprocal Rank Fusion."""

import re
import uuid
import warnings
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from citewise.config import get_settings
from citewise.embeddings import Embedder, SparseVec

DENSE = "dense"
SPARSE = "bm25"


@dataclass
class ChildPoint:
    user_id: uuid.UUID
    document_id: uuid.UUID
    parent_id: uuid.UUID
    text: str


@dataclass
class Hit:
    parent_id: uuid.UUID
    document_id: uuid.UUID
    score: float


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, base_collection: str, embedder: Embedder):
        self.client = client
        self.embedder = embedder
        # One collection per embedding model, so switching models never mixes vector spaces.
        slug = re.sub(r"[^a-z0-9]+", "_", embedder.name.lower()).strip("_")
        self.collection = f"{base_collection}__{slug}"[:255]

    async def ensure_collection(self) -> None:
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE: models.VectorParams(size=self.embedder.dimension, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # embedded Qdrant warns that indexes are a no-op
            for field in ("user_id", "document_id"):
                await self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
                )

    async def upsert(self, points: list[ChildPoint]) -> None:
        if not points:
            return
        dense, sparse = await self.embedder.embed_documents([p.text for p in points])
        await self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={DENSE: d, SPARSE: _sparse(s)},
                    payload={
                        "user_id": str(p.user_id),
                        "document_id": str(p.document_id),
                        "parent_id": str(p.parent_id),
                    },
                )
                for p, d, s in zip(points, dense, sparse, strict=True)
            ],
            wait=True,
        )

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_match("document_id", [document_id])),
            wait=True,
        )

    async def search(
        self,
        query: str,
        user_id: uuid.UUID,
        document_ids: list[uuid.UUID] | None,
        limit: int,
    ) -> list[Hit]:
        dense, sparse = await self.embedder.embed_query(query)
        conditions = [_condition("user_id", [user_id])]
        if document_ids:
            conditions.append(_condition("document_id", document_ids))
        flt = models.Filter(must=conditions)
        result = await self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, filter=flt, limit=limit * 2),
                models.Prefetch(query=_sparse(sparse), using=SPARSE, filter=flt, limit=limit * 2),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return [
            Hit(
                parent_id=uuid.UUID(p.payload["parent_id"]),
                document_id=uuid.UUID(p.payload["document_id"]),
                score=p.score,
            )
            for p in result.points
        ]


def _sparse(vec: SparseVec) -> models.SparseVector:
    return models.SparseVector(indices=vec.indices, values=vec.values)


def _condition(key: str, values: list[uuid.UUID]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=[str(v) for v in values]))


def _match(key: str, values: list[uuid.UUID]) -> models.Filter:
    return models.Filter(must=[_condition(key, values)])


_store: VectorStore | None = None


def build_client() -> AsyncQdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    if s.qdrant_path == ":memory:":
        return AsyncQdrantClient(location=":memory:")
    return AsyncQdrantClient(path=s.qdrant_path)


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        from citewise.embeddings import get_embedder

        _store = VectorStore(build_client(), get_settings().qdrant_collection, get_embedder())
    return _store


def set_vector_store(store: VectorStore | None) -> None:
    global _store
    _store = store
