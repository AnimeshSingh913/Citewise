"""Blob storage for uploaded originals. Originals are kept so failed documents can be
reprocessed and the index can be rebuilt when the embedding model changes."""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Protocol

from citewise.config import Settings, get_settings


class Storage(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


def make_storage_key(user_id: uuid.UUID, document_id: uuid.UUID) -> str:
    # Keys are built only from server-generated UUIDs, never from the client's filename.
    return f"{user_id}/{document_id}"


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Storage key escapes the storage root")
        return path

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic: readers never see a half-written file

        await asyncio.to_thread(_write)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path(key).unlink, missing_ok=True)


class S3Storage:
    def __init__(self, bucket: str, endpoint_url: str | None, region: str | None):
        try:
            import boto3
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install boto3 to use the S3 storage backend: pip install boto3") from e
        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    async def put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self.client.put_object, Bucket=self.bucket, Key=key, Body=data)

    async def get(self, key: str) -> bytes:
        obj = await asyncio.to_thread(self.client.get_object, Bucket=self.bucket, Key=key)
        return await asyncio.to_thread(obj["Body"].read)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=key)


_storage: Storage | None = None


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        assert settings.s3_bucket
        return S3Storage(settings.s3_bucket, settings.s3_endpoint_url, settings.s3_region)
    return LocalStorage(settings.upload_dir)


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = build_storage(get_settings())
    return _storage


def set_storage(storage: Storage | None) -> None:
    global _storage
    _storage = storage
