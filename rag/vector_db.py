"""Chroma vector database helpers for local source-aware retrieval."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import chromadb

from rag import config
from rag.llm import embed_text


EMBED_RETRY_BACKOFF_SECONDS = (1, 2, 4)


@dataclass
class ChunkWriteStats:
    chunks_total: int = 0
    chunks_indexed: int = 0
    chunks_failed: int = 0


def get_client():
    """Return a persistent Chroma client rooted at the configured directory."""

    try:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    except Exception as exc:
        raise RuntimeError(f"Could not initialize Chroma at {config.CHROMA_DIR}.") from exc


def get_collection(name: str):
    """Return an existing Chroma collection, creating it if needed."""

    if not name or not name.strip():
        raise ValueError("collection name must not be empty.")

    try:
        return get_client().get_or_create_collection(name=name)
    except Exception as exc:
        raise RuntimeError(f"Could not open Chroma collection '{name}'.") from exc


def add_chunks(
    collection_name: str,
    chunks: list[dict[str, Any]],
    base_metadata: dict[str, Any],
) -> int:
    """Embed and upsert non-empty chunks into a Chroma collection.

    Returns the number of chunks written.
    """

    return add_chunks_with_stats(collection_name, chunks, base_metadata).chunks_indexed


def add_chunks_with_stats(
    collection_name: str,
    chunks: list[dict[str, Any]],
    base_metadata: dict[str, Any],
) -> ChunkWriteStats:
    """Embed and upsert chunks while skipping chunks that fail embedding.

    Returns per-document chunk totals so callers can report partial indexing.
    """

    collection = get_collection(collection_name)
    stats = ChunkWriteStats()
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if not text.strip():
            continue
        stats.chunks_total += 1

        metadata = _clean_metadata({**base_metadata, **chunk})
        metadata["text"] = text
        vector_id = _vector_id(metadata)

        embedding = _embed_text_with_retries(text)
        if embedding is None:
            stats.chunks_failed += 1
            _print_chunk_embedding_warning(metadata, text)
            continue

        ids.append(vector_id)
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)

    if not ids:
        return stats

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not write {len(ids)} chunks to Chroma collection '{collection_name}'."
        ) from exc

    stats.chunks_indexed = len(ids)
    return stats


def search_collection(
    collection_name: str,
    query: str,
    k: int = 8,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search one Chroma collection and return normalized source-aware results."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if not query.strip():
        return []

    collection = get_collection(collection_name)
    try:
        query_embedding = embed_text(query)
    except Exception as exc:
        raise RuntimeError("Could not embed search query.") from exc

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(f"Could not search Chroma collection '{collection_name}'.") from exc

    documents = _first_result_list(results, "documents")
    metadatas = _first_result_list(results, "metadatas")
    distances = _first_result_list(results, "distances")

    normalized: list[dict[str, Any]] = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = distances[index] if index < len(distances) else None
        normalized.append(
            {
                "text": text or "",
                "metadata": metadata,
                "distance": distance,
                "source": _source_from_metadata(metadata),
                "page": metadata.get("page"),
            }
        )
    return normalized


def _vector_id(metadata: dict[str, Any]) -> str:
    doc_type = metadata.get("doc_type") or "unknown_doc_type"
    doc_id = metadata.get("doc_id") or metadata.get("document_id") or "unknown_doc"
    page = metadata.get("page") or "unknown_page"
    chunk_id = metadata.get("chunk_id") or "unknown_chunk"
    return "::".join(_slug_part(value) for value in (doc_type, doc_id, page, chunk_id))


def _embed_text_with_retries(text: str) -> list[float] | None:
    last_error: Exception | None = None
    for attempt in range(len(EMBED_RETRY_BACKOFF_SECONDS) + 1):
        try:
            return embed_text(text)
        except Exception as exc:
            last_error = exc
            if attempt >= len(EMBED_RETRY_BACKOFF_SECONDS):
                _embed_text_with_retries.last_error = exc  # type: ignore[attr-defined]
                return None
            time.sleep(EMBED_RETRY_BACKOFF_SECONDS[attempt])
    _embed_text_with_retries.last_error = last_error  # type: ignore[attr-defined]
    return None


def _print_chunk_embedding_warning(metadata: dict[str, Any], text: str) -> None:
    error = getattr(_embed_text_with_retries, "last_error", None)
    preview = text[:300].replace("\n", " ")
    print(
        "Warning: failed to embed chunk after retries; skipping chunk. "
        f"file_path={metadata.get('file_path') or metadata.get('source') or 'unknown'} "
        f"page={metadata.get('page') or 'unknown'} "
        f"chunk_id={metadata.get('chunk_id') or 'unknown'} "
        f"word_count={len(text.split())} "
        f"char_count={len(text)} "
        f"preview={preview!r} "
        f"error={error}"
    )


def _slug_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _first_result_list(results: dict[str, Any], key: str) -> list[Any]:
    values = results.get(key) or []
    if not values:
        return []
    first = values[0]
    return first if isinstance(first, list) else []


def _source_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("source", "file_path", "remote_url", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""
