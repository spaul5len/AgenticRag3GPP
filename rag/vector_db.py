"""Chroma vector database helpers for local source-aware retrieval."""

from __future__ import annotations

import re
from typing import Any

import chromadb

from rag import config
from rag.llm import embed_text


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

    collection = get_collection(collection_name)
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if not text.strip():
            continue

        metadata = _clean_metadata({**base_metadata, **chunk})
        metadata["text"] = text
        vector_id = _vector_id(metadata)

        try:
            embedding = embed_text(text)
        except Exception as exc:
            raise RuntimeError(f"Could not embed chunk '{vector_id}'.") from exc

        ids.append(vector_id)
        documents.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)

    if not ids:
        return 0

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

    return len(ids)


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
