"""Local Ollama chat and embedding helpers."""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from rag import config


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CHAT_TIMEOUT_SECONDS = 60
_EMBED_TIMEOUT_SECONDS = 30


def _ensure_local_ollama_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"Ollama URL must use http or https: {base_url}")
    if parsed.hostname not in _LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing to call non-local Ollama host: {base_url}. "
            "Configure Ollama on localhost for this project."
        )
    return base_url.rstrip("/")


def _post_ollama(path: str, payload: dict, timeout: int) -> dict:
    base_url = _ensure_local_ollama_url(config.OLLAMA_URL)
    url = f"{base_url}{path}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to local Ollama. Start Ollama and confirm it is "
            f"listening at {base_url}."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Timed out calling local Ollama at {base_url}. The model may still "
            "be loading or Ollama may be overloaded."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Local Ollama request failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Local Ollama returned a non-JSON response.") from exc


def call_local_llm(prompt: str) -> str:
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    }

    data = _post_ollama("/api/chat", payload, timeout=_CHAT_TIMEOUT_SECONDS)
    return data["message"]["content"]


def embed_text(text: str, model: str | None = None) -> list[float]:
    """Call the local Ollama embedding API and return one embedding vector."""

    payload = {
        "model": model or config.EMBED_MODEL,
        "prompt": text,
    }
    data = _post_ollama("/api/embeddings", payload, timeout=_EMBED_TIMEOUT_SECONDS)

    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not all(
        isinstance(value, (int, float)) for value in embedding
    ):
        raise RuntimeError("Local Ollama embedding response did not include a numeric embedding.")
    return [float(value) for value in embedding]
