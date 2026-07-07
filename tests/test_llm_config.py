from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from rag import config
from rag import llm


def test_config_exports_required_values():
    assert config.BASE_DIR.name == "local_sa3_agentic_rag"
    assert config.DATA_DIR == config.BASE_DIR / "data"
    assert config.SPECS_DIR == config.DATA_DIR / "specs"
    assert config.MEETINGS_DIR == config.DATA_DIR / "meetings"
    assert config.CHROMA_DIR == config.BASE_DIR / "chroma_db"
    assert config.SQLITE_PATH == config.BASE_DIR / "metadata.sqlite"
    assert config.OLLAMA_URL.startswith("http://localhost")
    assert config.CHAT_MODEL
    assert config.EMBED_MODEL
    assert config.SPEC_COLLECTION
    assert config.MEETING_COLLECTION
    assert config.CHUNK_SIZE_WORDS > config.CHUNK_OVERLAP_WORDS > 0
    assert isinstance(config.BASE_DIR, Path)


def test_llm_exports_required_functions():
    assert callable(llm.call_local_llm)
    assert callable(llm.embed_text)


def test_call_local_llm_uses_local_ollama_chat_api(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": "answer"}}
    post = Mock(return_value=response)
    monkeypatch.setattr(llm.requests, "post", post)

    result = llm.call_local_llm("question", system_prompt="system", model="local-chat")

    assert result == "answer"
    post.assert_called_once()
    url = post.call_args.args[0]
    payload = post.call_args.kwargs["json"]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "local-chat"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]


def test_embed_text_uses_local_ollama_embedding_api(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"embedding": [1, 2.5, 3]}
    post = Mock(return_value=response)
    monkeypatch.setattr(llm.requests, "post", post)

    result = llm.embed_text("text", model="local-embed")

    assert result == [1.0, 2.5, 3.0]
    post.assert_called_once()
    url = post.call_args.args[0]
    payload = post.call_args.kwargs["json"]
    assert url == "http://localhost:11434/api/embeddings"
    assert payload == {"model": "local-embed", "prompt": "text"}


def test_ollama_connection_error_is_clear(monkeypatch):
    post = Mock(side_effect=requests.exceptions.ConnectionError("refused"))
    monkeypatch.setattr(llm.requests, "post", post)

    with pytest.raises(RuntimeError, match="Could not connect to local Ollama"):
        llm.call_local_llm("question")


def test_non_local_ollama_url_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_URL", "https://example.com")

    with pytest.raises(RuntimeError, match="Refusing to call non-local Ollama host"):
        llm.embed_text("text")
