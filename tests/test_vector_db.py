from rag import vector_db


class FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upserts.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )


def test_add_chunks_with_stats_skips_one_chunk_after_embedding_retries(
    monkeypatch, capsys
):
    collection = FakeCollection()
    calls = {"bad": 0}
    monkeypatch.setattr(vector_db, "get_collection", lambda collection_name: collection)
    monkeypatch.setattr(vector_db, "EMBED_RETRY_BACKOFF_SECONDS", (0, 0, 0))

    def fake_embed_text(text):
        if "bad chunk" in text:
            calls["bad"] += 1
            raise RuntimeError("ollama 500")
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr(vector_db, "embed_text", fake_embed_text)

    stats = vector_db.add_chunks_with_stats(
        "test_collection",
        [
            {"chunk_id": "chunk-0", "page": 1, "text": "good chunk one"},
            {"chunk_id": "chunk-1", "page": 2, "text": "bad chunk hex ECIES SUCI"},
            {"chunk_id": "chunk-2", "page": 3, "text": "good chunk two"},
        ],
        {
            "doc_type": "official_spec",
            "doc_id": "doc-1",
            "file_path": "/tmp/TS_33_501.txt",
        },
    )

    assert calls["bad"] == 4
    assert stats.chunks_total == 3
    assert stats.chunks_indexed == 2
    assert stats.chunks_failed == 1
    assert len(collection.upserts) == 1
    assert collection.upserts[0]["documents"] == ["good chunk one", "good chunk two"]

    warning = capsys.readouterr().out
    assert "Warning: failed to embed chunk after retries" in warning
    assert "file_path=/tmp/TS_33_501.txt" in warning
    assert "page=2" in warning
    assert "chunk_id=chunk-1" in warning
    assert "word_count=5" in warning
    assert "char_count=24" in warning
    assert "bad chunk hex ECIES SUCI" in warning
    assert "ollama 500" in warning
