from pathlib import Path

from rag import config
from rag import ingest_specs


def test_infer_related_spec_handles_common_3gpp_names():
    assert ingest_specs.infer_related_spec("TS_33_501.pdf") == "TS 33.501"
    assert ingest_specs.infer_related_spec("33501-g80.docx") == "TS 33.501"
    assert ingest_specs.infer_related_spec("33.102.txt") == "TS 33.102"
    assert ingest_specs.infer_related_spec("unrelated.md") is None


def test_build_spec_metadata_marks_official_spec(tmp_path):
    source = tmp_path / "TS_33_501.txt"

    metadata = ingest_specs.build_spec_metadata(source)

    assert metadata["doc_type"] == "official_spec"
    assert metadata["status"] == "official"
    assert metadata["collection_name"] == "official_3gpp_specs"
    assert metadata["related_spec"] == "TS 33.501"
    assert metadata["source_type"] == "local_file"


def test_ingest_specs_dry_run_scans_parses_chunks_without_writes(tmp_path, monkeypatch):
    specs_dir = tmp_path / "specs"
    nested_dir = specs_dir / "nested"
    nested_dir.mkdir(parents=True)
    source = nested_dir / "TS_33_501.txt"
    source.write_text("one two three four five", encoding="utf-8")
    unsupported = specs_dir / "ignore.csv"
    unsupported.write_text("a,b", encoding="utf-8")

    calls = {"add_chunks": 0, "register_document": 0}
    monkeypatch.setattr(config, "CHUNK_SIZE_WORDS", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 1)
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "metadata.sqlite")
    monkeypatch.setattr(
        ingest_specs.add_chunks,
        "__call__",
        lambda *args, **kwargs: calls.__setitem__("add_chunks", calls["add_chunks"] + 1),
    )
    monkeypatch.setattr(
        ingest_specs.metadata_db,
        "register_document",
        lambda *args, **kwargs: calls.__setitem__(
            "register_document", calls["register_document"] + 1
        ),
    )

    stats = ingest_specs.ingest_specs(specs_dir=specs_dir, dry_run=True)

    assert stats.scanned == 2
    assert stats.indexed == 1
    assert stats.chunks == 2
    assert stats.skipped_unsupported == 1
    assert stats.failed == 0
    assert calls == {"add_chunks": 0, "register_document": 0}
    assert not Path(config.SQLITE_PATH).exists()


def test_ingest_specs_skips_already_indexed_files(tmp_path, monkeypatch):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    source = specs_dir / "33501.txt"
    source.write_text("already indexed", encoding="utf-8")

    monkeypatch.setattr(ingest_specs.metadata_db, "is_already_indexed", lambda path: True)

    stats = ingest_specs.ingest_specs(specs_dir=specs_dir, dry_run=False)

    assert stats.scanned == 1
    assert stats.indexed == 0
    assert stats.skipped_indexed == 1


def test_ingest_specs_writes_chunks_then_registers_document(tmp_path, monkeypatch):
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    source = specs_dir / "33501.txt"
    source.write_text("one two three four", encoding="utf-8")

    events: list[tuple[str, object]] = []
    monkeypatch.setattr(config, "CHUNK_SIZE_WORDS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 0)
    monkeypatch.setattr(ingest_specs.metadata_db, "init_db", lambda: None)
    monkeypatch.setattr(ingest_specs.metadata_db, "is_already_indexed", lambda path: False)
    monkeypatch.setattr(ingest_specs.metadata_db, "file_hash", lambda path: "hash-123")

    def fake_add_chunks(collection_name, chunks, metadata):
        events.append(("add_chunks", collection_name))
        assert collection_name == config.SPEC_COLLECTION
        assert metadata["doc_type"] == "official_spec"
        assert metadata["status"] == "official"
        assert metadata["collection_name"] == "official_3gpp_specs"
        assert metadata["related_spec"] == "TS 33.501"
        assert metadata["doc_id"] == "hash-123"
        assert metadata["file_hash"] == "hash-123"
        assert metadata["file_path"] == str(source.resolve())
        return len(chunks)

    def fake_register_document(path, metadata):
        events.append(("register_document", path))
        assert path == source
        assert metadata["doc_type"] == "official_spec"
        assert metadata["status"] == "official"
        return 1

    monkeypatch.setattr(ingest_specs, "add_chunks", fake_add_chunks)
    monkeypatch.setattr(ingest_specs.metadata_db, "register_document", fake_register_document)

    stats = ingest_specs.ingest_specs(specs_dir=specs_dir, dry_run=False)

    assert stats.indexed == 1
    assert stats.chunks == 1
    assert events == [("add_chunks", config.SPEC_COLLECTION), ("register_document", source)]
