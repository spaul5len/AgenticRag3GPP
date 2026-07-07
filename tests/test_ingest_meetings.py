import csv
from pathlib import Path

from rag import config
from rag import ingest_meetings


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(ingest_meetings.METADATA_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_locate_meeting_file_checks_root_tdocs_and_minutes(tmp_path):
    meeting_root = tmp_path / "SA3_123"
    tdocs = meeting_root / "tdocs"
    minutes = meeting_root / "minutes"
    minutes.mkdir(parents=True)
    tdocs.mkdir()
    root_file = meeting_root / "root.txt"
    tdoc_file = tdocs / "tdoc.txt"
    minutes_file = minutes / "minutes.md"
    root_file.write_text("root", encoding="utf-8")
    tdoc_file.write_text("tdoc", encoding="utf-8")
    minutes_file.write_text("minutes", encoding="utf-8")

    assert ingest_meetings.locate_meeting_file(meeting_root, "root.txt") == root_file
    assert ingest_meetings.locate_meeting_file(meeting_root, "tdoc.txt") == tdoc_file
    assert ingest_meetings.locate_meeting_file(meeting_root, "minutes.md") == minutes_file
    assert ingest_meetings.locate_meeting_file(meeting_root, "missing.txt") is None


def test_build_meeting_metadata_preserves_fields_and_never_marks_official(tmp_path):
    metadata = ingest_meetings.build_meeting_metadata(
        {
            "file_name": "S3-230001.txt",
            "tdoc_id": "S3-230001",
            "title": "Proposed security update",
            "source_company": "Example Corp",
            "agenda_item": "7.1",
            "work_item": "FS_SEC",
            "release": "Rel-19",
            "status": "agreed",
            "related_spec": "TS 33.501",
            "meeting_id": "SA3_123",
            "meeting_date": "2026-01-15",
        },
        tmp_path / "SA3_123" / "metadata.csv",
    )

    assert metadata == {
        "doc_type": "meeting_doc",
        "collection_name": "sa3_meeting_documents",
        "title": "Proposed security update",
        "source_company": "Example Corp",
        "meeting_id": "SA3_123",
        "tdoc_id": "S3-230001",
        "agenda_item": "7.1",
        "work_item": "FS_SEC",
        "release": "Rel-19",
        "status": "agreed",
        "related_spec": "TS 33.501",
        "meeting_date": "2026-01-15",
        "remote_url": None,
        "source_type": "local_file",
        "downloaded_at": None,
    }


def test_build_meeting_metadata_defaults_status_and_meeting_id(tmp_path):
    metadata = ingest_meetings.build_meeting_metadata(
        {"file_name": "minutes.md", "status": "", "meeting_id": ""},
        tmp_path / "SA3_124" / "metadata.csv",
    )

    assert metadata["doc_type"] == "meeting_doc"
    assert metadata["status"] == "unknown"
    assert metadata["meeting_id"] == "SA3_124"
    assert metadata["title"] == "minutes.md"


def test_ingest_meetings_dry_run_reads_metadata_and_skips_writes(tmp_path, monkeypatch):
    meeting_root = tmp_path / "meetings" / "SA3_123"
    tdocs = meeting_root / "tdocs"
    minutes = meeting_root / "minutes"
    tdocs.mkdir(parents=True)
    minutes.mkdir()
    (tdocs / "S3-230001.txt").write_text("one two three four five", encoding="utf-8")
    (minutes / "minutes.md").write_text("alpha beta gamma", encoding="utf-8")
    (tdocs / "ignore.csv").write_text("a,b", encoding="utf-8")
    write_metadata(
        meeting_root / "metadata.csv",
        [
            {
                "file_name": "S3-230001.txt",
                "tdoc_id": "S3-230001",
                "title": "TDoc",
                "status": "proposed",
                "meeting_id": "SA3_123",
            },
            {
                "file_name": "minutes.md",
                "title": "Minutes",
                "status": "minutes",
            },
            {"file_name": "missing.docx"},
            {"file_name": "ignore.csv"},
        ],
    )

    calls = {"add_chunks_with_stats": 0, "register_document": 0}
    monkeypatch.setattr(config, "CHUNK_SIZE_WORDS", 3)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 1)
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "metadata.sqlite")
    monkeypatch.setattr(
        ingest_meetings,
        "add_chunks_with_stats",
        lambda *args, **kwargs: calls.__setitem__(
            "add_chunks_with_stats", calls["add_chunks_with_stats"] + 1
        ),
    )
    monkeypatch.setattr(
        ingest_meetings.metadata_db,
        "register_document",
        lambda *args, **kwargs: calls.__setitem__(
            "register_document", calls["register_document"] + 1
        ),
    )

    stats = ingest_meetings.ingest_meetings(
        meetings_dir=tmp_path / "meetings", dry_run=True
    )

    assert stats.metadata_files == 1
    assert stats.scanned == 4
    assert stats.indexed == 2
    assert stats.chunks == 3
    assert stats.chunks_total == 3
    assert stats.skipped_missing == 1
    assert stats.skipped_unsupported == 1
    assert stats.failed == 0
    assert calls == {"add_chunks_with_stats": 0, "register_document": 0}
    assert not Path(config.SQLITE_PATH).exists()


def test_ingest_meetings_writes_chunks_then_registers_document(tmp_path, monkeypatch):
    meeting_root = tmp_path / "meetings" / "SA3_123"
    tdocs = meeting_root / "tdocs"
    tdocs.mkdir(parents=True)
    source = tdocs / "S3-230001.txt"
    source.write_text("one two three four", encoding="utf-8")
    write_metadata(
        meeting_root / "metadata.csv",
        [
            {
                "file_name": "S3-230001.txt",
                "tdoc_id": "S3-230001",
                "title": "Approved proposal",
                "source_company": "Example Corp",
                "status": "approved",
                "meeting_id": "SA3_123",
                "related_spec": "TS 33.501",
            }
        ],
    )

    events: list[tuple[str, object]] = []
    monkeypatch.setattr(config, "CHUNK_SIZE_WORDS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 0)
    monkeypatch.setattr(ingest_meetings.metadata_db, "init_db", lambda: None)
    monkeypatch.setattr(ingest_meetings.metadata_db, "is_already_indexed", lambda path: False)
    monkeypatch.setattr(ingest_meetings.metadata_db, "file_hash", lambda path: "hash-456")

    from rag.vector_db import ChunkWriteStats

    def fake_add_chunks_with_stats(collection_name, chunks, metadata):
        events.append(("add_chunks_with_stats", collection_name))
        assert collection_name == "sa3_meeting_documents"
        assert metadata["doc_type"] == "meeting_doc"
        assert metadata["collection_name"] == "sa3_meeting_documents"
        assert metadata["status"] == "approved"
        assert metadata["tdoc_id"] == "S3-230001"
        assert metadata["source_company"] == "Example Corp"
        assert metadata["related_spec"] == "TS 33.501"
        assert metadata["doc_id"] == "hash-456"
        assert metadata["file_path"] == str(source.resolve())
        return ChunkWriteStats(chunks_total=len(chunks), chunks_indexed=len(chunks))

    def fake_register_document(path, metadata):
        events.append(("register_document", path))
        assert path == source
        assert metadata["doc_type"] == "meeting_doc"
        assert metadata["status"] == "approved"
        return 1

    monkeypatch.setattr(
        ingest_meetings, "add_chunks_with_stats", fake_add_chunks_with_stats
    )
    monkeypatch.setattr(
        ingest_meetings.metadata_db, "register_document", fake_register_document
    )

    stats = ingest_meetings.ingest_meetings(
        meetings_dir=tmp_path / "meetings", dry_run=False
    )

    assert stats.indexed == 1
    assert stats.chunks == 1
    assert stats.chunks_total == 1
    assert stats.chunks_failed == 0
    assert events == [
        ("add_chunks_with_stats", "sa3_meeting_documents"),
        ("register_document", source),
    ]


def test_ingest_meetings_skips_already_indexed_files(tmp_path, monkeypatch):
    meeting_root = tmp_path / "meetings" / "SA3_123"
    tdocs = meeting_root / "tdocs"
    tdocs.mkdir(parents=True)
    source = tdocs / "S3-230001.txt"
    source.write_text("already indexed", encoding="utf-8")
    write_metadata(
        meeting_root / "metadata.csv",
        [{"file_name": "S3-230001.txt", "tdoc_id": "S3-230001"}],
    )

    monkeypatch.setattr(ingest_meetings.metadata_db, "is_already_indexed", lambda path: True)

    stats = ingest_meetings.ingest_meetings(
        meetings_dir=tmp_path / "meetings", dry_run=False
    )

    assert stats.scanned == 1
    assert stats.indexed == 0
    assert stats.skipped_indexed == 1
