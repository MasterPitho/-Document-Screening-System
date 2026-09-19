import pytest

from app.db.database import build_database
from app.db.models import WatchlistEntry
from app.db.repositories import WatchlistRepository
from app.services.watchlist import normalize_document_number


def _repo(tmp_path):
    db = build_database(f"sqlite:///{(tmp_path / 'wl.db').as_posix()}")
    db.create_all()
    return WatchlistRepository(db)


def test_normalize_removes_separators_and_uppercases():
    assert normalize_document_number("L-898 902c") == "L898902C"
    assert normalize_document_number("") == ""


def test_create_and_get(tmp_path):
    repo = _repo(tmp_path)
    entry = repo.create(name="Subject A", document_number="L-898 902c",
                        reason="Lookout", severity="HIGH")
    assert entry.id is not None
    assert entry.document_number_normalized == "L898902C"
    assert repo.get(entry.id) is not None
    assert repo.get(999999) is None


def test_match_ignores_separators_and_case(tmp_path):
    repo = _repo(tmp_path)
    repo.create(name="Subject A", document_number="L-898 902c",
                reason="Lookout", severity="HIGH")
    match = repo.match("l898902c")
    assert match is not None and match.id is not None
    assert repo.match("X1234567") is None


def test_create_duplicates_allowed(tmp_path):
    repo = _repo(tmp_path)
    a = repo.create(name="A", document_number="DOC123", reason="r", severity="LOW")
    b = repo.create(name="B", document_number="DOC123", reason="r2", severity="LOW")
    assert a.id != b.id


def test_update_only_replaces_provided_fields(tmp_path):
    repo = _repo(tmp_path)
    entry = repo.create(name="A", document_number="DOC123", reason="r", severity="LOW")
    updated = repo.update(entry.id, reason="new reason")
    assert updated.name == "A"
    assert updated.document_number == "DOC123"
    assert updated.reason == "new reason"
    assert updated.severity == "LOW"
    assert repo.update(999999, reason="x") is None


def test_update_resets_normalized_document_number(tmp_path):
    repo = _repo(tmp_path)
    entry = repo.create(name="A", document_number="DOC123", reason="r", severity="LOW")
    updated = repo.update(entry.id, document_number="DOC-456")
    assert updated.document_number_normalized == "DOC456"


def test_delete_returns_bool(tmp_path):
    repo = _repo(tmp_path)
    entry = repo.create(name="A", document_number="DOC123", reason="r", severity="LOW")
    assert repo.delete(entry.id) is True
    assert repo.delete(entry.id) is False


def test_invalid_severity_rejected(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError):
        repo.create(name="A", document_number="DOC123", reason="r", severity="EXTREME")


def test_list_and_search(tmp_path):
    repo = _repo(tmp_path)
    repo.create(name="Alpha", document_number="AA111", reason="first", severity="LOW")
    repo.create(name="Beta", document_number="BB222", reason="second", severity="HIGH")
    total, rows = repo.list()
    assert total == 2 and len(rows) == 2
    total, rows = repo.list(limit=1, offset=0)
    assert total == 2 and len(rows) == 1 and rows[0].name == "Alpha"
    count, found = repo.search("BB22")
    assert count == 1 and found[0].name == "Beta"
    count, found = repo.search("second")
    assert count == 1 and found[0].name == "Beta"


def test_seed_defaults_only_when_empty(tmp_path):
    repo = _repo(tmp_path)
    assert repo.seed_defaults() == 3
    assert repo.seed_defaults() == 0
    total, rows = repo.list()
    assert total == 3
    assert all(row.is_demo_data for row in rows)
    assert all(row.source == "DEMO_DATA_NOT_FOR_OPERATIONAL_USE" for row in rows)


def test_watchlist_entry_importable():
    assert WatchlistEntry.__tablename__ == "watchlist_entries"