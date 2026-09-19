import datetime

from app.services.ledger import (
    GENESIS_PREV_HASH, canonical_json, sha256_hex,
)


def _repo(tmp_path, name):
    from app.db.database import build_database
    from app.db.repositories import LedgerRepository
    db = build_database(f"sqlite:///{(tmp_path / name).as_posix()}")
    db.create_all()
    return LedgerRepository(db)


def test_canonical_json_is_key_sorted():
    a = canonical_json({"b": 1, "a": {"z": 2, "y": 1}})
    b = canonical_json({"a": {"y": 1, "z": 2}, "b": 1})
    assert a == b


def test_sha256_shapec():
    assert sha256_hex("abc") == sha256_hex("abc")
    assert len(sha256_hex("anything")) == 64


def test_seed_genesis_only_once(tmp_path):
    repo = _repo(tmp_path, "l.db")
    g1 = repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1, 0, 0, 0))
    assert g1 is not None
    assert g1.entry_index == 0
    assert g1.entry_type == "GENESIS"
    assert g1.prev_hash == GENESIS_PREV_HASH
    assert repo.seed_genesis() is None
    assert repo.head().entry_index == 0


def test_append_chains_and_verifies(tmp_path):
    repo = _repo(tmp_path, "l2.db")
    repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1))
    e1 = repo.append(entry_type="SCREENING", payload={"risk_score": 10},
                     request_id="A", created_at=datetime.datetime(2026, 9, 2))
    e2 = repo.append(entry_type="SCREENING", payload={"risk_score": 95},
                     request_id="B", created_at=datetime.datetime(2026, 9, 3))
    assert e2.prev_hash == e1.entry_hash
    assert repo.head().entry_index == 2
    check = repo.verify()
    assert check["valid"] is True and check["checked"] == 3 and check["broken_at"] is None


def test_verify_detects_tampered_payload(tmp_path):
    from sqlalchemy import update

    from app.db.models import LedgerEntry
    path = f"sqlite:///{(tmp_path / 'l3.db').as_posix()}"
    from app.db.database import build_database
    from app.db.repositories import LedgerRepository
    db = build_database(path)
    db.create_all()
    repo = LedgerRepository(db)
    repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1))
    repo.append(entry_type="SCREENING", payload={"risk_score": 10}, request_id="A",
                created_at=datetime.datetime(2026, 9, 2))
    with db.session() as session:
        session.execute(update(LedgerEntry).where(LedgerEntry.entry_index == 1)
                        .values(payload={"risk_score": 999}))
        session.commit()
    check = repo.verify()
    assert check["valid"] is False and check["broken_at"] == 1


def test_list_get_offsets(tmp_path):
    repo = _repo(tmp_path, "l4.db")
    repo.seed_genesis(created_at=datetime.datetime(2026, 9, 1))
    for i in range(5):
        repo.append(entry_type="X", payload={"i": i},
                    created_at=datetime.datetime(2026, 9, 2, i))
    total, rows = repo.list(limit=2, offset=1)
    assert total == 6
    assert [r.entry_index for r in rows] == [4, 3]
    assert repo.get(2).payload == {"i": 1}
    assert repo.get(99) is None