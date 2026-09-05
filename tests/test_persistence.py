"""Repository-layer and DB-backed API tests (pagination, filters, stats,
factors, duplicate request_id handling, session recovery)."""

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.db.database import build_database, utcnow_naive
from app.db.models import Screening
from app.db.repositories import (
    AuditLogRepository,
    DuplicateRequestError,
    PersistenceError,
    ScreeningRepository,
    UserRepository,
    AuthTokenRepository,
)
from app.main import create_app
from app.services import mrz as mrz_mod
from app.services.face_recognition import DummyBackend, ModelManager


class CleanTampering:
    def analyze(self, image_bytes):
        return {"status": "CLEAN", "score": 0.0, "confidence": 0.0,
                "signals": {}, "suspicious_regions": [], "explanation": []}


def _factor(name, weight, detail=""):
    return {"factor": name, "weight": weight, "detail": detail}


def _create(repo: ScreeningRepository, *, request_id=None, **overrides) -> Screening:
    kwargs = dict(
        request_id=request_id or uuid.uuid4().hex,
        processing_time_ms=120,
        document_type="PASSPORT",
        mrz_status="VALID",
        face_status="MATCH",
        face_similarity=0.92,
        tampering_status="CLEAN",
        tampering_score=0.0,
        risk_score=10,
        risk_level="LOW_RISK",
        decision="CLEARED",
        status_color="GREEN",
        module_states={"mrz": "PASS", "face": "PASS", "tampering": "PASS"},
        factor_list=[_factor("MRZ_OK", 0), _factor("FACE_MATCH", 0)],
        mrz_source="form",
        user_id=None,
    )
    kwargs.update(overrides)
    return repo.create(**kwargs)


def _app():
    settings = create_app().state.settings
    manager = ModelManager(settings, backend=DummyBackend([]))
    app = create_app(model_manager=manager)
    app.state.tampering = CleanTampering()
    return app


def _register_and_login(client, username):
    password = "SuperSecret123!"
    client.post("/api/v1/auth/register", json={
        "username": username, "email": f"{username}@example.com",
        "full_name": "Test Officer", "password": password,
    })
    login = client.post("/api/v1/auth/login",
                        json={"username": username, "password": password})
    return login.json()["token"]


# ---------------------------------------------------------------------------
# Repository: create & read
# ---------------------------------------------------------------------------

def test_repo_create_and_read_by_id_and_request_id():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        repo = ScreeningRepository(db)
        created = _create(repo)
        assert created.id is not None
        assert created.risk_level == "LOW_RISK"

        by_id = repo.get(created.id)
        assert by_id is not None and by_id.request_id == created.request_id

        by_rid = repo.get_by_request_id(created.request_id)
        assert by_rid is not None and by_rid.id == created.id

        assert repo.get(2_000_000_000) is None
        assert repo.get_by_request_id("nope-does-not-exist") is None
    finally:
        db.dispose()


def test_repo_duplicate_request_id_raises():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        repo = ScreeningRepository(db)
        rid = uuid.uuid4().hex
        _create(repo, request_id=rid)
        with pytest.raises(DuplicateRequestError):
            _create(repo, request_id=rid)
    finally:
        db.dispose()


def test_repo_recovers_after_failed_create():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        repo = ScreeningRepository(db)
        rid = uuid.uuid4().hex
        _create(repo, request_id=rid)
        with pytest.raises(DuplicateRequestError):
            _create(repo, request_id=rid)
        # A subsequent, unrelated create must still succeed (clean sessions).
        later = _create(repo)
        assert later.id is not None
    finally:
        db.dispose()


def test_repo_factor_normalization():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        repo = ScreeningRepository(db)
        created = _create(
            repo,
            factor_list=[
                _factor("TAMPERING_SUSPECTED", 40, "Heuristic signal"),
                _factor("MRZ_LOW_CONFIDENCE", 10, ""),
            ],
        )
        rows = repo.list_factors(created.id)
        assert len(rows) == 2
        tamper = next(r for r in rows if r.factor_name == "TAMPERING_SUSPECTED")
        assert tamper.severity == "HIGH"
        assert tamper.weight == 40
        assert tamper.description == "Heuristic signal"
        assert tamper.screening_id == created.id
        low = next(r for r in rows if r.factor_name == "MRZ_LOW_CONFIDENCE")
        assert low.severity == "LOW"
    finally:
        db.dispose()


# ---------------------------------------------------------------------------
# Repository: pagination, filters, stats
# ---------------------------------------------------------------------------

def test_repo_list_pagination_and_filters():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        repo = ScreeningRepository(db)
        for i in range(5):
            _create(repo, risk_score=10, risk_level="LOW_RISK",
                    decision="CLEARED")
        _create(repo, risk_score=80, risk_level="HIGH_RISK",
                decision="HIGH_RISK_REVIEW_REQUIRED")
        _create(repo, risk_score=40, risk_level="MEDIUM_RISK",
                decision="SECONDARY_INSPECTION_REQUIRED")

        total, rows = repo.list(limit=3, offset=0)
        assert total >= 7
        assert len(rows) == 3

        total, rows = repo.list(decision="CLEARED")
        assert len(rows) == total >= 5
        assert all(r.decision == "CLEARED" for r in rows)

        total, rows = repo.list(risk_level="HIGH_RISK")
        assert total >= 1 and all(r.risk_level == "HIGH_RISK" for r in rows)

        full_total, _ = repo.list()
        future = utcnow_naive()
        total_future, _ = repo.list(date_from=future)
        assert total_future == 0
        past = utcnow_naive()
        import datetime
        past = past - datetime.timedelta(days=1)
        total_past, _ = repo.list(date_from=past)
        assert total_past == full_total
    finally:
        db.dispose()


def test_repo_stats_and_summary():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        repo = ScreeningRepository(db)
        _create(repo, decision="CLEARED", risk_level="LOW_RISK", risk_score=10,
                mrz_status="VALID", face_status="MATCH", tampering_status="CLEAN")
        _create(repo, decision="HIGH_RISK_REVIEW_REQUIRED", risk_level="HIGH_RISK",
                risk_score=90, mrz_status="INVALID", face_status="MISMATCH",
                tampering_status="SUSPICIOUS")

        stats = repo.stats()
        assert stats["total"] >= 2
        assert stats["cleared"] >= 1
        assert stats["high_risk"] >= 1
        assert stats["mrz_failures"] >= 1
        assert stats["face_mismatches"] >= 1
        assert stats["suspicious_tampering"] >= 1
        assert stats["by_decision"]["CLEARED"] >= 1
        assert stats["by_risk_level"]["HIGH_RISK"] >= 1

        summary = repo.summary()
        assert summary["total"] == stats["total"]
        assert summary["avg_processing_time_ms"] > 0
        assert summary["by_decision"].get("CLEARED", 0) >= 1
    finally:
        db.dispose()


# ---------------------------------------------------------------------------
# Repository: audit + users + tokens
# ---------------------------------------------------------------------------

def test_audit_log_repository_record():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        screening = _create(ScreeningRepository(db))
        event = AuditLogRepository(db).record(
            screening_id=screening.id, event_type="screening.completed",
            request_id=screening.request_id, message="decision=CLEARED",
        )
        assert event.id is not None
        assert event.screening_id == screening.id
    finally:
        db.dispose()


def test_user_and_token_repositories():
    settings = Settings.from_env()
    db = build_database(settings.database_url)
    db.create_all(fail_silently=False)
    try:
        users = UserRepository(db)
        unique = f"repo_{uuid.uuid4().hex[:8]}"
        user = users.create(username=unique, email=f"{unique}@example.com",
                            full_name="Repo", role="officer", password_hash="x")
        assert users.get_by_username(unique).id == user.id
        assert users.get_by_email(f"{unique}@example.com").id == user.id
        assert users.get_by_id(user.id).id == user.id
        with pytest.raises(PersistenceError):
            users.create(username=unique, email="other@example.com",
                         full_name="", role="officer", password_hash="x")

        tokens = AuthTokenRepository(db)
        tok = tokens.create(user_id=user.id, token_hash="abc123hash",
                            expires_at=utcnow_naive())
        assert tok.id is not None
        found, owner = users.find_by_token_hash("abc123hash")
        assert found is not None and owner.id == user.id
        tokens.delete_by_hash("abc123hash")
        assert users.find_by_token_hash("abc123hash") == (None, None)
    finally:
        db.dispose()


# ---------------------------------------------------------------------------
# DB-backed API: aggregates, filters, factors, duplicates
# ---------------------------------------------------------------------------

def test_stats_filters_and_factors_endpoints():
    client = TestClient(_app())
    token = _register_and_login(client, f"api_{uuid.uuid4().hex[:8]}")
    headers = {"Authorization": f"Bearer {token}"}

    repo = client.app.state.screening_repo
    rid = uuid.uuid4().hex
    record = repo.create(
        request_id=rid, processing_time_ms=95, document_type="PASSPORT",
        mrz_status="INVALID", face_status="MISMATCH", face_similarity=0.1,
        tampering_status="SUSPICIOUS", tampering_score=90.0,
        risk_score=90, risk_level="HIGH_RISK",
        decision="HIGH_RISK_REVIEW_REQUIRED", status_color="RED",
        module_states={"mrz": "FAIL", "face": "FAIL", "tampering": "FAIL"},
        factor_list=[_factor("TAMPERING_SUSPECTED", 40, "ela"),
                     _factor("FACE_MISMATCH", 35, "cosine")],
        mrz_source="ocr",
    )
    _create(repo, risk_score=10, risk_level="LOW_RISK", decision="CLEARED")

    stats = client.get("/api/v1/stats", headers=headers)
    assert stats.status_code == 200
    body = stats.json()
    assert body["total"] >= 2
    assert body["high_risk"] >= 1
    assert body["mrz_failures"] >= 1
    assert body["face_mismatches"] >= 1
    assert body["suspicious_tampering"] >= 1

    listing = client.get(
        "/api/v1/screenings?decision=CLEARED&limit=10", headers=headers)
    assert listing.status_code == 200
    listed = listing.json()
    assert listed["total"] >= 1
    assert listed["decision"] == "CLEARED"
    assert all(r["decision"] == "CLEARED" for r in listed["records"])

    filtered = client.get(
        "/api/v1/screenings?risk_level=HIGH_RISK", headers=headers)
    assert filtered.status_code == 200
    assert filtered.json()["total"] >= 1
    assert all(r["risk_level"] == "HIGH_RISK" for r in filtered.json()["records"])

    # Factor rows are normalized and connected to the screening.
    factors = client.get(f"/api/v1/screenings/{record.id}/factors", headers=headers)
    assert factors.status_code == 200
    names = {f["factor_name"] for f in factors.json()}
    assert {"TAMPERING_SUSPECTED", "FACE_MISMATCH"} <= names
    assert any(f["severity"] == "HIGH" and f["description"] == "ela"
               for f in factors.json())

    # Not found for unknown id.
    assert client.get("/api/v1/screenings/999999999/factors",
                      headers=headers).status_code == 404

    # Unauthenticated access is rejected.
    assert client.get("/api/v1/stats").status_code == 401
    assert client.get(f"/api/v1/screenings/{record.id}").status_code == 401


def test_screen_with_reused_request_id_returns_409(monkeypatch):
    monkeypatch.setattr(mrz_mod, "extract_mrz_from_image",
                        lambda doc, settings: {
                            "detected": False, "source": "ocr",
                            "status": "NOT_DETECTED", "confidence": 0.0,
                            "module_state": "NOT_AVAILABLE"})
    client = TestClient(_app())
    token = _register_and_login(client, f"dup_{uuid.uuid4().hex[:8]}")
    headers = {"Authorization": f"Bearer {token}"}
    rid = uuid.uuid4().hex

    def _post():
        return client.post(
            "/api/v1/screen",
            headers={"X-Request-ID": rid, **headers},
            files={"document_image": ("doc.jpg", jpeg_bytes(), "image/jpeg")},
        )

    first = _post()
    assert first.status_code == 200
    assert first.json()["persistence"]["status"] == "stored"

    second = _post()
    assert second.status_code == 409
    body = second.json()
    assert body["success"] is False
    assert body["error"]["code"] == "CONFLICT"
    assert body["request_id"] == rid

    # The duplicate attempt must not have created a second record.
    listing = client.get("/api/v1/screenings", headers=headers)
    matching = [r for r in listing.json()["records"] if r["request_id"] == rid]
    assert len(matching) == 1


def jpeg_bytes(width=800, height=500):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()