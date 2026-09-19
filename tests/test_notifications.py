from datetime import datetime, timezone


def test_notification_repository_flow(tmp_path):
    from app.db.database import build_database
    from app.db.repositories import NotificationRepository, ScreeningRepository
    db = build_database(f"sqlite:///{(tmp_path / 'n.db').as_posix()}")
    db.create_all()
    screen_repo = ScreeningRepository(db)
    notif_repo = NotificationRepository(db)

    low = screen_repo.create(
        request_id="LOW-001", processing_time_ms=10, document_type="PASSPORT",
        mrz_status="VALID", face_status="MATCH", face_similarity=0.9,
        tampering_status="CLEAN", tampering_score=0.0, liveness_status="LIVE",
        risk_score=5, risk_level="LOW_RISK", decision="CLEARED", status_color="GREEN",
        module_states={}, factor_list=[], mrz_source="form")
    assert notif_repo.create_for_screening(low) is None

    high = screen_repo.create(
        request_id="HIGH-001", processing_time_ms=20, document_type="PASSPORT",
        mrz_status="INVALID", face_status="MISMATCH", face_similarity=0.1,
        tampering_status="SUSPICIOUS", tampering_score=90.0,
        risk_score=95, risk_level="HIGH_RISK",
        decision="HIGH_RISK_REVIEW_REQUIRED", status_color="RED",
        module_states={}, factor_list=[], mrz_source="form")
    note = notif_repo.create_for_screening(high)
    assert note is not None and note.read is False
    again = notif_repo.create_for_screening(high)
    assert again is None  # idempotent

    total, rows = notif_repo.list(unread_only=True)
    assert total == 1 and rows[0].id == note.id
    marked = notif_repo.mark_read(note.id)
    assert marked is not None and marked.read is True
    assert notif_repo.mark_read(999999) is None
    total, _ = notif_repo.list(unread_only=True)
    assert total == 0


def test_notifications_endpoint_flows(app_and_client):
    _, client = app_and_client
    client.post("/api/v1/auth/register", json={
        "username": "notif_o", "email": "notif@example.com",
        "full_name": "N O", "password": "Password123!",
    })
    token = client.post("/api/v1/auth/login", json={
        "username": "notif_o", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    listing = client.get("/api/v1/notifications", headers=headers)
    assert listing.status_code == 200 and listing.json() == []

    app = app_and_client[0]
    screen_repo = app.state.screening_repo
    notif_repo = app.state.notification_repo
    screen_repo.create(
        request_id="API-HIGH-1", processing_time_ms=10, document_type="PASSPORT",
        mrz_status="INVALID", face_status="MISMATCH", face_similarity=0.0,
        tampering_status="SUSPICIOUS", tampering_score=90.0,
        risk_score=95, risk_level="HIGH_RISK",
        decision="HIGH_RISK_REVIEW_REQUIRED", status_color="RED",
        module_states={}, factor_list=[], mrz_source="form",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None))
    notif_repo.create_for_screening(screen_repo.get_by_request_id("API-HIGH-1"))

    items = client.get("/api/v1/notifications?unread_only=true", headers=headers).json()
    assert len(items) == 1
    nid = items[0]["id"]
    patched = client.patch(f"/api/v1/notifications/{nid}/read", headers=headers)
    assert patched.status_code == 200 and patched.json()["read"] is True
    assert client.get("/api/v1/notifications?unread_only=true",
                      headers=headers).json() == []
    assert client.patch("/api/v1/notifications/999999/read",
                        headers=headers).status_code == 404
    assert client.get("/api/v1/notifications",
                      headers={}).status_code == 401