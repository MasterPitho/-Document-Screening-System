import json

import pytest

from app.config import Settings
from app.services.risk_engine import RiskEngine, UnknownRiskFactorError

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"

GREEN = {"status": "CLEAN", "score": 0.0}
FACE = {"status": "MATCH", "similarity_score": 0.9}
MRZ_OK = {"status": "VALID", "detected": True,
          "data": {"checks": {}, "is_expired": False}}
LIVE = {"liveness_status": "LIVE"}


def _jpeg(width=800, height=500):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


def _auth_token(client):
    client.post("/api/v1/auth/register", json={
        "username": "wl_int", "email": "wlint@example.com",
        "full_name": "WL Int", "password": "Password123!",
    })
    token = client.post("/api/v1/auth/login", json={
        "username": "wl_int", "password": "Password123!",
    }).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_extra_factors_add_to_score():
    engine = RiskEngine(Settings.from_env())
    result = engine.evaluate(
        mrz_result=MRZ_OK, face_result=FACE, tamper_result=GREEN,
        liveness_result=LIVE,
        extra_factors=[{"factor": "ON_WATCHLIST", "detail": "x"}],
    )
    assert result["score"] == 40
    assert any(f["factor"] == "ON_WATCHLIST" for f in result["factors"])
    assert result["decision"] != "CLEARED"


def test_extra_factors_unknown_raises():
    engine = RiskEngine(Settings.from_env())
    with pytest.raises(UnknownRiskFactorError):
        engine.evaluate(
            mrz_result=MRZ_OK, face_result=FACE, tamper_result=GREEN,
            liveness_result=LIVE,
            extra_factors=[{"factor": "NOT_A_REAL_FACTOR", "detail": "x"}],
        )


def test_screen_matching_watchlist_adds_factor(app_and_client):
    _, client = app_and_client
    headers = _auth_token(client)
    created = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "Test Subject", "document_number": "L898902C3",
        "reason": "Integration test", "severity": "HIGH",
    })
    assert created.status_code == 201
    watchlist_id = created.json()["id"]

    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    factors = body["risk_assessment"]["factors"]
    assert any(f["factor"] == "ON_WATCHLIST" for f in factors)
    assert body["watchlist"] == {"match": True, "watchlist_id": watchlist_id,
                                 "severity": "HIGH"}
    # Privacy: the subject's NAME never appears anywhere in the response/persistence.
    assert "Test Subject" not in json.dumps(body)


def test_screen_non_matching_has_no_watchlist(app_and_client):
    _, client = app_and_client
    headers = _auth_token(client)
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["watchlist"] is None