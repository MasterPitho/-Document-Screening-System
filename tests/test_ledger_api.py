import io
import json

from PIL import Image

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _auth_token(client):
    client.post("/api/v1/auth/register", json={
        "username": "ledger_o", "email": "ledger@example.com",
        "full_name": "L O", "password": "Password123!",
    })
    return client.post("/api/v1/auth/login", json={
        "username": "ledger_o", "password": "Password123!",
    }).json()["token"]


def _screen(client, headers):
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, "JPEG")
    return client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", buf.getvalue(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )


def test_ledger_endpoints_and_privacy(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client)}"}

    head = client.get("/api/v1/ledger/head", headers=headers)
    assert head.status_code == 200
    assert head.json()["entry_type"] == "GENESIS"
    assert head.json()["entry_index"] == 0

    resp = _screen(client, headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["persistence"]["ledger_index"] == 1

    verify = client.get("/api/v1/ledger/verify", headers=headers)
    assert verify.json() == {"valid": True, "checked": 2, "broken_at": None}

    listing = client.get("/api/v1/ledger?limit=5&offset=0", headers=headers)
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert listing.json()["total"] >= 2
    assert items[0]["entry_type"] == "SCREENING"
    # Privacy: no document number / no names anywhere in the ledger payloads.
    dumped = json.dumps(listing.json())
    for secret in ("L898902C", "ANNA MARIA", "maria"):
        assert secret not in dumped

    assert client.get("/api/v1/ledger", headers={}).status_code == 401