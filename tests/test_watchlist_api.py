def _auth_token(client):
    client.post("/api/v1/auth/register", json={
        "username": "wl_officer",
        "email": "wl@example.com",
        "full_name": "WL Officer",
        "password": "Password123!",
    })
    login_resp = client.post("/api/v1/auth/login", json={
        "username": "wl_officer",
        "password": "Password123!",
    })
    assert login_resp.status_code == 200
    return {"Authorization": f"Bearer {login_resp.json()['token']}"}


def test_list_returns_seeded_demo_entries(app_and_client):
    _, client = app_and_client
    headers = _auth_token(client)
    resp = client.get("/api/v1/watchlists", headers=headers)
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) > 0
    assert all(item["is_demo_data"] for item in items)
    assert items[0]["source"] == "DEMO_DATA_NOT_FOR_OPERATIONAL_USE"


def test_create_list_patch_delete_flow(app_and_client):
    _, client = app_and_client
    headers = _auth_token(client)
    created = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "New Subject", "document_number": "L-898 902c",
        "reason": "Vigilance case", "severity": "HIGH",
    })
    assert created.status_code == 201
    body = created.json()
    assert body["is_demo_data"] is False
    assert body["source"] == "MANUAL_ENTRY"
    assert body["document_number"] == "L-898 902c"
    entry_id = body["id"]

    listing = client.get("/api/v1/watchlists", headers=headers).json()
    assert any(item["id"] == entry_id for item in listing)

    patched = client.patch(f"/api/v1/watchlists/{entry_id}", headers=headers,
                           json={"reason": "Updated rationale"})
    assert patched.status_code == 200
    assert patched.json()["reason"] == "Updated rationale"
    assert patched.json()["name"] == "New Subject"

    searched = client.get("/api/v1/watchlists?search=Updated",
                          headers=headers).json()
    assert any(item["id"] == entry_id for item in searched)

    deleted = client.delete(f"/api/v1/watchlists/{entry_id}", headers=headers)
    assert deleted.status_code == 204
    assert client.delete(f"/api/v1/watchlists/{entry_id}", headers=headers).status_code == 404


def test_create_validation_and_auth(app_and_client):
    _, client = app_and_client
    assert client.get("/api/v1/watchlists").status_code == 401
    headers = _auth_token(client)
    bad = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "X", "document_number": "L898902C", "severity": "EXTREME"})
    assert bad.status_code == 422
    short = client.post("/api/v1/watchlists", headers=headers, json={
        "name": "OK", "document_number": "AB", "severity": "LOW"})
    assert short.status_code == 422