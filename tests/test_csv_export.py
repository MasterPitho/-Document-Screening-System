import io

from PIL import Image

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _auth_token(client, username="csv_o", email="csv@example.com"):
    client.post("/api/v1/auth/register", json={
        "username": username, "email": email,
        "full_name": "C O", "password": "Password123!",
    })
    return client.post("/api/v1/auth/login", json={
        "username": username, "password": "Password123!",
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


def test_export_csv_bom_headers_and_rows(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client)}"}
    resp = _screen(client, headers)
    assert resp.status_code == 200
    request_id = resp.json()["request_id"]

    out = client.get("/api/v1/report/export.csv", headers=headers)
    assert out.status_code == 200
    assert out.headers["content-type"] == "text/csv; charset=utf-8"
    assert out.headers["content-disposition"].startswith("attachment; filename=")
    raw = out.content
    assert raw[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    header = lines[0]
    assert "request_id" in header and "risk_score" in header and "decision" in header
    assert request_id[:8] in text  # request_id is the correlation token (allowed)


def test_export_never_contains_doc_numbers(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client, username='csv_o2', email='csv2@example.com')}"}
    assert _screen(client, headers).status_code == 200
    text = client.get("/api/v1/report/export.csv",
                      headers=headers).content.decode("utf-8-sig")
    assert "L898902C" not in text
    assert "ANNA MARIA" not in text


def test_export_requires_auth(app_and_client):
    _, client = app_and_client
    assert client.get("/api/v1/report/export.csv").status_code == 401


def test_export_respects_date_filter(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client, username='csv_o3', email='csv3@example.com')}"}
    assert _screen(client, headers).status_code == 200
    ok = client.get(
        "/api/v1/report/export.csv",
        params={"date_from": "2020-01-01", "date_to": "2099-12-31"},
        headers=headers)
    assert ok.status_code == 200 and len(ok.content.decode("utf-8-sig").splitlines()) >= 2
    empty = client.get(
        "/api/v1/report/export.csv",
        params={"date_from": "2999-01-01", "date_to": "2999-12-31"},
        headers=headers)
    assert len(empty.content.decode("utf-8-sig").splitlines()) == 1  # header only