import io

from PIL import Image

from app.services.pdf import is_pdf_content_type, render_first_page

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _pdf_bytes():
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 portrait
    page.insert_text((72, 120), "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                      fontname="helv", fontsize=8)
    return doc.tobytes()


def _auth_token(client, username="pdf_o", email="pdf@example.com"):
    client.post("/api/v1/auth/register", json={
        "username": username, "email": email,
        "full_name": "P O", "password": "Password123!",
    })
    return client.post("/api/v1/auth/login", json={
        "username": username, "password": "Password123!",
    }).json()["token"]


def test_is_pdf_detection():
    assert is_pdf_content_type("application/pdf", "scan.pdf") is True
    assert is_pdf_content_type("", "doc.pdf") is True
    assert is_pdf_content_type("image/jpeg", "photo.jpg") is False


def test_render_first_page_returns_jpeg():
    data = render_first_page(_pdf_bytes())
    assert data is not None and data[:2] == b"\xff\xd8"


def test_render_rejects_invalid_bytes():
    assert render_first_page(b"this is not a pdf") is None
    assert render_first_page(b"") is None


def test_screen_pdf_reports_converted(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client)}"}
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("scan.pdf", _pdf_bytes(), "application/pdf")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["document"]["converted_from_pdf"] is True


def test_screen_invalid_pdf_rejected(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client, username='pdf_o2', email='pdf2@example.com')}"}
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("fake.pdf", b"not really a pdf", "application/pdf")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 422


def test_screen_image_has_no_pdf_flag(app_and_client):
    _, client = app_and_client
    headers = {"Authorization": f"Bearer {_auth_token(client, username='pdf_o3', email='pdf3@example.com')}"}
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, "JPEG")
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("id.jpg", buf.getvalue(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["document"].get("converted_from_pdf") is False