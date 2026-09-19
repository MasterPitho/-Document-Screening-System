from app.services.mrz import (
    DocumentParserRouter,
    VisaTD2Parser,
    calculate_icao_checksum,
    parse_td2_visa,
)

# Deterministic, fully-compliant TD2 vector built exactly per ICAO 9303 Part 4.
# Line 2: doc(9)+chk | nationality(3) | dob(6)+chk | sex(1) | expiry(6)+chk |
#         optional(7) | composite-chk(1). Composite covers
#         line2[0:10] + line2[13:20] + line2[21:28] + line2[28:35].
LINE1 = ("VBIND" + "TEST<<PERSON" + "<" * (36 - len("VBIND") - len("TEST<<PERSON")))
_DOC = "123456789"
_DOB = "900101"
_EXP = "301231"
_OPT = "ASDFGHJ"
_BODY = (_DOC + str(calculate_icao_checksum(_DOC)) + "IND"
         + _DOB + str(calculate_icao_checksum(_DOB)) + "M"
         + _EXP + str(calculate_icao_checksum(_EXP)) + _OPT)
_COMPOSITE = _BODY[0:10] + _BODY[13:20] + _BODY[21:28] + _BODY[28:35]
LINE2 = _BODY + str(calculate_icao_checksum(_COMPOSITE))


def test_vector_shape():
    assert len(LINE1) == 36 and len(LINE2) == 36


def test_parse_valid_td2_visa():
    parsed = parse_td2_visa(LINE1, LINE2)
    assert parsed["status"] == "VALID"
    assert parsed["doc_type"] == "VB"
    assert parsed["issuing_country"] == "IND"
    assert parsed["document_number"] == "123456789"
    assert parsed["full_name"] == "PERSON TEST"
    assert parsed["nationality"] == "IND"
    assert parsed["date_of_birth"] == "900101"
    assert parsed["gender"] == "M"
    assert parsed["expiry_date"] == "301231"
    assert parsed["optional_data"] == "ASDFGHJ"
    assert all(parsed["checks"].values())
    assert parsed["is_expired"] is False


def test_parse_detects_corrupted_digit():
    corrupt = list(LINE2)
    corrupt[0] = "9" if corrupt[0] != "9" else "8"
    parsed = parse_td2_visa(LINE1, "".join(corrupt))
    assert parsed["status"] == "INVALID"
    assert parsed["checks"]["document_number_valid"] is False


def test_parse_malformed_length():
    assert parse_td2_visa("TOOSHORT", LINE2)["status"] == "MALFORMED"


def test_expired_visa_detected():
    doc = "123456789"; chk = calculate_icao_checksum(doc)
    dob = "900101"; dob_chk = calculate_icao_checksum(dob)
    exp = "200101"; exp_chk = calculate_icao_checksum(exp)  # past expiry
    opt = "ASDFGHJ"
    full = (doc + str(chk) + "IND" + dob + str(dob_chk) + "M"
            + exp + str(exp_chk) + opt)
    comp_data = full[0:10] + full[13:20] + full[21:28] + full[28:35]
    line2 = full + str(calculate_icao_checksum(comp_data))
    parsed = parse_td2_visa(LINE1, line2)
    assert parsed["status"] == "VALID"
    assert parsed["is_expired"] is True


def test_parser_from_lines_detected():
    parser = VisaTD2Parser()
    result = parser.parse(None, line1=LINE1, line2=LINE2)
    assert result.detected is True
    assert result.status == "VALID"
    assert result.document_type == "VISA"
    assert result.format == "TD2"
    assert result.data["document_number"] == "123456789"


def test_router_aliases():
    router = DocumentParserRouter()
    assert router.resolve("visa") == "visa"
    assert router.resolve("td2") == "visa"
    assert router.resolve("passport") == "passport"


def test_screen_endpoint_accepts_visa_type(app_and_client):
    import io

    from PIL import Image
    _, client = app_and_client
    client.post("/api/v1/auth/register", json={
        "username": "visa_o", "email": "visa@example.com",
        "full_name": "V O", "password": "Password123!",
    })
    token = client.post("/api/v1/auth/login", json={
        "username": "visa_o", "password": "Password123!",
    }).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    buf = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buf, "JPEG")
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("visa.jpg", buf.getvalue(), "image/jpeg")},
        data={"mrz_line1": LINE1, "mrz_line2": LINE2,
              "document_type": "td2"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["document"]["document_type"] == "VISA"