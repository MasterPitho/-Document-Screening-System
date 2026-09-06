"""ICAO 9303 TD3 MRZ parsing, validation, and OCR extraction tests."""

import io

from app.config import Settings
from app.services import mrz as mrz_mod
from PIL import Image

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
VALID_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _settings() -> Settings:
    return Settings.from_env()


def jpeg_bytes(width=500, height=800):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_valid_td3_mrz_and_composite_checksum():
    result = mrz_mod.parse_td3_mrz(VALID_LINE1, VALID_LINE2)
    assert result["status"] == "VALID"
    assert result["checks"] == {
        "passport_number_valid": True,
        "dob_valid": True,
        "expiry_valid": True,
        "composite_valid": True,
    }


def test_mrz_checksum_failures_are_explicit():
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, "X" + VALID_LINE2[1:])["status"] == "INVALID"
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, VALID_LINE2[:19] + "9" + VALID_LINE2[20:])["status"] == "INVALID"
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, VALID_LINE2[:27] + "8" + VALID_LINE2[28:])["status"] == "INVALID"
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, VALID_LINE2[:-1] + "9")["status"] == "INVALID"


def test_mrz_malformed_length_and_characters():
    assert mrz_mod.parse_td3_mrz(VALID_LINE1[:-1], VALID_LINE2)["status"] == "MALFORMED"
    malformed = VALID_LINE1[:2] + "!" + VALID_LINE1[3:]
    assert mrz_mod.parse_td3_mrz(malformed, VALID_LINE2)["status"] == "MALFORMED"


def test_invalid_sex_character_is_malformed():
    bad = VALID_LINE2[:20] + "X" + VALID_LINE2[21:]
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, bad)["status"] == "MALFORMED"


def test_invalid_nationality_field_is_malformed():
    bad = VALID_LINE2[:10] + "12A" + VALID_LINE2[13:]
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, bad)["status"] == "MALFORMED"


def test_leap_year_dob_is_accepted():
    line = list(VALID_LINE2)
    line[13:19] = "000229"
    line[19] = str(mrz_mod.calculate_icao_checksum("000229"))
    line[21:27] = "490101"
    line[27] = str(mrz_mod.calculate_icao_checksum("490101"))
    composite = "".join(line[0:10]) + "".join(line[13:20]) + "".join(line[21:28]) + "".join(line[28:43])
    line[43] = str(mrz_mod.calculate_icao_checksum(composite))
    result = mrz_mod.parse_td3_mrz(VALID_LINE1, "".join(line))
    assert result["status"] == "VALID"
    assert result["checks"]["dob_valid"] is True


def test_non_leap_feb_29_dob_is_rejected():
    line = list(VALID_LINE2)
    line[13:19] = "010229"
    line[19] = str(mrz_mod.calculate_icao_checksum("010229"))
    result = mrz_mod.parse_td3_mrz(VALID_LINE1, "".join(line))
    assert result["status"] != "VALID"


def test_mrz_year_pivot_decoding():
    assert mrz_mod.mrz_year_full(0) == 2000
    assert mrz_mod.mrz_year_full(49) == 2049
    assert mrz_mod.mrz_year_full(50) == 1950
    assert mrz_mod.mrz_year_full(99) == 1999
    assert mrz_mod.mrz_year_full(69, pivot=70) == 2069
    assert mrz_mod.mrz_year_full(70, pivot=70) == 1970
    assert mrz_mod.mrz_year_full(49, pivot=50) == 2049


def _line2_with_expiry(expiry):
    line = list(VALID_LINE2)
    line[21:27] = expiry
    line[27] = str(mrz_mod.calculate_icao_checksum(expiry))
    composite = "".join(line[0:10]) + "".join(line[13:20]) + "".join(line[21:28]) + "".join(line[28:43])
    line[43] = str(mrz_mod.calculate_icao_checksum(composite))
    return "".join(line)


def test_mrz_expiry_uses_pivot_year():
    future = mrz_mod.parse_td3_mrz(VALID_LINE1, _line2_with_expiry("490101"))
    assert future["status"] == "VALID"
    assert future["is_expired"] is False
    past = mrz_mod.parse_td3_mrz(VALID_LINE1, _line2_with_expiry("500101"))
    assert past["status"] == "VALID"
    assert past["is_expired"] is True


def _extract(monkeypatch, image_bytes, ocr_output):
    monkeypatch.setattr(mrz_mod.pytesseract, "image_to_string", lambda *a, **k: ocr_output)
    return mrz_mod.extract_mrz_from_image(image_bytes, _settings())


def test_ocr_does_not_pad_wrong_length_candidates(monkeypatch):
    result = _extract(
        monkeypatch, jpeg_bytes(), "P<UTO\nL898902C36UTO")
    assert result["detected"] is False
    assert result["status"] == "NOT_DETECTED"


def test_ocr_selects_valid_td3_pair(monkeypatch):
    result = _extract(monkeypatch, jpeg_bytes(), f"{VALID_LINE1}\n{VALID_LINE2}")
    assert result["detected"] is True
    assert result["status"] == "VALID"
    assert result["data"]["checks"]["composite_valid"] is True


def test_ocr_failure_is_safe(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("tesseract unavailable")

    monkeypatch.setattr(mrz_mod.pytesseract, "image_to_string", fail)
    result = mrz_mod.extract_mrz_from_image(jpeg_bytes(), _settings())
    assert result["detected"] is False
    assert result["status"] == "OCR_FAILED"
    assert "tesseract unavailable" not in result["reason"]
    assert result["reason"]


def test_ocr_low_confidence_is_not_valid(monkeypatch):
    weak_line1 = VALID_LINE1.replace("P", "V", 1)
    result = _extract(monkeypatch, jpeg_bytes(), f"{weak_line1}\n{VALID_LINE2[:-1]}9")
    assert result["detected"] is False
    assert result["status"] in {"NOT_DETECTED", "OCR_LOW_CONFIDENCE"}


def test_checksum_invalid_44char_candidate_is_not_detected(monkeypatch):
    bad_line2 = "X" + VALID_LINE2[1:]
    assert mrz_mod.parse_td3_mrz(VALID_LINE1, bad_line2)["status"] == "INVALID"
    result = _extract(monkeypatch, jpeg_bytes(), f"{VALID_LINE1}\n{bad_line2}")
    assert result["detected"] is False
    assert result["status"] in {"NOT_DETECTED", "OCR_LOW_CONFIDENCE"}


def test_wrong_length_ocr_candidate_is_not_detected(monkeypatch):
    short_line = VALID_LINE1[:-1]
    result = _extract(monkeypatch, jpeg_bytes(), f"{short_line}\n{VALID_LINE2}")
    assert result["detected"] is False


def test_invalid_composite_checksum_is_not_detected(monkeypatch):
    bad_line2 = VALID_LINE2[:-1] + "9"
    result = _extract(monkeypatch, jpeg_bytes(), f"{VALID_LINE1}\n{bad_line2}")
    assert result["detected"] is False
    assert result["status"] in {"NOT_DETECTED", "OCR_LOW_CONFIDENCE"}


def test_invalid_date_is_not_detected(monkeypatch):
    bad_line2 = VALID_LINE2[:17] + "3" + VALID_LINE2[18:]
    result = _extract(monkeypatch, jpeg_bytes(), f"{VALID_LINE1}\n{bad_line2}")
    assert result["detected"] is False


def test_invalid_characters_are_not_detected(monkeypatch):
    malformed = VALID_LINE1[:2] + "!" + VALID_LINE1[3:]
    result = _extract(monkeypatch, jpeg_bytes(), f"{malformed}\n{VALID_LINE2}")
    assert result["detected"] is False


def test_detected_true_requires_status_valid(monkeypatch):
    result = _extract(monkeypatch, jpeg_bytes(), f"{VALID_LINE1}\n{VALID_LINE2}")
    if result["detected"]:
        assert result["status"] == "VALID"
