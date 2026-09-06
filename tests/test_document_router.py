"""Document parser strategy + router tests (TD1 national ID + selection)."""

import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services import mrz as mrz_mod
from app.services.mrz import (
    DocumentParserRouter,
    NationalIDTD1Parser,
    PARSER_ALIASES,
    TD3PassportParser,
    parse_td1_national_id,
)


def _settings() -> Settings:
    return Settings.from_env()


def _jpeg(width=400, height=250):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _td1_triple(expiry="350101", dob="740812", doc_no="750000000"):
    check = mrz_mod.calculate_icao_checksum
    name = "SMITH<<JOHN"
    line1 = "IDUTO" + name + "<" * (25 - len(name))
    line2 = (
        doc_no + str(check(doc_no)) + "UTO"
        + dob + str(check(dob)) + "M"
        + expiry + str(check(expiry)) + "<<"
    )
    optional1 = "123456789012345"
    optional2 = "12345678901234"
    composite = line2[0:10] + line2[13:20] + line2[21:28] + optional1
    line3 = optional1 + str(check(composite)) + optional2
    assert len(line1) == len(line2) == len(line3) == 30
    return line1, line2, line3


def test_td1_parse_valid_and_checksums():
    line1, line2, line3 = _td1_triple()
    result = parse_td1_national_id(line1, line2, line3)
    assert result["status"] == "VALID"
    assert result["checks"] == {
        "document_number_valid": True,
        "dob_valid": True,
        "expiry_valid": True,
        "composite_valid": True,
    }
    assert result["document_number"] == "750000000"
    assert result["is_expired"] is False


def test_td1_document_number_checksum_failure_is_invalid():
    line1, line2, line3 = _td1_triple()
    bad = "X" + line2[1:]
    result = parse_td1_national_id(line1, bad, line3)
    assert result["status"] == "INVALID"
    assert result["checks"]["document_number_valid"] is False


def test_td1_malformed_length():
    line1, line2, line3 = _td1_triple()
    assert parse_td1_national_id(line1[:-1], line2, line3)["status"] == "MALFORMED"
    assert parse_td1_national_id(line1, line2, line3[:-5])["status"] == "MALFORMED"


def test_td1_invalid_sex_character_is_malformed():
    line1, line2, line3 = _td1_triple()
    bad = line2[:20] + "X" + line2[21:]
    assert parse_td1_national_id(line1, bad, line3)["status"] == "MALFORMED"


def test_td1_expiry_uses_pivot_year():
    line1, line2, line3 = _td1_triple(expiry="500101")
    past = parse_td1_national_id(line1, line2, line3)
    assert past["status"] == "VALID"
    assert past["is_expired"] is True
    line1, line2, line3 = _td1_triple(expiry="490101")
    future = parse_td1_national_id(line1, line2, line3)
    assert future["status"] == "VALID"
    assert future["is_expired"] is False


def test_parser_aliases_cover_api_document_types():
    assert set(PARSER_ALIASES) == {"td3", "passport", "td1", "national_id", "aadhaar"}


def test_router_selects_passport_by_portrait_ratio():
    router = DocumentParserRouter()
    portrait = _jpeg(width=400, height=600)  # ratio ~0.67 < 1.45
    assert router.select(portrait).name == "passport"


def test_router_selects_national_id_by_landscape_ratio():
    router = DocumentParserRouter()
    landscape = _jpeg(width=400, height=250)  # ratio 1.6 >= 1.45
    assert router.select(landscape).name == "national_id"


def test_router_respects_explicit_document_type():
    router = DocumentParserRouter()
    arbitrary = _jpeg()
    assert router.select(arbitrary, document_type="td3").name == "passport"
    assert router.select(arbitrary, document_type="passport").name == "passport"
    assert router.select(arbitrary, document_type="td1").name == "national_id"
    assert router.select(arbitrary, document_type="national_id").name == "national_id"
    assert router.select(arbitrary, document_type="aadhaar").name == "national_id"


def test_router_rejects_unknown_document_type():
    router = DocumentParserRouter()
    with pytest.raises(ValueError):
        router.select(_jpeg(), document_type="drivers_licence")


def test_router_defaults_to_passport_on_unreadable_bytes():
    router = DocumentParserRouter()
    assert router.select(b"not-an-image").name == "passport"


def test_extract_auto_portrait_routes_to_passport(monkeypatch):
    monkeypatch.setattr(mrz_mod.pytesseract, "image_to_string", lambda *a, **k: "")
    result = mrz_mod.extract_mrz_from_image(_jpeg(width=400, height=600), _settings())
    assert result["format"] == "TD3"
    assert result["document_type"] == "PASSPORT"
    assert result["detected"] is False


def test_extract_auto_landscape_routes_to_national_id(monkeypatch):
    monkeypatch.setattr(mrz_mod.pytesseract, "image_to_string", lambda *a, **k: "")
    result = mrz_mod.extract_mrz_from_image(_jpeg(width=400, height=250), _settings())
    assert result["format"] == "TD1"
    assert result["document_type"] == "NATIONAL_ID"
    assert result["detected"] is False
    assert result["qr_present"] is False


def test_extract_explicit_document_type_overrides_ratio(monkeypatch):
    monkeypatch.setattr(mrz_mod.pytesseract, "image_to_string", lambda *a, **k: "")
    landscape = _jpeg(width=400, height=250)
    result = mrz_mod.extract_mrz_from_image(
        landscape, _settings(), document_type="passport")
    assert result["format"] == "TD3"
    assert result["document_type"] == "PASSPORT"


def test_extract_td1_via_ocr_is_detected(monkeypatch):
    line1, line2, line3 = _td1_triple()
    monkeypatch.setattr(
        mrz_mod.pytesseract, "image_to_string", lambda *a, **k: f"{line1}\n{line2}\n{line3}")
    result = mrz_mod.extract_mrz_from_image(_jpeg(width=400, height=250), _settings())
    assert result["detected"] is True
    assert result["status"] == "VALID"
    assert result["format"] == "TD1"
    assert result["document_type"] == "NATIONAL_ID"
    assert result["data"]["checks"]["composite_valid"] is True


def test_extract_td1_ocr_candidate_short_is_not_detected(monkeypatch):
    line1, line2, line3 = _td1_triple()
    short = line1[:-1]
    monkeypatch.setattr(
        mrz_mod.pytesseract, "image_to_string", lambda *a, **k: f"{short}\n{line2}\n{line3}")
    result = mrz_mod.extract_mrz_from_image(_jpeg(width=400, height=250), _settings())
    assert result["detected"] is False
    assert result["status"] == "NOT_DETECTED"
    assert result["format"] == "TD1"


# ---------------------------------------------------------------------------
# get_document_parser factory
# ---------------------------------------------------------------------------

def test_factory_explicit_types():
    assert isinstance(mrz_mod.get_document_parser("passport"), TD3PassportParser)
    assert isinstance(mrz_mod.get_document_parser("td3"), TD3PassportParser)
    assert isinstance(mrz_mod.get_document_parser("national_id"), NationalIDTD1Parser)
    assert isinstance(mrz_mod.get_document_parser("td1"), NationalIDTD1Parser)
    assert isinstance(mrz_mod.get_document_parser("aadhaar"), NationalIDTD1Parser)


def test_factory_auto_routes_by_image_and_defaults_to_passport():
    portrait = np.asarray(Image.open(io.BytesIO(_jpeg(width=400, height=600))))
    landscape = np.asarray(Image.open(io.BytesIO(_jpeg(width=400, height=250))))
    assert mrz_mod.get_document_parser("auto", portrait).name == "passport"
    assert mrz_mod.get_document_parser("auto", landscape).name == "national_id"
    assert mrz_mod.get_document_parser("auto").name == "passport"
    assert mrz_mod.get_document_parser().name == "passport"


def test_factory_unknown_document_type_raises():
    with pytest.raises(ValueError):
        mrz_mod.get_document_parser("drivers_licence")


def test_national_id_parser_alias_is_backward_compatible():
    assert mrz_mod.NationalIDParser is NationalIDTD1Parser


# ---------------------------------------------------------------------------
# Parser parse() line-input fast paths
# ---------------------------------------------------------------------------

_TD3_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
_TD3_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def test_td3_parser_accepts_pre_extracted_lines():
    parser = TD3PassportParser()
    result = parser.parse(b"", _settings(), line1=_TD3_LINE1, line2=_TD3_LINE2)
    assert result.detected is True
    assert result.status == "VALID"
    assert result.raw["source"] == "form"
    assert result.data["checks"]["composite_valid"] is True


def test_td3_parser_bad_checksum_lines_never_detected():
    parser = TD3PassportParser()
    bad = _TD3_LINE2[:8] + "4" + _TD3_LINE2[9:]
    result = parser.parse(b"", _settings(), line1=_TD3_LINE1, line2=bad)
    assert result.detected is False
    assert result.status == "INVALID"


def test_td3_parser_accepts_ndarray_image(monkeypatch):
    monkeypatch.setattr(mrz_mod.pytesseract, "image_to_string", lambda *a, **k: "")
    parser = TD3PassportParser()
    array = np.zeros((400, 200, 3), np.uint8)
    result = parser.parse(array, _settings())
    assert result.detected is False
    assert result.status == "NOT_DETECTED"
    assert result.document_type == "PASSPORT"


def test_td3_parser_accepts_ndarray_image_with_lines():
    parser = TD3PassportParser()
    array = np.zeros((400, 200, 3), np.uint8)
    result = parser.parse(array, _settings(), line1=_TD3_LINE1, line2=_TD3_LINE2)
    assert result.detected is True
    assert result.status == "VALID"


def test_national_id_parser_accepts_pre_extracted_lines():
    parser = NationalIDTD1Parser()
    line1, line2, line3 = _td1_triple()
    result = parser.parse(b"", _settings(), line1=line1, line2=line2, line3=line3)
    assert result.detected is True
    assert result.status == "VALID"
    assert result.raw["source"] == "form"
    assert result.data["checks"]["composite_valid"] is True


def test_national_id_parser_bad_lines_never_falsely_valid():
    parser = NationalIDTD1Parser()
    line1, line2, line3 = _td1_triple()
    bad = "X" + line2[1:]
    result = parser.parse(b"", _settings(), line1=line1, line2=bad, line3=line3)
    assert result.detected is False
    assert result.status == "INVALID"
