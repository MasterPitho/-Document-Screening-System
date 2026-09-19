from app.config import Settings


def test_pdf_settings_have_defaults():
    settings = Settings.from_env()
    assert settings.max_pdf_bytes == 20 * 1024 * 1024
    assert settings.pdf_render_max_dimension == 4096


def test_on_watchlist_weight_configured():
    settings = Settings.from_env()
    assert settings.risk_weights["ON_WATCHLIST"] == 40


def test_validate_accepts_new_factors():
    Settings.from_env().validate()  # must not raise