import json
from urllib.parse import parse_qs

import web_ui as ui


def test_save_and_load_masks_secret_status(tmp_path):
    path = tmp_path / "config_local.py"
    cfg = ui.load_config(path=path)
    cfg["ALPACA_API_KEY"] = "dummy-key"
    cfg["ALPACA_SECRET_KEY"] = "dummy-secret"
    cfg["POLYGON_API_KEY"] = "dummy-poly"
    cfg["HOLDINGS_CSV"] = r"C:\dummy\holdings.csv"
    cfg["PAPER_TRADING"]["blocked_symbols"] = ["TQQQ", "UVXY"]

    ui.save_config(cfg, path=path)
    loaded = ui.load_config(path=path)
    status = ui.public_status(loaded)

    assert loaded["ALPACA_SECRET_KEY"] == "dummy-secret"
    assert status["alpaca_secret"] == "configured"
    assert "dummy-secret" not in json.dumps(status)
    assert loaded["PAPER_TRADING"]["blocked_symbols"] == ["TQQQ", "UVXY"]


def test_form_blank_secret_preserves_existing_value():
    prior = ui.load_config()
    prior["ALPACA_API_KEY"] = "keep-key"
    prior["ALPACA_SECRET_KEY"] = "keep-secret"

    form = parse_qs(
        "ALPACA_API_KEY=&ALPACA_SECRET_KEY=new-secret&POLYGON_API_KEY=&"
        "HOLDINGS_CSV=C%3A%5Cdummy%5Choldings.csv&starting_cash=123456&"
        "max_order_notional=5000&max_symbol_notional=20000&max_gross_notional=100000&"
        "allow_short=on&blocked_symbols=tqqq%2C+uvxy&slippage_bps=1.5&commission_bps=0&"
        "alpaca_base_url=https%3A%2F%2Fpaper-api.alpaca.markets&log_dir=paper_logs"
    )
    cfg = ui.config_from_form(form, prior)

    assert cfg["ALPACA_API_KEY"] == "keep-key"
    assert cfg["ALPACA_SECRET_KEY"] == "new-secret"
    assert cfg["POLYGON_API_KEY"] == prior["POLYGON_API_KEY"]
    assert cfg["PAPER_TRADING"]["starting_cash"] == 123456.0
    assert cfg["PAPER_TRADING"]["allow_short"] is True
    assert cfg["PAPER_TRADING"]["blocked_symbols"] == ["TQQQ", "UVXY"]


def test_save_replaces_only_managed_block(tmp_path):
    path = tmp_path / "config_local.py"
    path.write_text("VEST_HOLDINGS = {'x': 1}\n\n# old local note\n", encoding="utf-8")
    cfg = ui.load_config(path=path)
    cfg["POLYGON_API_KEY"] = "first"
    ui.save_config(cfg, path=path)

    cfg["POLYGON_API_KEY"] = "second"
    ui.save_config(cfg, path=path)
    text = path.read_text(encoding="utf-8")

    assert text.count(ui.BEGIN) == 1
    assert text.count(ui.END) == 1
    assert "VEST_HOLDINGS = {'x': 1}" in text
    assert "POLYGON_API_KEY = 'second'" in text
    assert "POLYGON_API_KEY = 'first'" not in text
