import broker_benchmark as bb


def test_benchmark_callables_records_success_and_failure():
    def ok():
        return {"ok": True}

    def bad():
        raise RuntimeError("boom")

    raw, summary = bb.benchmark_callables({"ok": ok, "bad": bad}, n=3)
    assert len(raw) == 6
    err = summary.set_index("endpoint").loc["bad"]
    assert err["error_rate_pct"] == 100.0
    good = summary.set_index("endpoint").loc["ok"]
    assert good["ok"] == 3
    assert good["p50_ms"] >= 0


def test_alpaca_callables_submit_order_is_opt_in():
    class FakeBroker:
        def __init__(self):
            self.submitted = 0
        def get_account(self): return {}
        def get_positions(self): return []
        def get_orders(self): return []
        def get_fills(self): return []
        def submit_order(self, intent):
            self.submitted += 1
            return {"id": "abc"}
        def cancel_order(self, oid):
            return {"id": oid}

    b = FakeBroker()
    calls = bb.alpaca_callables(b, submit_test_order=False)
    assert "submit_cancel_order" not in calls
    for fn in calls.values():
        fn()
    assert b.submitted == 0

    calls = bb.alpaca_callables(b, submit_test_order=True, symbol="AAPL", notional=1.0)
    calls["submit_cancel_order"]()
    assert b.submitted == 1


def test_write_reports(tmp_path):
    raw, summary = bb.benchmark_callables({"ok": lambda: None}, n=1)
    out = bb.write_reports(raw, summary, tmp_path)
    assert (out / "broker_latency_raw.csv").exists()
    assert (out / "broker_latency_summary.csv").exists()
