"""Local Web UI for API keys, paper risk limits, and common paths.

The server binds to 127.0.0.1 only. Secrets are written to git-ignored
config_local.py and are never echoed back into the page.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


CONFIG_PATH = Path("config_local.py")
BEGIN = "# BEGIN TRADE_SIM_WEB_UI"
END = "# END TRADE_SIM_WEB_UI"

DEFAULTS = dict(
    ALPACA_API_KEY="",
    ALPACA_SECRET_KEY="",
    POLYGON_API_KEY="",
    HOLDINGS_CSV="",
    PAPER_TRADING=dict(
        starting_cash=100_000.0,
        max_order_notional=5_000.0,
        max_symbol_notional=20_000.0,
        max_gross_notional=100_000.0,
        allow_short=False,
        blocked_symbols=[],
        slippage_bps=1.0,
        commission_bps=0.0,
        alpaca_base_url="https://paper-api.alpaca.markets",
        log_dir="paper_logs",
    ),
)


def _literal_assignments(text):
    """Safely read literal assignments from config_local.py."""
    out = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                continue
    return out


def load_config(path=CONFIG_PATH):
    """Load local config and fill missing keys from DEFAULTS."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if path.exists():
        vals = _literal_assignments(path.read_text(encoding="utf-8"))
        for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "POLYGON_API_KEY", "HOLDINGS_CSV"):
            if k in vals:
                cfg[k] = vals[k]
        if isinstance(vals.get("PAPER_TRADING"), dict):
            cfg["PAPER_TRADING"].update(vals["PAPER_TRADING"])
    return cfg


def masked(value):
    """Return only configured/missing status, never the secret value."""
    return "configured" if value else "missing"


def public_status(cfg):
    """Status payload for UI display."""
    return dict(
        alpaca_key=masked(cfg.get("ALPACA_API_KEY")),
        alpaca_secret=masked(cfg.get("ALPACA_SECRET_KEY")),
        polygon_key=masked(cfg.get("POLYGON_API_KEY")),
        holdings_csv=cfg.get("HOLDINGS_CSV", ""),
        paper_trading=cfg.get("PAPER_TRADING", {}),
    )


def _csv_snapshot(path, tail=5):
    """Read a CSV summary for the monitor. Local logs only; no trading side effects."""
    info = dict(path=str(path), exists=path.exists(), rows=0, columns=[], tail=[], error="")
    if not path.exists():
        return info
    try:
        import pandas as pd
        df = pd.read_csv(path)
        info.update(
            rows=int(len(df)),
            columns=list(df.columns),
            tail=df.tail(tail).fillna("").to_dict(orient="records"),
        )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _file_status(path):
    """Return log-file freshness metadata."""
    if not path.exists():
        return dict(exists=False, mtime="", age_minutes=None, size=0)
    st = path.stat()
    mt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    age = (datetime.now(timezone.utc) - mt).total_seconds() / 60
    return dict(exists=True, mtime=mt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                age_minutes=float(age), size=int(st.st_size))


def monitor_data(cfg=None, include_alpaca=False):
    """Summarize paper/forward-test status.

    By default this only reads local logs. include_alpaca=True enables read-only
    broker API calls.
    """
    cfg = cfg or load_config()
    paper = cfg.get("PAPER_TRADING", {})
    log_dir = Path(paper.get("log_dir", "paper_logs"))
    names = ["paper_plan.csv", "paper_orders.csv", "paper_fills.csv",
             "paper_summary.csv", "paper_research.csv"]
    files = {name: _file_status(log_dir / name) for name in names}
    csvs = {name: _csv_snapshot(log_dir / name) for name in names}
    research_tail = csvs["paper_research.csv"]["tail"]
    orders_tail = csvs["paper_orders.csv"]["tail"]
    fills_tail = csvs["paper_fills.csv"]["tail"]
    fill_status = {}
    for row in fills_tail:
        status = str(row.get("status", "") or "unknown")
        fill_status[status] = fill_status.get(status, 0) + 1
    latest_research = research_tail[-1] if research_tail else {}
    latest_order = orders_tail[-1] if orders_tail else {}
    out = dict(
        config=public_status(cfg),
        log_dir=str(log_dir),
        files=files,
        csvs=csvs,
        latest_research=latest_research,
        latest_order=latest_order,
        fill_status=fill_status,
        alpaca=None,
    )
    if include_alpaca:
        out["alpaca"] = _alpaca_monitor_snapshot(cfg)
    return out


def _alpaca_monitor_snapshot(cfg):
    """Optionally read Alpaca paper state. Uses read endpoints only."""
    try:
        from paper_broker import AlpacaPaperBroker
        base = cfg.get("PAPER_TRADING", {}).get("alpaca_base_url", "https://paper-api.alpaca.markets")
        b = AlpacaPaperBroker(base_url=base)
        account = b.get_account()
        positions = b.get_positions()
        orders = b.get_orders(limit=10)
        fills = b.get_fills(limit=10)
        return dict(ok=True, account=account, positions=positions, orders=orders, fills=fills, error="")
    except Exception as exc:
        return dict(ok=False, account={}, positions=[], orders=[], fills=[],
                    error=f"{type(exc).__name__}: {exc}")


def _managed_block(cfg):
    paper = cfg["PAPER_TRADING"]
    lines = [
        BEGIN,
        "# Managed by web_ui.py. Keep real keys local and out of git.",
        f"ALPACA_API_KEY = {cfg.get('ALPACA_API_KEY', '')!r}",
        f"ALPACA_SECRET_KEY = {cfg.get('ALPACA_SECRET_KEY', '')!r}",
        f"POLYGON_API_KEY = {cfg.get('POLYGON_API_KEY', '')!r}",
        f"HOLDINGS_CSV = {cfg.get('HOLDINGS_CSV', '')!r}",
        "PAPER_TRADING = {",
        f"    'starting_cash': {float(paper.get('starting_cash', 100000.0))!r},",
        f"    'max_order_notional': {float(paper.get('max_order_notional', 5000.0))!r},",
        f"    'max_symbol_notional': {float(paper.get('max_symbol_notional', 20000.0))!r},",
        f"    'max_gross_notional': {float(paper.get('max_gross_notional', 100000.0))!r},",
        f"    'allow_short': {bool(paper.get('allow_short', False))!r},",
        f"    'blocked_symbols': {list(paper.get('blocked_symbols', []))!r},",
        f"    'slippage_bps': {float(paper.get('slippage_bps', 1.0))!r},",
        f"    'commission_bps': {float(paper.get('commission_bps', 0.0))!r},",
        f"    'alpaca_base_url': {paper.get('alpaca_base_url', 'https://paper-api.alpaca.markets')!r},",
        f"    'log_dir': {paper.get('log_dir', 'paper_logs')!r},",
        "}",
        END,
        "",
    ]
    return "\n".join(lines)


def save_config(cfg, path=CONFIG_PATH):
    """Replace or append the managed block while preserving other local config."""
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    block = _managed_block(cfg)
    if BEGIN in old and END in old:
        pre, rest = old.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        new = pre.rstrip() + "\n\n" + block + post.lstrip("\n")
    else:
        new = old.rstrip() + ("\n\n" if old.strip() else "") + block
    path.write_text(new, encoding="utf-8")


def _num(form, key, default=0.0):
    try:
        return float(form.get(key, [default])[0] or default)
    except ValueError:
        return float(default)


def config_from_form(form, prior):
    """Convert HTML form fields to config. Blank secret fields preserve old values."""
    paper = prior["PAPER_TRADING"].copy()
    for k in ("starting_cash", "max_order_notional", "max_symbol_notional", "max_gross_notional",
              "slippage_bps", "commission_bps"):
        paper[k] = _num(form, k, paper.get(k, 0.0))
    paper["allow_short"] = form.get("allow_short", ["off"])[0] == "on"
    paper["blocked_symbols"] = [s.strip().upper() for s in form.get("blocked_symbols", [""])[0].split(",") if s.strip()]
    paper["alpaca_base_url"] = form.get("alpaca_base_url", [paper.get("alpaca_base_url", "")])[0].strip()
    paper["log_dir"] = form.get("log_dir", [paper.get("log_dir", "paper_logs")])[0].strip() or "paper_logs"
    cfg = prior.copy()
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "POLYGON_API_KEY"):
        val = form.get(key, [""])[0].strip()
        if val:
            cfg[key] = val
    cfg["HOLDINGS_CSV"] = form.get("HOLDINGS_CSV", [prior.get("HOLDINGS_CSV", "")])[0].strip()
    cfg["PAPER_TRADING"] = paper
    return cfg


STYLE = """
body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f7f8f5;color:#20231f}
main{max-width:1080px;margin:0 auto;padding:28px}
h1{font-size:28px;margin:0 0 8px}
h2{font-size:18px;margin:24px 0 12px}
.nav{display:flex;gap:10px;margin:14px 0 20px;flex-wrap:wrap}
.nav a{border:1px solid #c9d0c0;border-radius:999px;padding:7px 11px;color:#20573a;text-decoration:none;background:#fbfcfa;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
section{background:white;border:1px solid #d9ded2;border-radius:8px;padding:18px}
label{display:block;font-size:13px;color:#4b5148;margin:12px 0 6px}
input{width:100%;box-sizing:border-box;padding:10px;border:1px solid #c9d0c0;border-radius:6px;font-size:14px;background:#fff}
input[type=checkbox]{width:auto}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.status{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
.pill{border:1px solid #c9d0c0;border-radius:999px;padding:6px 10px;background:#fbfcfa;font-size:13px}
.ok{border-color:#7aa56f;background:#eef7ec}.miss{border-color:#c48b7f;background:#fff2ef}
button{margin-top:18px;background:#20573a;color:white;border:0;border-radius:6px;padding:11px 15px;font-size:14px;cursor:pointer}
.hint{font-size:13px;color:#667062;line-height:1.45}.toast{margin:12px 0;color:#20573a;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #edf0e8;padding:8px;text-align:left;vertical-align:top}
pre{background:#f7f8f5;border:1px solid #d9ded2;border-radius:6px;padding:10px;overflow:auto;font-size:12px;max-height:280px}
.metric{font-size:24px;font-weight:700;margin:4px 0}.bad{color:#9a3e32}.good{color:#20573a}.muted{color:#667062}
@media(max-width:760px){.grid,.row{grid-template-columns:1fr}main{padding:18px}}
"""


def render_page(message=""):
    cfg = load_config()
    st = public_status(cfg)
    p = cfg["PAPER_TRADING"]
    def esc(x): return html.escape(str(x), quote=True)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>trade_sim local config</title><style>{STYLE}</style></head>
<body><main>
<h1>trade_sim Local Config</h1>
<p class="hint">Binds to 127.0.0.1 only. Saves to git-ignored <code>config_local.py</code>; secrets are not echoed back. Leave secret fields blank to keep the current value.</p>
<div class="nav"><a href="/">Config</a><a href="/monitor">Monitor Portal</a><a href="/api/status">API status JSON</a></div>
{f'<div class="toast">{esc(message)}</div>' if message else ''}
<div class="status">
  <span class="pill {'ok' if st['alpaca_key']=='configured' else 'miss'}">Alpaca Key: {st['alpaca_key']}</span>
  <span class="pill {'ok' if st['alpaca_secret']=='configured' else 'miss'}">Alpaca Secret: {st['alpaca_secret']}</span>
  <span class="pill {'ok' if st['polygon_key']=='configured' else 'miss'}">Polygon Key: {st['polygon_key']}</span>
</div>
<form method="post" action="/save">
<div class="grid">
<section><h2>API Keys</h2>
<label>Alpaca API Key</label><input name="ALPACA_API_KEY" type="password" autocomplete="off" placeholder="blank = keep current value">
<label>Alpaca Secret Key</label><input name="ALPACA_SECRET_KEY" type="password" autocomplete="off" placeholder="blank = keep current value">
<label>Polygon / Massive API Key</label><input name="POLYGON_API_KEY" type="password" autocomplete="off" placeholder="blank = keep current value">
<label>Holdings CSV path</label><input name="HOLDINGS_CSV" value="{esc(cfg.get('HOLDINGS_CSV',''))}">
</section>
<section><h2>Paper Trading</h2>
<div class="row"><div><label>Starting cash</label><input name="starting_cash" value="{esc(p.get('starting_cash'))}"></div>
<div><label>Max order notional</label><input name="max_order_notional" value="{esc(p.get('max_order_notional'))}"></div></div>
<div class="row"><div><label>Max symbol notional</label><input name="max_symbol_notional" value="{esc(p.get('max_symbol_notional'))}"></div>
<div><label>Max gross notional</label><input name="max_gross_notional" value="{esc(p.get('max_gross_notional'))}"></div></div>
<div class="row"><div><label>Slippage bps</label><input name="slippage_bps" value="{esc(p.get('slippage_bps'))}"></div>
<div><label>Commission bps</label><input name="commission_bps" value="{esc(p.get('commission_bps'))}"></div></div>
<label>Blocked symbols, comma-separated</label><input name="blocked_symbols" value="{esc(','.join(p.get('blocked_symbols', [])))}">
<label>Alpaca base URL</label><input name="alpaca_base_url" value="{esc(p.get('alpaca_base_url'))}">
<label>Log dir</label><input name="log_dir" value="{esc(p.get('log_dir'))}">
<label><input type="checkbox" name="allow_short" {'checked' if p.get('allow_short') else ''}> Allow short</label>
</section>
</div>
<button type="submit">Save Local Config</button>
</form>
<p class="hint">This page is not investment advice. After saving, CLI tools can fall back to <code>config_local.py</code> for keys.</p>
</main></body></html>"""


def _fmt_age(info):
    if not info.get("exists"):
        return "missing"
    age = info.get("age_minutes")
    return f"{age:.1f} min ago" if age is not None else ""


def _render_records(records, empty="No records yet"):
    if not records:
        return f"<p class='hint'>{empty}</p>"
    return "<pre>" + html.escape(json.dumps(records, ensure_ascii=False, indent=2, default=str)) + "</pre>"


def render_monitor(include_alpaca=False):
    data = monitor_data(include_alpaca=include_alpaca)
    st = data["config"]
    latest = data["latest_research"]
    verdict = str(latest.get("research_verdict", "NO LOG") or "NO LOG")
    verdict_cls = "good" if verdict == "PASS" else "bad"
    fail = latest.get("fail_reasons", "")
    raw = latest.get("raw_net_bps", "")
    season = latest.get("season_excess_bps", "")
    conc = latest.get("concentration_pct", "")
    def esc(x): return html.escape(str(x), quote=True)
    file_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{'yes' if info['exists'] else 'no'}</td>"
        f"<td>{esc(_fmt_age(info))}</td><td>{esc(info.get('size', 0))}</td></tr>"
        for name, info in data["files"].items()
    )
    status = data["fill_status"] or {}
    alpaca = data.get("alpaca")
    alpaca_block = "<p class='hint'>Alpaca has not been refreshed. By default the monitor only reads local logs; use the button above to call paper read endpoints.</p>"
    if alpaca is not None:
        if alpaca.get("ok"):
            acct = alpaca.get("account", {})
            alpaca_block = (
                f"<p>Account status: <b>{esc(acct.get('status',''))}</b> | "
                f"equity <b>{esc(acct.get('equity',''))}</b> | cash <b>{esc(acct.get('cash',''))}</b></p>"
                f"<h2>Recent Alpaca Orders</h2>{_render_records(alpaca.get('orders', []))}"
                f"<h2>Recent Alpaca Fills</h2>{_render_records(alpaca.get('fills', []))}"
            )
        else:
            alpaca_block = f"<p class='bad'>Alpaca snapshot failed: {esc(alpaca.get('error',''))}</p>"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>trade_sim monitor</title><style>{STYLE}</style></head>
<body><main>
<h1>Monitor Portal</h1>
<p class="hint">Read-only monitor: by default it reads local CSV logs under <code>{esc(data['log_dir'])}</code> and never submits orders.</p>
<div class="nav"><a href="/">Config</a><a href="/monitor">Refresh Local Logs</a><a href="/monitor?alpaca=1">Refresh Alpaca Snapshot</a><a href="/api/monitor">Monitor JSON</a></div>
<div class="status">
  <span class="pill {'ok' if st['alpaca_key']=='configured' else 'miss'}">Alpaca Key: {st['alpaca_key']}</span>
  <span class="pill {'ok' if st['alpaca_secret']=='configured' else 'miss'}">Alpaca Secret: {st['alpaca_secret']}</span>
  <span class="pill {'ok' if st['polygon_key']=='configured' else 'miss'}">Polygon Key: {st['polygon_key']}</span>
</div>
<div class="grid">
<section><h2>Research Gate</h2>
<div class="metric {verdict_cls}">{esc(verdict)}</div>
<p class="hint">fail reasons: {esc(fail or 'none')}</p>
<table><tr><th>raw net bps</th><th>season excess bps</th><th>concentration %</th></tr>
<tr><td>{esc(raw)}</td><td>{esc(season)}</td><td>{esc(conc)}</td></tr></table>
</section>
<section><h2>Execution Logs</h2>
<table><tr><th>file</th><th>exists</th><th>freshness</th><th>bytes</th></tr>{file_rows}</table>
</section>
</div>
<div class="grid">
<section><h2>Latest Planned Order</h2>{_render_records([data['latest_order']] if data['latest_order'] else [])}</section>
<section><h2>Fill Status Tail</h2><pre>{esc(json.dumps(status, ensure_ascii=False, indent=2))}</pre></section>
</div>
<section><h2>Recent Plan</h2>{_render_records(data['csvs']['paper_plan.csv']['tail'])}</section>
<section><h2>Alpaca Paper Snapshot</h2>{alpaca_block}</section>
<p class="hint">Disclaimer: the monitor is only a log/account status view. It is not investment advice and does not replace signal falsification.</p>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/api/status":
            self._send(200, json.dumps(public_status(load_config()), ensure_ascii=False), "application/json")
        elif parsed.path == "/api/monitor":
            data = monitor_data(include_alpaca=qs.get("alpaca", ["0"])[0] == "1")
            self._send(200, json.dumps(data, ensure_ascii=False, default=str), "application/json")
        elif parsed.path == "/monitor":
            self._send(200, render_monitor(include_alpaca=qs.get("alpaca", ["0"])[0] == "1"))
        else:
            self._send(200, render_page())

    def do_POST(self):
        if self.path != "/save":
            self._send(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        cfg = config_from_form(form, load_config())
        save_config(cfg)
        self._send(200, render_page("Saved to config_local.py"))


def run(host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"trade_sim config UI: http://{host}:{port}")
    httpd.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description="trade_sim local config web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    run(args.host, args.port)


if __name__ == "__main__":
    main()
