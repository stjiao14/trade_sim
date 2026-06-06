"""本机配置 Web UI:管理 API key、paper 风控和常用路径。

只监听 127.0.0.1;敏感值写入 git-ignored 的 config_local.py,页面不回显 secret。
"""
from __future__ import annotations

import argparse
import ast
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


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
    """安全读取 config_local.py 里的字面量赋值。"""
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
    """读取本地配置,缺省值从 DEFAULTS 补齐。"""
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
    """只返回配置状态,不回显 secret。"""
    return "configured" if value else "missing"


def public_status(cfg):
    """UI 展示用状态。"""
    return dict(
        alpaca_key=masked(cfg.get("ALPACA_API_KEY")),
        alpaca_secret=masked(cfg.get("ALPACA_SECRET_KEY")),
        polygon_key=masked(cfg.get("POLYGON_API_KEY")),
        holdings_csv=cfg.get("HOLDINGS_CSV", ""),
        paper_trading=cfg.get("PAPER_TRADING", {}),
    )


def _managed_block(cfg):
    paper = cfg["PAPER_TRADING"]
    lines = [
        BEGIN,
        "# 由 web_ui.py 管理。真实 key 只留本机,不要提交。",
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
    """替换/追加 managed block,保留其它本地配置。"""
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
    """HTML form -> config。空 secret 字段表示保留原值。"""
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
<h1>trade_sim 本地配置</h1>
<p class="hint">只监听 127.0.0.1。保存到 git-ignored 的 <code>config_local.py</code>;secret 不回显,留空表示保留原值。</p>
{f'<div class="toast">{esc(message)}</div>' if message else ''}
<div class="status">
  <span class="pill {'ok' if st['alpaca_key']=='configured' else 'miss'}">Alpaca Key: {st['alpaca_key']}</span>
  <span class="pill {'ok' if st['alpaca_secret']=='configured' else 'miss'}">Alpaca Secret: {st['alpaca_secret']}</span>
  <span class="pill {'ok' if st['polygon_key']=='configured' else 'miss'}">Polygon Key: {st['polygon_key']}</span>
</div>
<form method="post" action="/save">
<div class="grid">
<section><h2>API Keys</h2>
<label>Alpaca API Key</label><input name="ALPACA_API_KEY" type="password" autocomplete="off" placeholder="留空=保留当前值">
<label>Alpaca Secret Key</label><input name="ALPACA_SECRET_KEY" type="password" autocomplete="off" placeholder="留空=保留当前值">
<label>Polygon / Massive API Key</label><input name="POLYGON_API_KEY" type="password" autocomplete="off" placeholder="留空=保留当前值">
<label>持仓 CSV 路径</label><input name="HOLDINGS_CSV" value="{esc(cfg.get('HOLDINGS_CSV',''))}">
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
<button type="submit">保存本地配置</button>
</form>
<p class="hint">本页面不构成投资建议。配置保存后,CLI 会自动从 <code>config_local.py</code> 兜底读取 key。</p>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if self.path == "/api/status":
            self._send(200, json.dumps(public_status(load_config()), ensure_ascii=False), "application/json")
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
        self._send(200, render_page("已保存到 config_local.py"))


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
