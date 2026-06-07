# trade_sim

`trade_sim` is a small research and risk-analysis toolkit. It is not an auto-trading bot.

The repo has two main machines:

1. **Signal falsification**: put any trading signal through the same failure gates and find out whether it is a repeatable edge or just cost drag, noise, momentum, one-name concentration, or a single-regime artifact.
2. **Personal risk X-ray**: read a local holdings CSV and quantify true GOOGL look-through exposure, factor exposure, number of independent bets, and the risk distribution of holding vs selling RSU shares after vesting.

Real dollar amounts and API keys do not belong in git. Put real holdings in a local CSV and real config in git-ignored `config_local.py`. The repo only commits dummy examples in `config_example.py`.

## Feature Map

| Module | Question it answers | Main output | API needed |
|---|---|---|---|
| `intraday_seasonality_backtest.py` | Does the original intraday seasonality strategy have a real edge? | Four diagnostic gates, long-window quarterly report, long-only vs long/short comparison | yfinance for short windows; Polygon/Massive for long windows |
| `signal_lab.py` | Can any new signal survive the same falsification battery? | `PASS/FAIL`, failure reasons, random floor, regime and concentration diagnostics | None if you already have a panel |
| `forward_paper.py` | Turn a researched signal into next-day paper order plans | `paper_plan.csv`, `paper_orders.csv`, research gate logs | Polygon/Massive; optional Alpaca paper |
| `paper_broker.py` | Local simulated broker plus Alpaca paper read/submit adapter | fills, orders, positions, account snapshot | Optional Alpaca paper |
| `broker_benchmark.py` | Measure broker API latency and reliability | p50/p95/max latency, error rate | Alpaca paper |
| `txtadel_analysis.py` | Audit Txtadel-style posted overnight ETF baskets | parsed orders, posted-vs-recomputed return, capped inverse-vol weight fit | None for PDF parsing; optional Polygon/Massive or yfinance for weight-fit tests |
| `overnight_basket_backtest.py` | Backtest Txtadel-style close-to-next-open ETF baskets | overnight return metrics, Sharpe, drawdown, benchmark excess | Polygon/Massive or yfinance |
| `concentration_analysis.py` | What is the true GOOG/GOOGL look-through exposure? | direct stock + ETF look-through + RSU + shock losses | yfinance for prices/holdings weights; fallback weights available |
| `factor_xray.py` | What systematic factors drive the whole portfolio? | factor beta/R2, ENB, DR, PC1, risk contribution | yfinance |
| `vest_diversify_sim.py` | What is the risk tradeoff of holding vs selling vested RSU? | terminal-value percentiles, standard deviation, max drawdown, breakeven drift | yfinance calibration; fallback values if unavailable |
| `web_ui.py` | Configure local keys/paths and monitor paper execution | git-ignored local config, paper-log status, optional Alpaca snapshot | None by default; optional Alpaca paper |

## Typical Workflows

### 1. Falsify a Trading Signal

Build a panel shaped like `DataFrame[date, slot, ticker, ret]`, then write a scoring function:

```python
from signal_lab import falsify, print_verdict

def my_signal(hist):
    return hist.mean()

v = falsify(panel, my_signal)
print_verdict(v)
```

A signal only passes if it is positive after cost, beats the random floor, is not dominated by a single name, does not only work on high-volatility days, survives dropping the top-volatility days, and has a positive daily bootstrap lower confidence bound.

### 2. Re-test Intraday Seasonality Over a Long Window

The yfinance path is useful for a quick 60-day smoke test. A real persistence check needs longer 30-minute history from Polygon/Massive:

```python
import intraday_seasonality_backtest as bt
bt.long_window_report_polygon()
```

Current research conclusion: the intraday seasonality strategy did not survive the 18-month multi-regime test. The full-window result is roughly negative after cost and the isolated seasonality excess is near the random-selection floor. The recent positive result appears concentrated in one short regime with elevated concentration and high-volatility dependence, so it looks more like momentum/regime contamination than a stable standalone seasonality edge.

### 3. Run a Paper Forward Test

The intended workflow is research first, paper execution second. Shadow mode writes the order plan without touching a broker:

```bash
python forward_paper.py --mode shadow --out paper_logs
```

To submit to Alpaca paper, configure credentials through the local Web UI or environment variables, then explicitly choose the broker:

```bash
python forward_paper.py --broker alpaca-paper --mode paper --notional 1000 --require-pass --out paper_logs
```

`--require-pass` is the guardrail: if the research gate fails, the runner writes diagnostics but skips execution.

### 4. X-ray Personal Portfolio Risk

Put the real holdings CSV path into `config_local.py` or the Web UI, then run:

```bash
python concentration_analysis.py
python factor_xray.py
python vest_diversify_sim.py
```

Together these scripts answer: how much true GOOGL/tech exposure you have, how many independent bets the portfolio really contains, and how much annual GOOG outperformance would be needed to justify holding vested RSU instead of diversifying.

## Setup

```bash
pip install numpy pandas yfinance pytest
```

Optional long-history intraday data uses Polygon/Massive:

```bash
export POLYGON_API_KEY="..."
```

Windows PowerShell:

```powershell
$env:POLYGON_API_KEY="..."
```

Recommended local config:

```python
HOLDINGS_CSV = r"C:\path\to\your_holdings.csv"
```

`config_local.py` is ignored by git.

### Local Web UI

If you do not want to edit `config_local.py` manually, start the local-only configuration UI:

```bash
python web_ui.py
```

On Windows, if `python` is captured by the Microsoft Store alias, use the bundled Python path:

```powershell
C:\Users\stjia\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe web_ui.py
```

Open:

```text
http://127.0.0.1:8765
```

The page can configure Alpaca paper keys, Polygon keys, the holdings CSV path, and paper-trading risk limits. It saves to git-ignored `config_local.py`. Saved secrets are shown only as `configured/missing`; they are not echoed back into the page. Environment variables still take precedence for temporary overrides.

The same local service also provides a monitor portal:

```text
http://127.0.0.1:8765/monitor
```

The monitor defaults to local CSV logs only and does not submit orders. It shows:

- API-key configuration status
- Whether `paper_plan.csv`, `paper_orders.csv`, `paper_fills.csv`, `paper_summary.csv`, and `paper_research.csv` exist and how fresh they are
- The latest research-gate `PASS/FAIL`, failure reasons, raw net, seasonality excess, and concentration
- The latest planned order and recent fill-status tail
- Optional Alpaca paper account, recent orders, and recent fills when you manually click `Refresh Alpaca Snapshot`

The monitor is an execution status page, not proof of strategy quality. Strategy quality is still determined by the falsification gates in `signal_lab.py` and `intraday_seasonality_backtest.py`.

## Quick Test

```bash
pytest -q
```

The test suite covers:

- synthetic guardrails for intraday seasonality: no edge, true seasonality, and momentum trap
- regime fragility, P&L attribution, long/short, and Polygon data-contract checks
- generic signal falsification in `signal_lab.py`
- local broker, risk gate, Alpaca paper adapter, and broker benchmark math
- GOOGL look-through, factor x-ray, RSU diversification simulation, and Web UI math/monitor behavior

## A. Signal Falsification

### `intraday_seasonality_backtest.py`

This is the real-data backtest for the original intraday seasonality strategy. The strategy picks one stock for each of the 13 regular-session 30-minute slots. The key question is not "can a backtest make money?" but:

- Is there genuine cross-sectional seasonality?
- Is it tradeable after realistic round-trip cost?
- Is the apparent profit actually momentum, one-name concentration, or a few high-volatility days?

Run the short yfinance version:

```bash
python intraday_seasonality_backtest.py
```

Run the Polygon long-window version:

```python
import intraday_seasonality_backtest as bt
bt.long_window_report_polygon()
```

### `signal_lab.py`

Generic signal falsification library. Write a `signal_fn(hist) -> Series` and run the same gauntlet:

```python
from signal_lab import falsify, print_verdict

def my_signal(hist):
    return hist.mean()

v = falsify(panel, my_signal)
print_verdict(v)
```

`hist` is a lookback panel for one slot or bucket: rows are past trading days, columns are tickers or units, and values are returns. Higher scores are selected first.

Built-in signals:

- `seasonality_signal`: same-slot mean, reproducing the original backtest
- `momentum_signal`: lookback cumulative return
- `reversal_signal`: negative lookback cumulative return

### `paper_broker.py` / `paper_runner.py`

Paper execution layer. It does not prove alpha; it converts researched order intents into auditable paper orders, fills, and positions.

Local offline paper account:

```python
from paper_broker import LocalPaperBroker, OrderIntent, RiskGate, RiskLimits

broker = LocalPaperBroker(cash=100_000, price_map={"AAPL": 200.0}, slippage_bps=1.0)
gate = RiskGate(RiskLimits(max_order_notional=5_000, max_symbol_notional=20_000))
broker.submit_order(OrderIntent("AAPL", "buy", notional=1_000, reason="demo"), risk_gate=gate)
print(broker.positions_frame())
```

Convert backtest or signal results into orders:

```python
from paper_runner import intents_from_picks, run_intents, print_broker_report

intents = intents_from_picks(backtest_result, notional_per_trade=1_000)
run_intents(broker, intents, risk_gate=gate)
print_broker_report(broker)
```

The live paper adapter is `AlpacaPaperBroker`. It only supports `https://paper-api.alpaca.markets` to avoid accidental live trading. Polygon/Massive is for market data; broker APIs are for paper execution and account state.

### `forward_paper.py`

Forward paper runner. It pulls Polygon 30-minute data, builds the next trading day's 13-slot order plan from the most recent lookback window, and writes CSV logs.

```bash
python forward_paper.py --mode shadow --out paper_logs
```

Local paper execution:

```bash
python forward_paper.py --mode paper --notional 1000 --require-pass --out paper_logs
```

Alpaca paper execution:

```bash
$env:ALPACA_API_KEY="..."
$env:ALPACA_SECRET_KEY="..."
python forward_paper.py --broker alpaca-paper --mode paper --notional 1000 --require-pass --out paper_logs
```

Output files:

- `paper_plan.csv`: slot-level picks and scores
- `paper_orders.csv`: order intents
- `paper_fills.csv`: local paper fills
- `paper_summary.csv`: account summary
- `paper_research.csv`: research-gate metrics and failure reasons

Start with shadow mode, inspect the logs for a few weeks, then consider connecting the broker paper API.

### `broker_benchmark.py`

Broker API latency and reliability benchmark. By default it only reads Alpaca paper endpoints:

```bash
python broker_benchmark.py --broker alpaca-paper --n 10 --out broker_benchmark_logs
```

It writes:

- `broker_latency_raw.csv`: one row per call
- `broker_latency_summary.csv`: p50/p95/max/error-rate by endpoint

Only this explicit flag submits and cancels a test order:

```bash
python broker_benchmark.py --broker alpaca-paper --submit-test-order --symbol AAPL --notional 1
```

For this strategy, throughput is not the bottleneck; p95 latency, error rate, and stable account/orders/fills reconciliation matter more.

### `txtadel_analysis.py`

Txtadel-style posted signals are overnight ETF baskets: buy near the close, sell on the next open, and allocate across roughly five liquid ETFs. The PDF export shows posted orders and returns, but not the full signal-generation rule. This module therefore audits only what is observable.

Run on a PDF export:

```bash
python txtadel_analysis.py "C:\path\to\Txtadel Online.pdf"
```

Add a market-data-backed weight fit:

```bash
python txtadel_analysis.py "C:\path\to\Txtadel Online.pdf" --fit-weights --provider auto --lookbacks 20,40,60,120 --cap 0.35
```

Add simple selection-rule overlap scoring:

```bash
python txtadel_analysis.py "C:\path\to\Txtadel Online.pdf" --score-selection --provider auto --lookbacks 20,40,60,120 --top-n 5
```

You can combine both:

```bash
python txtadel_analysis.py "C:\path\to\Txtadel Online.pdf" --fit-weights --score-selection --provider auto --lookbacks 20,40,60,120 --cap 0.35 --top-n 5
```

`--provider auto` tries Polygon/Massive first when `POLYGON_API_KEY` is configured, then falls back to yfinance. The weight fit tests whether posted weights resemble capped inverse-vol or risk-parity weights over the requested lookbacks. Read `mae_pct` and `rmse_pct` as average weight error in percentage points; lower is better. Read `corr` as directional similarity; higher is better.

The selection scoring tests transparent top-N rules against the posted ETF basket using only history before each posted date:

- `mean`: trailing average return
- `tstat`: trailing mean divided by volatility
- `momentum`: trailing cumulative return
- `reversal`: negative trailing cumulative return
- `low_vol`: inverse realized volatility

It reports:

- parsed `date, ticker, weight, buy_close, sell_open, gain`
- daily posted total return vs recomputed weighted return
- ticker frequency and concentration
- optional capped inverse-vol/risk-parity weight fit from downloaded daily closes
- optional candidate selection-rule overlap

Useful Python entry points:

```python
import txtadel_analysis as tx

orders, daily = tx.parse_txtadel_pdf("Txtadel Online.pdf")
tx.compare_posted_vs_recomputed(orders, daily)
returns, provider = tx.load_daily_returns_for_orders(orders, provider="auto")
tx.fit_inverse_vol_weighting(orders, returns, lookbacks=(20, 40, 60, 120), cap=0.35)
tx.score_candidate_selection_rules(orders, returns, lookbacks=(20, 40, 60, 120), top_n=5)
```

The first PDF audit parsed 10 dates and 50 rows. The nine completed daily totals recomputed to within a few thousandths of a percent, which confirms the posted return arithmetic. The capped inverse-vol fit was only partial: 120-day lookback was best at about `mae_pct=5.24`, `rmse_pct=6.42`, and `corr=0.63`. That means inverse-vol/risk-parity explains some structure in the weights but is not enough to prove the hidden weighting rule.

The first selection-rule pass found that simple trailing reversal rules were closest but still incomplete: 40-60 day reversal hit about `3.1/5` posted tickers on average, with `0` exact-match days and weak frequency correlation. That suggests the posted basket has some reversal-like structure, but these transparent rules do not fully recover the hidden ETF selection rule.

### `overnight_basket_backtest.py`

First-pass engine for Txtadel-style close-to-next-open ETF baskets. This is intentionally simpler than the final claimed strategy: it does not infer the hidden selection rule yet. It answers the base mechanical question: if a specified ETF basket is bought at today's close and sold at the next session's open, what are the return, Sharpe, drawdown, and benchmark comparison?

```bash
python overnight_basket_backtest.py --tickers XLU,GLD,EMXC,XLE,SMH,CGDV,XLI,AVUV --start 2019-01-01 --provider auto --cost-bps 1
```

Add the first two-axis risk gate:

```bash
python overnight_basket_backtest.py --tickers XLU,GLD,EMXC,XLE,SMH,CGDV,XLI,AVUV --start 2019-01-01 --provider auto --cost-bps 1 --gate vix-macro --vix-max 30 --risk-on XLY --risk-off XLP --macro-lookback 200 --macro-min -0.20 --decision-lag 1
```

The macro axis is:

```text
((XLY/XLP)_today - (XLY/XLP)_{today-200}) / (XLY/XLP)_today
```

`--decision-lag 1` is the honest default: today's close order uses the previous completed session's VIX and macro state. `--decision-lag 0` reproduces a same-day-close rule, but only use it if the real process has those inputs before submitting the MOC order.

`--provider auto` tries Polygon/Massive first when configured, then falls back to yfinance. The current version supports fixed equal-weight or custom-weight basket research from Python. The next layer is candidate ETF selection rules.

Useful Python entry points:

```python
import overnight_basket_backtest as ob

bars, provider = ob.load_daily_ohlc(["XLU", "GLD", "SMH"], "2019-01-01", provider="auto")
overnight = ob.overnight_returns(bars)
res = ob.backtest_static_basket(overnight, cost_bps=1.0)
ob.print_report(res, ob.performance_metrics(res))
```

## B. Personal Risk X-ray

### `concentration_analysis.py`

Computes true Alphabet exposure: direct GOOG/GOOGL, SPY/QQQ/TQQQ look-through, unvested RSU, and optional human-capital exposure.

```bash
python concentration_analysis.py path/to/holdings.csv
```

Or set `HOLDINGS_CSV` in `config_local.py` and run:

```bash
python concentration_analysis.py
```

Outputs:

- liquid-account GOOGL-equivalent exposure
- GOOGL-equivalent exposure including RSU
- GOOGL -30% and -50% shock losses
- beta/R2 of the GOOG/SPY/QQQ/TQQQ tech sleeve to GOOGL

That beta/R2 is for the tech sleeve, not the whole portfolio. Use `factor_xray.py` for the full-account regression.

### `factor_xray.py`

Full-portfolio factor x-ray plus true diversification score:

```bash
python factor_xray.py path/to/holdings.csv
```

Or:

```bash
python factor_xray.py
```

Outputs:

- ETF-proxy factor regression: market, tech, value, size, international, rates, gold, oil
- R2 and collinearity warning
- ENB: effective number of independent bets
- DR: diversification ratio; 1.0 means little diversification benefit
- PC1 share: how much cross-sectional variance is explained by the largest common factor
- risk contribution table by holding

### `vest_diversify_sim.py`

Monte Carlo for holding vs selling vested RSU:

```bash
python vest_diversify_sim.py
```

Local config:

```python
VEST_HOLDINGS = dict(
    goog_price=250.0,
    rsu_unvested_usd=0.0,
    liquid_goog_usd=0.0,
    liquid_basket_usd=0.0,
)
VEST_MONTHS = list(range(3, 49, 3))
BASKET_PROXY = "VT"
N_PATHS = 20000
```

The default assumption is `mu_GOOG == mu_basket`, which isolates risk rather than forecasting which asset will outperform. The report prints terminal-value distributions, standard deviation, max drawdown, and breakeven drift: the annual GOOG outperformance required for HOLD's median outcome to catch up to SELL.

## Privacy

- Do not commit real holdings CSV files.
- Do not commit `config_local.py`.
- Do not commit API keys.
- `config_example.py` contains dummy values only.

## Caveats

All scripts are research and risk-quantification tools, not investment advice. ETF factor proxies are collinear, so read R2, dominant exposures, and stability rather than treating each beta as an orthogonal factor. RSU simulation uses a simplified GBM without fat tails or regimes; a stricter next step would be block bootstrap or Student-t shocks.
