"""Monte Carlo risk simulator for HOLD vs SELL-after-vesting RSU decisions.

Put real dollar amounts in git-ignored config_local.py. This script is risk
quantification only and is not investment advice.
"""
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf


FALLBACK_SIGMA_GOOG = 0.32
FALLBACK_SIGMA_BASKET = 0.16
FALLBACK_RHO = 0.60


def _load_config():
    """Load local config; fall back to dummy example values with a loud warning."""
    try:
        import config_local as C
        source = "config_local.py"
    except ImportError:
        import config_example as C
        source = "config_example.py"
        print("WARNING: using dummy values from config_example.py; put real dollars in git-ignored config_local.py")

    # Support older HOLDINGS shape, but prefer the RSU-specific VEST_HOLDINGS config.
    if hasattr(C, "VEST_HOLDINGS"):
        H = dict(C.VEST_HOLDINGS)
    elif hasattr(C, "HOLDINGS") and {"goog_price", "rsu_unvested_usd",
                                      "liquid_goog_usd", "liquid_basket_usd"} <= set(C.HOLDINGS):
        H = dict(C.HOLDINGS)
    else:
        import config_example as E
        H = dict(E.VEST_HOLDINGS)
        if source == "config_local.py":
            print("WARNING: config_local.py does not define VEST_HOLDINGS; using dummy RSU config from config_example.py")

    H["vest_months"] = list(getattr(C, "VEST_MONTHS", H.get("vest_months", list(range(3, 49, 3)))))
    basket_proxy = getattr(C, "BASKET_PROXY", "VT")
    n_paths = int(getattr(C, "N_PATHS", 20000))
    return H, basket_proxy, n_paths, source


def _polygon_daily_close(ticker, start, end, api_key):
    """Polygon adjusted daily close as a date-indexed Series."""
    params = urllib.parse.urlencode(dict(adjusted="true", sort="asc", limit=50000, apiKey=api_key))
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?{params}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data.get("status") not in {"OK", "DELAYED"} or not data.get("results"):
        raise ValueError(f"Polygon {ticker} has no usable daily bars: {data.get('status')} {data.get('error')}")
    df = pd.DataFrame(data["results"])
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    return pd.Series(df["c"].astype(float).values, index=pd.Index(idx, name="date"), name=ticker)


def _calibrate_polygon(basket_proxy, lookback_days, api_key=None):
    """Estimate GOOG/basket sigma and rho from Polygon adjusted daily closes."""
    key = api_key or os.getenv("POLYGON_API_KEY")
    if not key:
        raise ValueError("POLYGON_API_KEY is not set")
    end = pd.Timestamp.today(tz="America/New_York").date()
    start = (pd.Timestamp(end) - pd.Timedelta(days=int(lookback_days * 1.7 + 30))).date()
    g = _polygon_daily_close("GOOGL", start, end, key)
    b = _polygon_daily_close(basket_proxy, start, end, key)
    px = pd.concat([g, b], axis=1).dropna().tail(lookback_days)
    if len(px) < max(100, lookback_days // 3):
        raise ValueError(f"Polygon has too few usable daily bars: {len(px)}")
    r = np.log(px).diff().dropna()
    sg = float(r["GOOGL"].std() * np.sqrt(252))
    sb = float(r[basket_proxy].std() * np.sqrt(252))
    rho = float(r["GOOGL"].corr(r[basket_proxy]))
    if not (np.isfinite(sg) and np.isfinite(sb) and np.isfinite(rho)):
        raise ValueError("Polygon calibration contains NaN/inf")
    return sg, sb, rho


def _calibrate_yfinance(basket_proxy, lookback_days):
    """yfinance fallback: estimate annualized vol/correlation from daily returns."""
    px = yf.download(["GOOGL", basket_proxy], period=f"{lookback_days}d", interval="1d",
                     progress=False, auto_adjust=False, threads=False)["Close"].dropna()
    r = np.log(px).diff().dropna()
    sg = float(r["GOOGL"].std() * np.sqrt(252))
    sb = float(r[basket_proxy].std() * np.sqrt(252))
    rho = float(r["GOOGL"].corr(r[basket_proxy]))
    if not (np.isfinite(sg) and np.isfinite(sb) and np.isfinite(rho)):
        raise ValueError("calibration contains NaN/inf")
    return sg, sb, rho


def calibrate(basket_proxy, lookback_days=1260, api_key=None):
    """Estimate GOOG/basket sigma/rho, preferring Polygon and falling back to yfinance."""
    try:
        return (*_calibrate_polygon(basket_proxy, lookback_days, api_key=api_key), "polygon")
    except Exception as exc:
        print(f"WARNING: Polygon calibration failed; falling back to yfinance: {exc}")
    return (*_calibrate_yfinance(basket_proxy, lookback_days), "yfinance")


def simulate(cfg, mu_g, mu_b, sg, sb, rho, N=20000, seed=0):
    P0 = cfg["goog_price"]
    if P0 <= 0:
        raise ValueError("goog_price must be > 0 to convert RSU dollar value to shares")
    rsu = cfg["rsu_unvested_usd"] / P0
    vm = list(cfg["vest_months"])
    if not vm:
        raise ValueError("vest_months cannot be empty")
    spt = rsu / len(vm)
    H = max(vm)
    lg, lb = cfg["liquid_goog_usd"], cfg["liquid_basket_usd"]
    rng = np.random.default_rng(seed)
    dt = 1 / 12
    L = np.linalg.cholesky([[sg**2, rho * sg * sb], [rho * sg * sb, sb**2]])
    z = rng.standard_normal((N, H, 2)) @ L.T
    Pg = np.concatenate([np.ones((N, 1)),
                         np.cumprod(np.exp((mu_g - .5 * sg**2) * dt + z[:, :, 0] * np.sqrt(dt)), 1)], 1)
    Pb = np.concatenate([np.ones((N, 1)),
                         np.cumprod(np.exp((mu_b - .5 * sb**2) * dt + z[:, :, 1] * np.sqrt(dt)), 1)], 1)
    hold = (lg + rsu * P0) * Pg + lb * Pb
    sell = np.zeros((N, H + 1))
    sold = np.zeros((N, H + 1))
    unv = np.full(N, rsu)
    vs = set(vm)
    for m in range(H + 1):
        if m in vs:
            # Sell this vest at the current GOOG price and move into the basket.
            # Vesting taxes have already occurred, so this is approximately tax-neutral.
            add = spt * P0 * Pg[:, m]
            sold[:, m:] += add[:, None] * (Pb[:, m:] / Pb[:, m][:, None])
            unv -= spt
        sell[:, m] = lg * Pg[:, m] + lb * Pb[:, m] + sold[:, m] + unv * P0 * Pg[:, m]

    def maxdd(p):
        peak = np.maximum.accumulate(p, 1)
        dd = np.divide(peak - p, peak, out=np.zeros_like(p), where=peak != 0)
        return np.max(dd, 1)

    return dict(hold=hold[:, -1], sell=sell[:, -1], hold_dd=maxdd(hold), sell_dd=maxdd(sell))


def breakeven_drift(cfg, base_mu, sg, sb, rho, lo=0.0, hi=0.10, tol=2e-3):
    """Binary-search GOOG annual excess drift where HOLD median equals SELL median."""
    def gap(e):
        r = simulate(cfg, base_mu + e, base_mu, sg, sb, rho)
        return np.median(r["hold"]) - np.median(r["sell"])

    for _ in range(20):
        mid = (lo + hi) / 2
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _stats(x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(p5=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4], std=float(np.std(x, ddof=1)))


def _money(x):
    return f"${x:,.0f}"


def print_report(cfg, sg, sb, rho, base_mu=0.07, N=20000):
    """Print HOLD vs SELL distribution, risk reduction, and breakeven drift."""
    r = simulate(cfg, base_mu, base_mu, sg, sb, rho, N=N)
    hs, ss = _stats(r["hold"]), _stats(r["sell"])
    hdd, sdd = _stats(r["hold_dd"] * 100), _stats(r["sell_dd"] * 100)
    e = breakeven_drift(cfg, base_mu, sg, sb, rho)

    print("== RSU HOLD vs SELL diversification simulation ==")
    print(f"Assumption: mu_GOOG = mu_basket = {base_mu:.1%}/yr | sigma GOOG {sg:.1%}, basket {sb:.1%}, rho {rho:.2f}")
    print(f"Vesting window: month {min(cfg['vest_months'])}-{max(cfg['vest_months'])} | paths {N:,}")
    print("\nTerminal value distribution:")
    rows = pd.DataFrame([
        dict(strategy="HOLD", median=hs["median"], p5=hs["p5"], p25=hs["p25"],
             p75=hs["p75"], p95=hs["p95"], std=hs["std"],
             dd_median=hdd["median"], dd_p95=hdd["p95"]),
        dict(strategy="SELL", median=ss["median"], p5=ss["p5"], p25=ss["p25"],
             p75=ss["p75"], p95=ss["p95"], std=ss["std"],
             dd_median=sdd["median"], dd_p95=sdd["p95"]),
    ])
    pretty = rows.copy()
    for c in ["median", "p5", "p25", "p75", "p95", "std"]:
        pretty[c] = pretty[c].map(_money)
    for c in ["dd_median", "dd_p95"]:
        pretty[c] = pretty[c].map(lambda x: f"{x:.1f}%")
    print(pretty.to_string(index=False))

    upside_giveup = hs["p95"] - ss["p95"]
    downside_improve = ss["p5"] - hs["p5"]
    std_cut = hs["std"] - ss["std"]
    dd_cut = hdd["median"] - sdd["median"]
    print("\nTradeoff:")
    print(f"Relative to HOLD, SELL gives up about {_money(upside_giveup)} at the 95th percentile, "
          f"improves the 5th-percentile downside by about {_money(downside_improve)}, "
          f"cuts standard deviation by {_money(std_cut)}, and lowers median max drawdown by {dd_cut:.1f} percentage points.")
    print(f"Breakeven: GOOG needs to outperform the basket by about +{e:.1%}/yr for HOLD median to catch up to SELL.")

    print("\nNotes:")
    print("- GBM is simplified: no fat tails/regimes; stricter versions could use block bootstrap or Student-t shocks.")
    print("- Sell-on-vest is treated as approximately tax-neutral: RSU vesting already triggers ordinary income tax.")
    print("- Basket proxy and rho materially affect the conclusion; lower rho increases diversification benefit.")
    print("- This script is not investment advice.")


def main():
    cfg, basket_proxy, n_paths, source = _load_config()
    print(f"Config source: {source} | basket proxy: {basket_proxy}")
    try:
        sg, sb, rho, provider = calibrate(basket_proxy)
        print(f"Historical sigma/rho calibration: {provider}")
    except Exception as exc:
        sg, sb, rho = FALLBACK_SIGMA_GOOG, FALLBACK_SIGMA_BASKET, FALLBACK_RHO
        print(f"WARNING: Polygon/yfinance calibration failed; using fallback sigma/rho: {exc}")
    print_report(cfg, sg, sb, rho, N=n_paths)


if __name__ == "__main__":
    main()
