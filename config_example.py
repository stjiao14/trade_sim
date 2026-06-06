HOLDINGS = dict(
    googl_shares=0,            # Direct GOOGL shares
    rsu_unvested_shares=0,     # Unvested RSU shares, treated as economic exposure
    spy_usd=0.0, qqq_usd=0.0, tqqq_usd=0.0,   # ETF market values in USD
)

SALARY_USD = 0.0              # Annual pre-tax salary; set 0 to disable human capital
HC_YEARS   = 0                # Number of years of human capital to include; 0 = disable

# Manual fallback Alphabet weight inside ETFs, GOOGL+GOOG combined.
# Check sources: QQQ -> Invesco holdings page; SPY -> SSGA SPDR holdings page.
FALLBACK_WEIGHTS = dict(spy_alphabet=0.038, qqq_alphabet=0.050)

# Optional local real-holdings CSV path. Put the real path in config_local.py.
HOLDINGS_CSV = ""

# Optional local API config shared by Web UI and CLI. Put real keys in config_local.py.
ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""
POLYGON_API_KEY = ""

# Dummy config for the RSU HOLD vs SELL diversification simulator.
# Put real numbers in config_local.py under VEST_HOLDINGS.
VEST_HOLDINGS = dict(
    goog_price=250.0,           # Current GOOG/GOOGL price, used only to convert RSU value to shares
    rsu_unvested_usd=100_000.0, # Current value of unvested RSU; dummy example, not real money
    liquid_goog_usd=20_000.0,   # Liquid-account GOOG-equivalent exposure; dummy example
    liquid_basket_usd=180_000.0,# Remaining liquid account, treated as diversified basket
)
VEST_MONTHS = list(range(3, 49, 3))
BASKET_PROXY = "VT"
N_PATHS = 20000

# Paper trading / forward-test dummy config. Put real keys in env vars or config_local.py.
PAPER_TRADING = dict(
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
)
