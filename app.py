import math
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

APP_VERSION = "MVP3 S&P 500"

st.set_page_config(
    page_title=f"AI Stock Scanner {APP_VERSION}",
    page_icon="📈",
    layout="wide"
)

DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","AMD","ORCL","CRM",
    "ADBE","NFLX","COST","WMT","V","MA","JPM","BRK-B","LLY","UNH"
]


@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500_tickers():
    """
    Load the current S&P 500 constituent list using multiple public sources.

    Source priority:
    1) GitHub CSV mirror of the S&P 500 constituent list
    2) Wikipedia HTML table
    3) Built-in fallback watchlist only if both public sources fail

    Returns:
        (tickers, error_message)
    """
    errors = []

    # Source 1: CSV mirror. This avoids HTML parser issues that can occur
    # on Streamlit Cloud with pd.read_html().
    csv_url = (
        "https://raw.githubusercontent.com/datasets/"
        "s-and-p-500-companies/master/data/constituents.csv"
    )
    try:
        sp500 = pd.read_csv(csv_url)
        if "Symbol" not in sp500.columns:
            raise ValueError("CSV source does not contain a Symbol column")

        symbols = (
            sp500["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        symbols = [s.replace(".", "-") for s in symbols if s]

        # Sanity check: a valid S&P 500 list should contain roughly 500 names.
        if len(symbols) < 450:
            raise ValueError(f"CSV source returned only {len(symbols)} symbols")

        return list(dict.fromkeys(symbols)), None
    except Exception as exc:
        errors.append(f"GitHub CSV: {type(exc).__name__}: {exc}")

    # Source 2: Wikipedia.
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(wiki_url)
        if not tables or "Symbol" not in tables[0].columns:
            raise ValueError("Wikipedia table does not contain a Symbol column")

        symbols = (
            tables[0]["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        symbols = [s.replace(".", "-") for s in symbols if s]

        if len(symbols) < 450:
            raise ValueError(f"Wikipedia returned only {len(symbols)} symbols")

        return list(dict.fromkeys(symbols)), None
    except Exception as exc:
        errors.append(f"Wikipedia: {type(exc).__name__}: {exc}")

    # Last-resort fallback so the app still runs.
    return DEFAULT_TICKERS.copy(), " | ".join(errors)

# -----------------------------
# Utility helpers
# -----------------------------
def safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

def clamp(x, lo=0.0, hi=100.0):
    x = safe_float(x)
    if x is None:
        return lo
    return max(lo, min(hi, x))

def score_higher_better(value, bands):
    if value is None:
        return 0.0
    score = 0.0
    for threshold, pts in bands:
        if value >= threshold:
            score = pts
    return score

def score_lower_better(value, bands):
    if value is None:
        return 0.0
    for max_value, pts in bands:
        if value <= max_value:
            return pts
    return 0.0

def normalize_tickers(text):
    result = []
    for raw in text.replace("\n", ",").replace(";", ",").split(","):
        t = raw.strip().upper()
        if t and t not in result:
            result.append(t)
    return result

def weighted_average(values):
    valid = [(v, w) for v, w in values if v is not None and w > 0]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_w

# -----------------------------
# Data layer
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker):
    t = yf.Ticker(ticker)

    info = {}
    fast = {}
    hist = pd.DataFrame()

    try:
        info = t.info or {}
    except Exception:
        pass

    try:
        fast = dict(t.fast_info)
    except Exception:
        pass

    try:
        hist = t.history(period="2y", auto_adjust=True)
    except Exception:
        pass

    price = safe_float(fast.get("last_price")) or safe_float(info.get("currentPrice"))
    if price is None and not hist.empty:
        price = safe_float(hist["Close"].iloc[-1])

    ma50 = ma200 = ret_6m = ret_12m = volatility = None
    if not hist.empty:
        close = hist["Close"].dropna()
        if len(close) >= 50:
            ma50 = safe_float(close.tail(50).mean())
        if len(close) >= 200:
            ma200 = safe_float(close.tail(200).mean())
        if len(close) >= 126:
            ret_6m = safe_float(close.iloc[-1] / close.iloc[-126] - 1)
        if len(close) >= 252:
            ret_12m = safe_float(close.iloc[-1] / close.iloc[-252] - 1)
        rets = close.pct_change().dropna()
        if len(rets) > 20:
            volatility = safe_float(rets.std() * np.sqrt(252))

    data = {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector") or "Unknown",
        "price": price,
        "market_cap": safe_float(info.get("marketCap")),
        "revenue_growth": safe_float(info.get("revenueGrowth")),
        "earnings_growth": safe_float(info.get("earningsGrowth")),
        "gross_margin": safe_float(info.get("grossMargins")),
        "operating_margin": safe_float(info.get("operatingMargins")),
        "profit_margin": safe_float(info.get("profitMargins")),
        "roe": safe_float(info.get("returnOnEquity")),
        "debt_to_equity": safe_float(info.get("debtToEquity")),
        "current_ratio": safe_float(info.get("currentRatio")),
        "free_cash_flow": safe_float(info.get("freeCashflow")),
        "operating_cash_flow": safe_float(info.get("operatingCashflow")),
        "forward_pe": safe_float(info.get("forwardPE")),
        "trailing_pe": safe_float(info.get("trailingPE")),
        "peg": safe_float(info.get("pegRatio")),
        "price_to_sales": safe_float(info.get("priceToSalesTrailing12Months")),
        "beta": safe_float(info.get("beta")),
        "target_mean": safe_float(info.get("targetMeanPrice")),
        "target_low": safe_float(info.get("targetLowPrice")),
        "target_high": safe_float(info.get("targetHighPrice")),
        "recommendation": str(info.get("recommendationKey") or "").lower(),
        "num_analysts": safe_float(info.get("numberOfAnalystOpinions")),
        "52w_high": safe_float(info.get("fiftyTwoWeekHigh")),
        "52w_low": safe_float(info.get("fiftyTwoWeekLow")),
        "ma50": ma50,
        "ma200": ma200,
        "ret_6m": ret_6m,
        "ret_12m": ret_12m,
        "volatility": volatility,
    }

    required = [
        "price","revenue_growth","earnings_growth","operating_margin",
        "free_cash_flow","forward_pe","debt_to_equity","target_mean"
    ]
    available = sum(data.get(k) is not None for k in required)
    data["data_quality_pct"] = round(available / len(required) * 100, 1)
    return data

# -----------------------------
# Scoring model
# -----------------------------
def quality_score(d):
    margin = score_higher_better(
        d["operating_margin"],
        [(-1,0),(0,5),(0.10,12),(0.20,20),(0.30,25)]
    )
    roe = score_higher_better(
        d["roe"],
        [(-1,0),(0,3),(0.10,8),(0.20,14),(0.30,18)]
    )
    cash = 20 if (d["free_cash_flow"] is not None and d["free_cash_flow"] > 0) else 0
    debt = score_lower_better(
        d["debt_to_equity"],
        [(30,17),(60,14),(100,10),(200,6),(400,2)]
    )
    liquidity = score_higher_better(
        d["current_ratio"],
        [(0,0),(0.8,4),(1.0,8),(1.5,10)]
    )
    return clamp(margin + roe + cash + debt + liquidity)

def growth_score(d):
    rev = score_higher_better(
        d["revenue_growth"],
        [(-1,0),(0,5),(0.05,10),(0.10,16),(0.20,23),(0.30,28)]
    )
    eps = score_higher_better(
        d["earnings_growth"],
        [(-1,0),(0,5),(0.05,10),(0.10,16),(0.20,23),(0.30,28)]
    )
    margin = score_higher_better(
        d["operating_margin"],
        [(-1,0),(0,4),(0.10,8),(0.20,12)]
    )
    fcf = 17 if (d["free_cash_flow"] is not None and d["free_cash_flow"] > 0) else 0
    momentum = 0
    if d["ret_6m"] is not None:
        momentum += score_higher_better(
            d["ret_6m"],
            [(-1,0),(0,3),(0.10,6),(0.20,9)]
        )
    if d["ret_12m"] is not None:
        momentum += score_higher_better(
            d["ret_12m"],
            [(-1,0),(0,2),(0.10,4),(0.20,6)]
        )
    return clamp(rev + eps + margin + fcf + momentum)

def stability_score(d):
    debt = score_lower_better(
        d["debt_to_equity"],
        [(30,22),(60,18),(100,13),(200,7),(400,3)]
    )
    beta = score_lower_better(
        d["beta"],
        [(0.8,16),(1.0,13),(1.2,10),(1.5,6),(3.0,2)]
    )
    vol = score_lower_better(
        d["volatility"],
        [(0.20,20),(0.30,16),(0.40,11),(0.60,6),(2.0,2)]
    )
    margin = score_higher_better(
        d["profit_margin"],
        [(-1,0),(0,5),(0.10,15),(0.20,24)]
    )
    cash = 18 if (d["free_cash_flow"] is not None and d["free_cash_flow"] > 0) else 0
    return clamp(debt + beta + vol + margin + cash)

def valuation_score(d):
    pe = d["forward_pe"] or d["trailing_pe"]
    peg = d["peg"]
    ps = d["price_to_sales"]

    components = []

    if pe is not None:
        pe_pts = score_lower_better(pe, [(12,35),(18,31),(25,25),(35,17),(50,9),(80,4)])
        components.append((pe_pts / 35 * 100, 0.45))

    if peg is not None:
        peg_pts = score_lower_better(peg, [(1.0,30),(1.5,25),(2.0,19),(3.0,11),(5.0,5)])
        components.append((peg_pts / 30 * 100, 0.30))

    if ps is not None:
        ps_pts = score_lower_better(ps, [(2,25),(4,20),(7,14),(12,8),(25,3)])
        components.append((ps_pts / 25 * 100, 0.25))

    return clamp(weighted_average(components) or 0)

def future_growth_score(d):
    rev = score_higher_better(
        d["revenue_growth"],
        [(-1,0),(0,4),(0.05,9),(0.10,14),(0.20,20)]
    )
    eps = score_higher_better(
        d["earnings_growth"],
        [(-1,0),(0,4),(0.05,9),(0.10,14),(0.20,20)]
    )

    upside = None
    if d["target_mean"] and d["price"] and d["price"] > 0:
        upside = d["target_mean"] / d["price"] - 1

    upside_pts = score_higher_better(
        upside,
        [(-1,0),(0,5),(0.10,10),(0.20,15),(0.30,20)]
    )

    rec_pts = {
        "strong_buy": 20,
        "buy": 16,
        "hold": 9,
        "underperform": 3,
        "sell": 0
    }.get(d["recommendation"], 7)

    coverage_pts = score_higher_better(
        d["num_analysts"],
        [(0,0),(5,5),(10,10),(20,15),(30,20)]
    )

    return clamp(rev + eps + upside_pts + rec_pts + coverage_pts)

# -----------------------------
# Red-flag engine
# -----------------------------
def red_flag_engine(d):
    flags = []
    level = 0

    if d["data_quality_pct"] < 50:
        flags.append("Dữ liệu thiếu nhiều")
        level = max(level, 2)

    if d["free_cash_flow"] is not None and d["free_cash_flow"] < 0:
        flags.append("FCF âm")
        level = max(level, 1)

    if d["debt_to_equity"] is not None and d["debt_to_equity"] > 250:
        flags.append("Nợ/Equity cao")
        level = max(level, 2)

    if d["revenue_growth"] is not None and d["revenue_growth"] < -0.10:
        flags.append("Doanh thu giảm >10%")
        level = max(level, 2)

    if d["earnings_growth"] is not None and d["earnings_growth"] < -0.20:
        flags.append("Lợi nhuận giảm mạnh")
        level = max(level, 2)

    if d["operating_margin"] is not None and d["operating_margin"] < 0:
        flags.append("Operating margin âm")
        level = max(level, 2)

    pe = d["forward_pe"] or d["trailing_pe"]
    if pe is not None and pe > 80:
        flags.append("P/E rất cao")
        level = max(level, 1)

    if not flags:
        return 0, "Không thấy red flag định lượng lớn"
    return level, "; ".join(flags)

# -----------------------------
# Fair-value / buy-zone engine
# -----------------------------
def fair_value_engine(d, group):
    price = d["price"]
    analyst = d["target_mean"]
    pe = d["forward_pe"] or d["trailing_pe"]

    # Method 1: Analyst consensus
    analyst_value = analyst if analyst and analyst > 0 else None

    # Method 2: Earnings multiple normalization
    multiple_value = None
    if price and pe and pe > 0:
        normalized_pe = 28 if group == "Growth" else 22
        multiple_value = price * normalized_pe / pe

    # Method 3: Conservative momentum/quality anchor
    technical_value = None
    if price:
        anchor = price
        if d["ma200"]:
            anchor = (price + d["ma200"]) / 2
        quality_adjustment = 1.05 if group == "Growth" else 1.03
        technical_value = anchor * quality_adjustment

    fair = weighted_average([
        (analyst_value, 0.45),
        (multiple_value, 0.35),
        (technical_value, 0.20),
    ])

    if fair is None:
        return None, None, None, None

    if group == "Growth":
        buy_low, buy_high = fair * 0.82, fair * 0.90
        excellent = fair * 0.75
    else:
        buy_low, buy_high = fair * 0.88, fair * 0.95
        excellent = fair * 0.82

    return fair, excellent, buy_low, buy_high

def technical_label(d):
    p, m50, m200 = d["price"], d["ma50"], d["ma200"]
    if p is None:
        return "N/A"
    if m50 and m200:
        if p > m50 > m200:
            return "Uptrend"
        if p < m50 < m200:
            return "Downtrend"
    return "Mixed"

def position_sizing(price, account_value, risk_pct, stop_pct, max_position_pct):
    if not price or price <= 0:
        return None, None, None
    risk_dollars = account_value * risk_pct
    stop_price = price * (1 - stop_pct)
    risk_per_share = price - stop_price
    if risk_per_share <= 0:
        return None, None, None

    shares_by_risk = math.floor(risk_dollars / risk_per_share)
    max_position_dollars = account_value * max_position_pct
    shares_by_cap = math.floor(max_position_dollars / price)

    shares = max(0, min(shares_by_risk, shares_by_cap))
    position_value = shares * price
    return shares, stop_price, position_value

def scan_one(ticker, account_value, risk_pct):
    d = fetch_stock_data(ticker)

    quality = quality_score(d)
    growth = growth_score(d)
    stability = stability_score(d)
    valuation = valuation_score(d)
    future = future_growth_score(d)

    group = "Growth" if growth >= stability else "Stable"

    if group == "Growth":
        total = (
            0.25 * growth +
            0.20 * quality +
            0.20 * future +
            0.15 * valuation +
            0.10 * stability +
            0.10 * d["data_quality_pct"]
        )
        max_position_pct = 0.075
        stop_pct = 0.10
    else:
        total = (
            0.25 * quality +
            0.20 * stability +
            0.20 * valuation +
            0.15 * future +
            0.10 * growth +
            0.10 * d["data_quality_pct"]
        )
        max_position_pct = 0.10
        stop_pct = 0.08

    red_level, red_text = red_flag_engine(d)
    fair, excellent, buy_low, buy_high = fair_value_engine(d, group)

    status = "WATCH"
    if red_level >= 2:
        status = "REVIEW"
    elif d["price"] and fair and buy_low and buy_high:
        p = d["price"]
        if p <= excellent:
            status = "EXCELLENT ZONE"
        elif buy_low <= p <= buy_high:
            status = "BUY ZONE"
        elif p < buy_low:
            status = "VALUE ZONE"
        elif p <= fair:
            status = "NEAR BUY"
        else:
            status = "WATCH"

    shares, stop_price, position_value = position_sizing(
        d["price"], account_value, risk_pct, stop_pct, max_position_pct
    )

    return {
        "Ticker": ticker,
        "Company": d["name"],
        "Sector": d["sector"],
        "Group": group,
        "Price": d["price"],
        "Total Score": round(total, 1),
        "Quality": round(quality, 1),
        "Growth": round(growth, 1),
        "Stability": round(stability, 1),
        "Future Growth": round(future, 1),
        "Valuation": round(valuation, 1),
        "Data Quality %": d["data_quality_pct"],
        "Fair Value Est.": fair,
        "Excellent Zone": excellent,
        "Buy Zone Low": buy_low,
        "Buy Zone High": buy_high,
        "Status": status,
        "Red Flag Level": red_level,
        "Red Flags": red_text,
        "Technical": technical_label(d),
        "Analyst Target": d["target_mean"],
        "Suggested Shares": shares,
        "Suggested Stop": stop_price,
        "Position Value": position_value,
        "6M Return": d["ret_6m"],
        "12M Return": d["ret_12m"],
    }

# -----------------------------
# UI
# -----------------------------
st.title(f"📈 AI Stock Scanner — {APP_VERSION}")
st.caption(
    "Research prototype for screening U.S. stocks. "
    "It does not guarantee returns and does not place live trades."
)

with st.sidebar:
    st.header("Scanner Settings")

    universe = st.radio(
        "Universe",
        ["S&P 500", "Custom tickers"],
        index=0,
        help="S&P 500 loads the current public constituent list. Custom tickers lets you scan your own list."
    )

    ticker_text = st.text_area(
        "Custom tickers",
        value=", ".join(DEFAULT_TICKERS),
        height=180,
        disabled=(universe == "S&P 500")
    )

    account_value = st.number_input(
        "Portfolio value ($)",
        min_value=1000.0,
        value=100000.0,
        step=5000.0
    )

    risk_pct = st.slider(
        "Risk per trade",
        min_value=0.25,
        max_value=1.00,
        value=0.75,
        step=0.05
    ) / 100.0

    min_score = st.slider("Minimum score", 0, 100, 65)
    max_names = st.slider("Maximum rows", 5, 100, 30)

    run = st.button("Run Scanner", type="primary", use_container_width=True)

if not run:
    st.markdown(
        """
### MVP3 S&P 500 includes
- S&P 500 universe mode plus custom-ticker mode
- Live/public market and fundamental data via `yfinance`
- Data-quality validation
- Growth and Stable scoring models
- Future-growth and valuation scoring
- Preliminary fair-value and buy-zone engine
- Red-flag system
- Position sizing and CSV export

**Important:** MVP3 is a screening and research tool. Its fair-value model is still preliminary,
not a full institutional DCF. A 30–70% return is a research target to investigate, not a promised
or forced outcome. Validate finalists with filings and deeper valuation before using real money.
        """
    )
    st.stop()

if universe == "S&P 500":
    tickers, sp500_error = load_sp500_tickers()
    if sp500_error:
        st.warning(
            "Could not refresh the full S&P 500 constituent list, so the app is using "
            "the built-in fallback watchlist for this run."
        )
        with st.expander("S&P 500 load error details"):
            st.code(sp500_error)
else:
    tickers = normalize_tickers(ticker_text)

if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

rows = []
progress = st.progress(0, text="Loading data...")

for i, ticker in enumerate(tickers):
    try:
        rows.append(scan_one(ticker, account_value, risk_pct))
    except Exception as e:
        rows.append({
            "Ticker": ticker,
            "Company": ticker,
            "Sector": "Unknown",
            "Group": "N/A",
            "Price": None,
            "Total Score": 0,
            "Quality": 0,
            "Growth": 0,
            "Stability": 0,
            "Future Growth": 0,
            "Valuation": 0,
            "Data Quality %": 0,
            "Fair Value Est.": None,
            "Excellent Zone": None,
            "Buy Zone Low": None,
            "Buy Zone High": None,
            "Status": "DATA ERROR",
            "Red Flag Level": 3,
            "Red Flags": str(e),
            "Technical": "N/A",
            "Analyst Target": None,
            "Suggested Shares": None,
            "Suggested Stop": None,
            "Position Value": None,
            "6M Return": None,
            "12M Return": None,
        })
    progress.progress((i + 1) / len(tickers), text=f"Scanning {ticker}...")

progress.empty()

df = pd.DataFrame(rows)
qualified = (
    df[df["Total Score"] >= min_score]
    .sort_values(["Red Flag Level", "Total Score"], ascending=[True, False])
    .head(max_names)
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scanned", len(df))
c2.metric("Qualified", len(qualified))
c3.metric(
    "Buy/Value Zones",
    int(qualified["Status"].isin(["EXCELLENT ZONE","BUY ZONE","VALUE ZONE"]).sum())
)
c4.metric("Review", int((df["Red Flag Level"] >= 2).sum()))
c5.metric("Avg Data Quality", f"{df['Data Quality %'].mean():.0f}%")

st.subheader("Scanner Results")

show_cols = [
    "Ticker","Group","Price","Total Score","Quality","Growth","Stability",
    "Future Growth","Valuation","Data Quality %","Fair Value Est.",
    "Buy Zone Low","Buy Zone High","Status","Red Flag Level",
    "Suggested Shares","Suggested Stop","Position Value"
]

st.dataframe(
    qualified[show_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Fair Value Est.": st.column_config.NumberColumn(format="$%.2f"),
        "Buy Zone Low": st.column_config.NumberColumn(format="$%.2f"),
        "Buy Zone High": st.column_config.NumberColumn(format="$%.2f"),
        "Suggested Stop": st.column_config.NumberColumn(format="$%.2f"),
        "Position Value": st.column_config.NumberColumn(format="$%.0f"),
    }
)

st.download_button(
    "Download Full Results CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"AI_Stock_Scanner_MVP3_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

st.subheader("Stock Detail")

detail_list = qualified["Ticker"].tolist() if len(qualified) else df["Ticker"].tolist()
selected = st.selectbox("Select ticker", detail_list)
row = df[df["Ticker"] == selected].iloc[0]

left, right = st.columns(2)

with left:
    st.write(f"**Company:** {row['Company']}")
    st.write(f"**Sector:** {row['Sector']}")
    st.write(f"**Style:** {row['Group']}")
    st.write(f"**Status:** {row['Status']}")
    st.write(f"**Technical:** {row['Technical']}")
    st.write(f"**Data quality:** {row['Data Quality %']:.0f}%")

with right:
    st.write(f"**Red flag level:** {int(row['Red Flag Level'])}")
    st.write(f"**Red flags:** {row['Red Flags']}")
    if pd.notna(row["Analyst Target"]):
        st.write(f"**Analyst target mean:** ${row['Analyst Target']:.2f}")
    if pd.notna(row["Suggested Shares"]):
        st.write(f"**Suggested max shares (risk model):** {int(row['Suggested Shares'])}")
    if pd.notna(row["Suggested Stop"]):
        st.write(f"**Suggested stop reference:** ${row['Suggested Stop']:.2f}")

st.warning(
    "MVP3 S&P 500 is a research tool. It does not guarantee returns, "
    "does not replace professional advice, and does not execute trades. "
    "Use paper trading and validation before considering live capital."
)
