import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta

# =============================
# Streamlit App Config
# =============================
st.set_page_config(page_title="TSX Dividend Portfolio Tracker", layout="wide")
st.title("📈 TSX Dividend Portfolio Tracker — Shares, DRIP & Income Targets")

# =============================
# Default Portfolio (edit these or use the sidebar)
# =============================
default_portfolio = {
    "FFN.TO": {"shares": 4629, "avg_buy": 6.48},
    "EIT-UN.TO": {"shares": 9124, "avg_buy": 13.38},
    "ENB.TO": {"shares": 929, "avg_buy": 64.51},
    "FTS.TO": {"shares": 646, "avg_buy": 69.54},
    "POW.TO": {"shares": 546, "avg_buy": 37.12},
    "PPL.TO": {"shares": 1100, "avg_buy": 50.01},
    "RY.TO": {"shares": 318, "avg_buy": 169.75},
    "SRU-UN.TO": {"shares": 1518, "avg_buy": 26.46},
    "T.TO": {"shares": 2744, "avg_buy": 21.86},
}

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("Portfolio Setup")
portfolio = {}
for ticker, data in default_portfolio.items():
    shares = st.sidebar.number_input(
        f"{ticker} — Shares Owned", min_value=0, value=data["shares"], step=10
    )
    avg_buy = st.sidebar.number_input(
        f"{ticker} — Avg Buy Price (C$)", min_value=0.0, value=float(data["avg_buy"]), step=0.1
    )
    if shares > 0:
        portfolio[ticker] = {"shares": shares, "avg_buy": avg_buy}

st.sidebar.header("Options")
enable_drip = st.sidebar.checkbox("Enable DRIP (Dividend Reinvestment)", value=False)
income_target = st.sidebar.number_input("Target Monthly Income (C$)", min_value=0, value=1000, step=50)

# Refresh button clears caches
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()

# =============================
# Data Fetching Helpers (cached daily)
# =============================
@st.cache_data(ttl=60*60*24)
def fetch_stock_snapshot(portfolio: dict) -> pd.DataFrame:
    rows = []
    for ticker, pos in portfolio.items():
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or 0
        # Dividend per share (fallbacks for robustness)
        div_per_share = (
            info.get("dividendRate")
            or info.get("trailingAnnualDividendRate")
            or 0
        )
        div_yield = info.get("dividendYield") or 0

        shares = pos["shares"]
        avg_buy = pos["avg_buy"]

        market_value = shares * price
        cost_basis = shares * avg_buy
        gain_loss = market_value - cost_basis

        annual_income = shares * div_per_share if div_per_share else 0
        monthly_income = annual_income / 12 if annual_income else 0

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName"),
            "Shares": shares,
            "Price (C$)": price,
            "Market Value (C$)": market_value,
            "Cost Basis (C$)": cost_basis,
            "Unrealized Gain/Loss (C$)": gain_loss,
            "Dividend/Share (C$)": div_per_share,
            "Dividend Yield": div_yield,
            "Est. Annual Income (C$)": annual_income,
            "Est. Monthly Income (C$)": monthly_income,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=60*60*24)
def get_dividend_history(ticker: str) -> pd.Series:
    """Return dividend history series (payment date index -> amount/share)."""
    return yf.Ticker(ticker).dividends

# =============================
# Build Dividend Calendar (Projected 12 months)
# =============================
def build_dividend_calendar(portfolio: dict):
    all_dividends = []
    for ticker, pos in portfolio.items():
        div_hist = get_dividend_history(ticker)
        if div_hist is None or div_hist.empty:
            continue

        # last 12 months as basis
        recent = div_hist[div_hist.index > (div_hist.index.max() - pd.DateOffset(months=12))]
        if recent.empty:
            continue

        avg_div = float(recent.mean())  # average dividend per share per payment
        diffs = recent.index.to_series().diff().dropna().dt.days
        median_gap = float(diffs.median()) if not diffs.empty else 30.0

        # crude frequency detection
        if median_gap < 40:
            freq = "M"  # monthly
            step = pd.DateOffset(months=1)
        elif median_gap < 100:
            freq = "Q"  # quarterly
            step = pd.DateOffset(months=3)
        else:
            freq = "U"  # unknown/irregular -> assume monthly as fallback
            step = pd.DateOffset(months=1)

        last_paid = div_hist.index.max()
        # project next 12 payment dates
        next_date = last_paid
        for _ in range(12):
            next_date = next_date + step
            amount_total = pos["shares"] * avg_div
            all_dividends.append({"Date": next_date, "Ticker": ticker, "Amount": amount_total})

    if not all_dividends:
        return pd.DataFrame(), pd.DataFrame()

    df_divs = pd.DataFrame(all_dividends)
    df_monthly = (
        df_divs.groupby(df_divs["Date"].dt.to_period("M"))
        ["Amount"].sum()
        .reset_index()
    )
    df_monthly["Date"] = df_monthly["Date"].dt.to_timestamp()
    return df_divs.sort_values("Date"), df_monthly.sort_values("Date")

# =============================
# DRIP Simulation (simple projection using average monthly income)
# =============================
def simulate_drip(portfolio: dict, months: int = 12, enabled: bool = False) -> pd.DataFrame:
    """
    Simple DRIP simulation:
    - Estimate average monthly dividend per stock from last 12 months.
    - If DRIP enabled, reinvest monthly income into whole shares at current price.
    - Returns DataFrame with Month and projected Income that month (portfolio total).
    Note: This is a simplified model; actual pay dates/amounts vary.
    """
    # clone holdings so we don't mutate the sidebar inputs
    holdings = {t: {"shares": d["shares"], "avg_buy": d["avg_buy"]} for t, d in portfolio.items()}
    out = []

    for m in range(1, months + 1):
        month_income_total = 0.0
        for ticker, pos in holdings.items():
            div_hist = get_dividend_history(ticker)
            if div_hist is None or div_hist.empty:
                continue
            recent = div_hist[div_hist.index > (div_hist.index.max() - pd.DateOffset(months=12))]
            avg_div_per_payment = float(recent.mean()) if not recent.empty else 0.0

            # convert average per payment to rough monthly amount using detected frequency
            diffs = recent.index.to_series().diff().dropna().dt.days
            median_gap = float(diffs.median()) if not diffs.empty else 30.0
            if median_gap < 40:
                payments_per_year = 12
            elif median_gap < 100:
                payments_per_year = 4
            else:
                payments_per_year = 12  # fallback
            avg_monthly_div_per_share = (avg_div_per_payment * payments_per_year) / 12.0

            income = pos["shares"] * avg_monthly_div_per_share
            month_income_total += income

            # DRIP reinvestment into whole shares at current price
            if enabled and income > 0:
                price = yf.Ticker(ticker).info.get("regularMarketPrice", 0) or 0
                if price > 0:
                    new_shares = int(income // price)
                    if new_shares > 0:
                        pos["shares"] += new_shares
        out.append({"Month": m, "Income": month_income_total})

    return pd.DataFrame(out)

# =============================
# Fetch Snapshot & Build Views
# =============================
df = fetch_stock_snapshot(portfolio)

st.subheader("📋 Portfolio Overview")
st.dataframe(df, use_container_width=True)

# Totals
total_value = float(df["Market Value (C$)"].sum()) if not df.empty else 0.0
total_cost = float(df["Cost Basis (C$)"].sum()) if not df.empty else 0.0
total_gain = float(df["Unrealized Gain/Loss (C$)"].sum()) if not df.empty else 0.0
total_monthly_income = float(df["Est. Monthly Income (C$)"].sum()) if not df.empty else 0.0
total_annual_income = float(df["Est. Annual Income (C$)"].sum()) if not df.empty else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("💵 Portfolio Value", f"C${total_value:,.2f}")
col2.metric("📊 Unrealized Gain/Loss", f"C${total_gain:,.2f}")
col3.metric("💰 Annual Dividend Income", f"C${total_annual_income:,.2f}")

st.metric("📅 Monthly Dividend Income (avg)", f"C${total_monthly_income:,.2f}")

# Charts
st.subheader("Portfolio Allocation by Market Value")
if not df.empty:
    fig_alloc = px.pie(df, values="Market Value (C$)", names="Ticker", title="Allocation by Market Value")
    st.plotly_chart(fig_alloc, use_container_width=True)

st.subheader("Estimated Monthly Dividend Income by Stock")
if not df.empty:
    fig_income = px.bar(
        df, x="Ticker", y="Est. Monthly Income (C$)", color="Ticker", text_auto=True,
        title="Monthly Income per Stock (Projected Average)"
    )
    st.plotly_chart(fig_income, use_container_width=True)

st.subheader("Dividend Yield Comparison")
if not df.empty:
    fig_yield = px.bar(df, x="Ticker", y="Dividend Yield", color="Ticker", text_auto=True, title="Dividend Yields")
    st.plotly_chart(fig_yield, use_container_width=True)

# =============================
# Dividend Calendar (Projected from history)
# =============================
st.subheader("📅 Projected Dividend Calendar (Next 12 Months)")
df_divs, df_monthly = build_dividend_calendar(portfolio)
if not df_divs.empty:
    c1, c2 = st.columns(2)
    with c1:
        st.write("Upcoming Dividend Payments (per stock, projected)")
        st.dataframe(df_divs, use_container_width=True)
    with c2:
        st.write("Monthly Dividend Totals (projected)")
        st.bar_chart(df_monthly.set_index("Date")["Amount"])
else:
    st.info("No dividend history available to build a projection.")

# =============================
# DRIP Simulation & Income Target
# =============================
st.subheader("📈 Dividend Income Projection (DRIP Optional)")
df_drip = simulate_drip(portfolio, months=12, enabled=enable_drip)
if not df_drip.empty:
    st.line_chart(df_drip.set_index("Month")["Income"], height=300)
    st.caption("Projection uses recent dividend history; DRIP buys whole shares at current prices.")

st.subheader("🎯 Income Target Progress")
progress = 0.0
shortfall = 0.0
extra_needed = 0.0

if income_target > 0:
    progress = (total_monthly_income / income_target) * 100 if income_target else 0
    shortfall = max(0.0, income_target - total_monthly_income)

    avg_yield = float(df["Dividend Yield"].mean()) if not df.empty else 0.0
    if avg_yield > 0:
        # Needed capital to cover shortfall at current average yield
        extra_needed = (shortfall * 12.0) / avg_yield

st.metric("Target Monthly Income", f"C${income_target:,.2f}")
st.metric("Current Monthly Income", f"C${total_monthly_income:,.2f}")
st.metric("Progress", f"{progress:.1f}%")
if shortfall > 0:
    st.warning(f"⚠️ Short by C${shortfall:,.2f}/mo. At current average yield, ~C${extra_needed:,.0f} additional capital required.")
else:
    st.success("🎉 You’ve reached or exceeded your monthly income target!")

# =============================
# Price History (6 months)
# =============================
st.subheader("Price History (6 Months)")
for ticker in portfolio:
    hist = yf.download(ticker, period="6mo", interval="1d")
    if not hist.empty:
        st.line_chart(hist["Close"], height=250, use_container_width=True)