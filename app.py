import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import ta
import feedparser
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy import stats
from scipy.stats import norm as sp_norm
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from seq2seq_lstm import Seq2SeqLSTM, make_multi_sequences

SEQ_LEN = 30
PEERS   = ["AMZN", "MSFT", "GOOGL", "META", "AAPL", "SPY"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMZN · Intelligence Terminal",
    page_icon="▸",
    layout="wide",
)

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">
<style>
/* ── Root variables ──────────────────────────── */
:root {
    --bg0:    #05080f;
    --bg1:    #090d18;
    --bg2:    #0d1220;
    --bg3:    #111829;
    --bg4:    #162035;
    --b0:     #192540;
    --b1:     #1e2e4e;
    --b2:     #243660;
    --t0:     #d8e3f5;
    --t1:     #7d93b8;
    --t2:     #3d5070;
    --blue:   #3b82f6;
    --teal:   #10b981;
    --amber:  #f59e0b;
    --red:    #ef4444;
    --purple: #8b5cf6;
    --blueglow: rgba(59,130,246,0.12);
    --tealglow: rgba(16,185,129,0.10);
    --redglow:  rgba(239,68,68,0.10);
}

/* ── Base ────────────────────────────────────── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg0) !important;
    background-image:
        linear-gradient(rgba(59,130,246,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.018) 1px, transparent 1px);
    background-size: 48px 48px;
}

[data-testid="stHeader"] {
    background: rgba(5,8,15,0.92) !important;
    border-bottom: 1px solid var(--b0) !important;
    backdrop-filter: blur(12px);
}

#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Scrollbar ───────────────────────────────── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--b1); border-radius: 2px; }

/* ── Terminal header ─────────────────────────── */
.t-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 14px;
    border-bottom: 1px solid var(--b0);
    margin-bottom: 20px;
}
.t-ticker-block { display: flex; align-items: baseline; gap: 12px; }
.t-symbol {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--t0);
    letter-spacing: -0.02em;
}
.t-name {
    font-size: 0.9rem;
    color: var(--t1);
    font-weight: 400;
}
.t-exch {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--t2);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border: 1px solid var(--b0);
    padding: 2px 7px;
    border-radius: 3px;
}
.t-right { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.t-price-main {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--t0);
    letter-spacing: -0.03em;
}
.t-change-up {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--teal);
}
.t-change-down {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--red);
}
.t-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--t2);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    padding: 14px 0 0;
}
.t-meta span { margin-right: 20px; }

/* ── Live indicator ──────────────────────────── */
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.live-dot {
    width: 5px; height: 5px;
    background: var(--teal);
    border-radius: 50%;
    animation: livepulse 1.8s infinite;
}
@keyframes livepulse {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50% { opacity:0.6; box-shadow: 0 0 0 5px rgba(16,185,129,0); }
}

/* ── KPI cards ───────────────────────────────── */
.kpi-grid { display: flex; gap: 10px; margin-bottom: 24px; }
.kpi-card {
    flex: 1;
    position: relative;
    background: var(--bg2);
    border: 1px solid var(--b0);
    border-radius: 3px;
    padding: 14px 16px 12px;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--blue);
    opacity: 0.7;
}
.kpi-card.pos::after { background: var(--teal); }
.kpi-card.neg::after { background: var(--red); }
.kpi-card.neu::after { background: var(--amber); }

.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    font-weight: 700;
    color: var(--t2);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 9px;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--t0);
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 7px;
}
.kpi-delta { font-family: 'Space Mono', monospace; font-size: 0.7rem; }
.kpi-delta.up   { color: var(--teal); }
.kpi-delta.down { color: var(--red); }
.kpi-delta.neu  { color: var(--t1); }

/* ── Tabs ────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--b0) !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--t2) !important;
    padding: 10px 16px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    color: var(--t1) !important;
    background: var(--bg2) !important;
}

/* ── Section headers ─────────────────────────── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 26px 0 14px;
}
.sec-head-bar {
    width: 3px; height: 14px;
    background: var(--blue);
    border-radius: 2px;
    flex-shrink: 0;
}
.sec-head-bar.teal  { background: var(--teal); }
.sec-head-bar.amber { background: var(--amber); }
.sec-head-bar.red   { background: var(--red); }
.sec-head-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--t1);
    text-transform: uppercase;
    letter-spacing: 0.14em;
}
.sec-head-line {
    flex: 1;
    height: 1px;
    background: var(--b0);
}

/* ── Signal / insight boxes ──────────────────── */
.signal {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: var(--bg2);
    border: 1px solid var(--b0);
    border-left: 2px solid var(--blue);
    border-radius: 0 3px 3px 0;
    padding: 10px 14px;
    margin: 10px 0;
    font-size: 0.85rem;
    color: var(--t1);
    line-height: 1.55;
}
.signal-icon {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--blue);
    margin-top: 3px;
    flex-shrink: 0;
}
.signal.bull { border-left-color: var(--teal); background: rgba(16,185,129,0.04); }
.signal.bull .signal-icon { color: var(--teal); }
.signal.bear { border-left-color: var(--red); background: rgba(239,68,68,0.04); }
.signal.bear .signal-icon { color: var(--red); }
.signal.warn { border-left-color: var(--amber); background: rgba(245,158,11,0.04); }
.signal.warn .signal-icon { color: var(--amber); }

/* ── Info / arch blocks ──────────────────────── */
.info-block {
    background: var(--bg2);
    border: 1px solid var(--b0);
    border-radius: 3px;
    padding: 18px 20px;
    margin-bottom: 10px;
    font-size: 0.85rem;
    line-height: 1.7;
    color: var(--t1);
}
.info-block .ib-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--t0);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--b0);
}
.info-block code {
    background: var(--bg4);
    border: 1px solid var(--b0);
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.8em;
    color: var(--amber);
    font-family: 'Space Mono', monospace;
}

/* ── Streamlit metric override ───────────────── */
[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--b0) !important;
    border-radius: 3px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.58rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--t2) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.25rem !important;
    color: var(--t0) !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Form widgets ────────────────────────────── */
[data-testid="stSelectbox"] label p,
[data-testid="stSlider"] label p,
[data-testid="stCheckbox"] label p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--t2) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--bg2) !important;
    border-color: var(--b0) !important;
    border-radius: 3px !important;
}

/* ── Dataframes ──────────────────────────────── */
[data-testid="stDataFrame"] iframe { border-radius: 3px; }

/* ── Expander ────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--b0) !important;
    border-radius: 3px !important;
    background: var(--bg1) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--t1) !important;
}

/* ── Subheader override ──────────────────────── */
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--t1) !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid var(--b0) !important;
    margin-top: 24px !important;
}
/* Caption */
[data-testid="stCaptionContainer"] p {
    color: var(--t2) !important;
    font-size: 0.76rem !important;
}
/* Alert/info */
[data-testid="stAlert"] {
    border-radius: 3px !important;
}

/* ── Divider ─────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--b0) !important;
    margin: 10px 0 !important;
}
</style>
""")

# ── Chart style helper ────────────────────────────────────────────────────────
def _c(fig, height=None, legend_h=False):
    """Apply terminal chart style."""
    fig.update_layout(
        paper_bgcolor="#05080f",
        plot_bgcolor="#05080f",
        font=dict(family="Space Mono, monospace", color="#3d5070", size=10),
        xaxis=dict(gridcolor="#111829", linecolor="#192540",
                   tickfont=dict(color="#3d5070", size=9), showgrid=True),
        yaxis=dict(gridcolor="#111829", linecolor="#192540",
                   tickfont=dict(color="#3d5070", size=9), showgrid=True),
        hoverlabel=dict(bgcolor="#0d1220", bordercolor="#1e2e4e",
                        font=dict(family="Space Mono, monospace",
                                  color="#d8e3f5", size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#192540",
                    font=dict(family="Space Mono, monospace", size=9,
                              color="#7d93b8"),
                    orientation="h" if legend_h else "v",
                    y=1.04 if legend_h else None),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30 if legend_h else 20, b=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig

def _sec(title, accent="blue"):
    return f'<div class="sec-head"><div class="sec-head-bar {accent}"></div><span class="sec-head-text">{title}</span><div class="sec-head-line"></div></div>'

def _sig(text, variant=""):
    icon = "▶" if not variant else ("▲" if variant == "bull" else ("▼" if variant == "bear" else "◆"))
    return f'<div class="signal {variant}"><span class="signal-icon">{icon}</span><span>{text}</span></div>'

# ── Data loading ──────────────────────────────────────────────────────────────
_CSV = "amazon_stock.csv"

def _csv_fallback():
    """Load from bundled CSV when yfinance is rate-limited."""
    df = pd.read_csv(_CSV)
    # normalise column names to match yfinance output
    df.columns = [c.strip().title().replace("_", "") for c in df.columns]
    # 'Date' or 'Adjclose' etc.
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col and date_col != "Date":
        df.rename(columns={date_col: "Date"}, inplace=True)
    if "Adjclose" in df.columns:
        df.rename(columns={"Adjclose": "Adj Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    df.set_index("Date", inplace=True)
    return df

def _yf_fetch(ticker, period):
    d = yf.Ticker(ticker).history(period=period)
    d.reset_index(inplace=True)
    if "Date" in d.columns:
        d.set_index("Date", inplace=True)
    elif "Datetime" in d.columns:
        d.rename(columns={"Datetime": "Date"}, inplace=True)
        d.set_index("Date", inplace=True)
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d

@st.cache_data(ttl=3600)
def load_data():
    try:
        return _yf_fetch("AMZN", "2y")
    except Exception:
        try:
            return _csv_fallback()
        except Exception:
            st.error("Unable to load AMZN data — yfinance rate-limited and no local CSV found.")
            st.stop()

@st.cache_data(ttl=3600)
def load_peers():
    """Load peer Close prices. Returns whatever tickers succeed; never raises."""
    frames = {}
    for t in [p for p in PEERS if p != "AMZN"]:
        try:
            d = _yf_fetch(t, "1y")
            frames[t] = d["Close"]
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    peer_df = pd.DataFrame(frames).ffill().dropna(how="all")
    return peer_df

@st.cache_data(ttl=1800)
def load_news():
    feed = feedparser.parse("https://finance.yahoo.com/rss/headline?s=AMZN")
    return [e for e in feed.entries if "amazon" in e.title.lower() or "amzn" in e.title.lower()]

df   = load_data()
news = load_news()

# ── Technical indicators ──────────────────────────────────────────────────────
df["RSI"]         = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
_macd             = ta.trend.MACD(df["Close"])
df["MACD"]        = _macd.macd()
df["MACD_Signal"] = _macd.macd_signal()
df["MACD_Hist"]   = _macd.macd_diff()
_bb               = ta.volatility.BollingerBands(df["Close"])
df["BB_upper"]    = _bb.bollinger_hband()
df["BB_lower"]    = _bb.bollinger_lband()
df["BB_mid"]      = _bb.bollinger_mavg()
df["MA7"]         = df["Close"].rolling(7).mean()
df["MA30"]        = df["Close"].rolling(30).mean()
df["MA90"]        = df["Close"].rolling(90).mean()
df["Returns"]     = df["Close"].pct_change()
df["Volatility"]  = df["Returns"].rolling(30).std() * np.sqrt(252) * 100
df["BB_width"]    = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
df["MA_ratio"]    = df["MA7"] / df["MA30"]
df["Vol_ratio"]      = df["Volume"] / df["Volume"].rolling(20).mean()
df["High_Low_ratio"] = (df["High"] - df["Low"]) / df["Close"]

# ── Model definition ──────────────────────────────────────────────────────────
class OneStepLSTM(nn.Module):
    def __init__(self, inp_size=5):
        super().__init__()
        self.lstm = nn.LSTM(inp_size, 64, batch_first=True)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Forecast helpers ──────────────────────────────────────────────────────────
def forecast_lr(df, n_days):
    data = df[-120:].copy()
    data["ord"] = data.index.map(pd.Timestamp.toordinal)
    X, y = data["ord"].values.reshape(-1, 1), data["Close"].values
    mdl = LinearRegression().fit(X, y)
    resid_std = (y - mdl.predict(X)).std()
    future = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=n_days)
    Xf = future.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    mu = mdl.predict(Xf)
    ci = resid_std * np.sqrt(1 + np.arange(1, n_days + 1) / len(X))
    return (pd.Series(mu, index=future),
            pd.Series(mu - 1.96 * ci, index=future),
            pd.Series(mu + 1.96 * ci, index=future))

def forecast_lstm(df, seq_len, n_days):
    feats  = df[["Open", "High", "Low", "Close", "Volume"]].values
    scaler = MinMaxScaler().fit(feats)
    norm   = scaler.transform(feats)
    mdl = OneStepLSTM(inp_size=5).cpu()
    mdl.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu", weights_only=False))
    mdl.eval()
    X_h, y_h = make_multi_sequences(norm, seq_len=seq_len, horizon=1)
    with torch.no_grad():
        ph = mdl(X_h.float()).numpy().flatten()
    dummy4    = np.zeros((len(ph), 4))
    inv_ph    = scaler.inverse_transform(np.hstack([dummy4, ph.reshape(-1, 1)]))[:, -1]
    inv_yh    = scaler.inverse_transform(np.hstack([dummy4, y_h.numpy().reshape(-1, 1)]))[:, -1]
    resid_std = (inv_yh - inv_ph).std()
    preds, seq = [], norm[-seq_len:].copy()
    for _ in range(n_days):
        with torch.no_grad():
            p = mdl(torch.tensor(seq).unsqueeze(0).float()).item()
        preds.append(p)
        row = seq[-1].copy(); row[3] = p
        seq = np.vstack([seq[1:], row])
    inv = scaler.inverse_transform(
        np.hstack([np.zeros((n_days, 4)), np.array(preds).reshape(-1, 1)])
    )[:, -1]
    future = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    mu = pd.Series(inv, index=future)
    ci = resid_std * np.sqrt(np.arange(1, n_days + 1))
    return mu, mu - 1.96 * ci, mu + 1.96 * ci

def forecast_seq2seq(df, seq_len, n_days):
    closes = df[["Close"]].values
    scaler = MinMaxScaler().fit(closes)
    norm   = scaler.transform(closes)
    X, _   = make_multi_sequences(norm, seq_len=seq_len, horizon=n_days)
    mdl    = Seq2SeqLSTM(input_dim=1).cpu()
    mdl.load_state_dict(torch.load("seq2seq_lstm.pth", map_location="cpu", weights_only=False))
    mdl.eval()
    with torch.no_grad():
        p = mdl(X[-1:].float()).numpy().flatten()
    inv    = scaler.inverse_transform(p.reshape(-1, 1)).flatten()
    future = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    return pd.Series(inv, index=future), None, None

def monte_carlo_gbm(df, n_days=30, n_sims=500, seed=42):
    np.random.seed(seed)
    rets  = df["Returns"].dropna().values[-252:]
    mu    = rets.mean()
    sigma = rets.std()
    S0    = df["Close"].iloc[-1]
    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = S0
    for t in range(1, n_days + 1):
        Z = np.random.standard_normal(n_sims)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * Z)
    future = pd.date_range(df.index[-1], periods=n_days + 1)
    return future, paths

def backtest_rsi(df):
    sig  = (df["RSI"] < 30).astype(int)
    rets = df["Close"].pct_change().shift(-1)
    return (1 + (sig * rets).fillna(0)).cumprod()

def pivot_points(df):
    """Classic floor-trader pivot points from last full session."""
    h, l, c = df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    pp = (h + l + c) / 3
    return dict(
        PP=pp,
        R1=2*pp - l, R2=pp + (h - l), R3=h + 2*(pp - l),
        S1=2*pp - h, S2=pp - (h - l), S3=l - 2*(h - pp),
    )

def detect_anomalies(df, window=20, z_thresh=2.5):
    """Flag days where price return is > z_thresh standard deviations."""
    r = df["Close"].pct_change()
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std()
    z  = (r - mu) / (sd + 1e-9)
    anom = df[z.abs() > z_thresh].copy()
    anom["z_score"] = z[z.abs() > z_thresh]
    return anom

def signal_scorecard(df):
    """Return a list of (indicator, signal, value, direction) tuples."""
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    signals = []

    # RSI
    rsi = latest["RSI"]
    if rsi > 70:
        signals.append(("RSI (14)", "OVERBOUGHT", f"{rsi:.1f}", "bear"))
    elif rsi < 30:
        signals.append(("RSI (14)", "OVERSOLD", f"{rsi:.1f}", "bull"))
    else:
        signals.append(("RSI (14)", "NEUTRAL", f"{rsi:.1f}", ""))

    # MACD crossover
    if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        signals.append(("MACD", "BULLISH CROSS", f"{latest['MACD']:.3f}", "bull"))
    elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        signals.append(("MACD", "BEARISH CROSS", f"{latest['MACD']:.3f}", "bear"))
    elif latest["MACD"] > latest["MACD_Signal"]:
        signals.append(("MACD", "ABOVE SIGNAL", f"{latest['MACD']:.3f}", "bull"))
    else:
        signals.append(("MACD", "BELOW SIGNAL", f"{latest['MACD']:.3f}", "bear"))

    # Bollinger Bands
    close = latest["Close"]
    if close > latest["BB_upper"]:
        signals.append(("Bollinger Bands", "ABOVE UPPER", f"${close:.2f}", "bear"))
    elif close < latest["BB_lower"]:
        signals.append(("Bollinger Bands", "BELOW LOWER", f"${close:.2f}", "bull"))
    else:
        pct = (close - latest["BB_lower"]) / (latest["BB_upper"] - latest["BB_lower"] + 1e-9)
        signals.append(("Bollinger Bands", f"MID RANGE ({pct*100:.0f}%)", f"${close:.2f}", ""))

    # MA crossover (7 vs 30)
    if latest["MA7"] > latest["MA30"] and prev["MA7"] <= prev["MA30"]:
        signals.append(("MA 7/30 Cross", "GOLDEN CROSS ▲", f"${latest['MA7']:.2f}", "bull"))
    elif latest["MA7"] < latest["MA30"] and prev["MA7"] >= prev["MA30"]:
        signals.append(("MA 7/30 Cross", "DEATH CROSS ▼", f"${latest['MA7']:.2f}", "bear"))
    elif latest["MA7"] > latest["MA30"]:
        signals.append(("MA 7/30 Cross", "MA7 > MA30", f"${latest['MA7']:.2f}", "bull"))
    else:
        signals.append(("MA 7/30 Cross", "MA7 < MA30", f"${latest['MA7']:.2f}", "bear"))

    # Price vs MA90
    if close > latest["MA90"]:
        signals.append(("Price vs MA90", "ABOVE TREND", f"${close:.2f}", "bull"))
    else:
        signals.append(("Price vs MA90", "BELOW TREND", f"${close:.2f}", "bear"))

    # Volume
    vol_ratio = latest["Vol_ratio"]
    if vol_ratio > 1.5:
        signals.append(("Volume", "HIGH VOLUME", f"{vol_ratio:.2f}x avg", "warn"))
    elif vol_ratio < 0.6:
        signals.append(("Volume", "LOW VOLUME", f"{vol_ratio:.2f}x avg", ""))
    else:
        signals.append(("Volume", "NORMAL", f"{vol_ratio:.2f}x avg", ""))

    # Volatility regime
    vol = latest["Volatility"]
    if vol > 40:
        signals.append(("Volatility (30D Ann.)", "ELEVATED", f"{vol:.1f}%", "warn"))
    elif vol < 20:
        signals.append(("Volatility (30D Ann.)", "COMPRESSED", f"{vol:.1f}%", "bull"))
    else:
        signals.append(("Volatility (30D Ann.)", "NORMAL", f"{vol:.1f}%", ""))

    return signals

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ── Black-Scholes Options Pricing & Greeks ────────────────────────────────────
def black_scholes(S, K, T, r, sigma, opt_type="call"):
    """Return (price, delta, gamma, theta, vega, rho) for European option."""
    if T <= 0:
        return (max(S - K, 0) if opt_type == "call" else max(K - S, 0),
                1.0 if opt_type == "call" and S > K else 0.0,
                0, 0, 0, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        price = S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2)
        delta = sp_norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * sp_norm.cdf(d2) / 100
        theta = (-S * sp_norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * sp_norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * sp_norm.cdf(-d2) - S * sp_norm.cdf(-d1)
        delta = sp_norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * sp_norm.cdf(-d2) / 100
        theta = (-S * sp_norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * sp_norm.cdf(-d2)) / 365
    gamma = sp_norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * sp_norm.pdf(d1) * np.sqrt(T) / 100
    return price, delta, gamma, theta, vega, rho

# ── XGBoost with RandomizedSearchCV ──────────────────────────────────────────
@st.cache_data(ttl=7200)
def train_xgboost(_df):
    """Train XGBoost regressor with RandomizedSearchCV on technical features."""
    feat = _df[["RSI", "MACD", "MACD_Signal", "BB_width", "MA_ratio",
                "Vol_ratio", "Volatility", "Returns"]].copy()
    feat["Price_vs_MA30"]  = (_df["Close"] - _df["MA30"]) / _df["MA30"]
    feat["Price_vs_BB"]    = (_df["Close"] - _df["BB_mid"]) / (_df["BB_upper"] - _df["BB_lower"] + 1e-9)
    feat["High_Low_ratio"] = (_df["High"] - _df["Low"]) / _df["Close"]
    feat["Target"]         = _df["Close"].shift(-1)
    feat = feat.dropna()
    split = int(len(feat) * 0.8)
    X_tr, y_tr = feat.drop("Target", axis=1).iloc[:split], feat["Target"].iloc[:split]
    X_te, y_te = feat.drop("Target", axis=1).iloc[split:], feat["Target"].iloc[split:]

    param_dist = {
        "n_estimators":     [100, 200, 300, 400],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.1, 0.15],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha":        [0, 0.1, 0.5],
        "reg_lambda":       [0.5, 1.0, 1.5],
    }
    xgb = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=30, cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, refit=True
    )
    search.fit(X_tr, y_tr)
    best   = search.best_estimator_
    preds  = best.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te, preds))
    mape_v = mape(y_te.values, preds)
    r2     = r2_score(y_te, preds)
    fi     = pd.Series(best.feature_importances_, index=X_tr.columns).sort_values()
    return best, search.best_params_, rmse, mape_v, r2, fi, X_te.index, y_te.values, preds

# ── Walk-Forward Validation ───────────────────────────────────────────────────
@st.cache_data(ttl=7200)
def walk_forward_validation(_df, window=252, step=21):
    """
    Rolling walk-forward validation.
    Train on `window` days, predict next `step` days, advance by `step`.
    Returns actual prices, LR predictions, XGB predictions, and dates.
    """
    closes = _df["Close"].values
    dates  = _df.index
    feat_cols = ["RSI", "MACD", "MACD_Signal", "BB_width", "MA_ratio",
                 "Vol_ratio", "Volatility", "Returns", "High_Low_ratio"]
    _df2 = _df.copy()
    _df2["High_Low_ratio"] = (_df2["High"] - _df2["Low"]) / _df2["Close"]

    all_dates, all_actual, all_lr, all_xgb = [], [], [], []
    idx = window
    while idx + step <= len(_df):
        train_close = closes[idx - window : idx]
        test_close  = closes[idx : idx + step]
        train_dates = dates[idx - window : idx]
        test_dates  = dates[idx : idx + step]

        # LR on ordinal dates
        ord_tr = np.array([d.toordinal() for d in train_dates]).reshape(-1, 1)
        ord_te = np.array([d.toordinal() for d in test_dates]).reshape(-1, 1)
        lr = LinearRegression().fit(ord_tr, train_close)
        lr_pred = lr.predict(ord_te)

        # XGB on features
        try:
            feat_tr = _df2[feat_cols].iloc[idx - window : idx].copy()
            feat_tr["Target"] = closes[idx - window + 1 : idx + 1]
            feat_tr = feat_tr.dropna()
            if len(feat_tr) > 10:
                xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                   random_state=42, verbosity=0)
                xgb.fit(feat_tr.drop("Target", axis=1), feat_tr["Target"])
                feat_te = _df2[feat_cols].iloc[idx : idx + step].fillna(method="ffill").values
                xgb_pred = xgb.predict(feat_te)
            else:
                xgb_pred = lr_pred
        except Exception:
            xgb_pred = lr_pred

        all_dates.extend(test_dates)
        all_actual.extend(test_close)
        all_lr.extend(lr_pred)
        all_xgb.extend(xgb_pred)
        idx += step

    return (np.array(all_dates), np.array(all_actual),
            np.array(all_lr), np.array(all_xgb))

# ── Cointegration & Pair Trading ──────────────────────────────────────────────
def run_cointegration(peers_df, base="AMZN"):
    """Test AMZN vs each peer for cointegration using Engle-Granger test."""
    if base not in peers_df.columns:
        return []
    base_s  = peers_df[base].dropna()
    results = []
    for ticker in [c for c in peers_df.columns if c != base]:
        peer_s = peers_df[ticker].dropna()
        common = base_s.index.intersection(peer_s.index)
        if len(common) < 60:
            continue
        score, pval, _ = coint(base_s[common], peer_s[common])
        # Calculate spread and half-life
        ols = sm.OLS(base_s[common].values, sm.add_constant(peer_s[common].values)).fit()
        hedge = ols.params[1]
        spread = base_s[common] - hedge * peer_s[common]
        # Spread ADF test
        adf_stat, adf_p, *_ = adfuller(spread.dropna())
        # Half-life via AR(1)
        lag_spread = spread.shift(1).dropna()
        delta      = spread.diff().dropna()
        common_idx = lag_spread.index.intersection(delta.index)
        ar_ols     = sm.OLS(delta[common_idx].values, sm.add_constant(lag_spread[common_idx].values)).fit()
        half_life  = max(1, int(-np.log(2) / ar_ols.params[1])) if ar_ols.params[1] < 0 else None
        results.append({
            "Pair":       f"AMZN/{ticker}",
            "Hedge Ratio": round(hedge, 4),
            "Coint p-val": round(pval, 4),
            "ADF p-val":   round(adf_p, 4),
            "Half-Life":   f"{half_life}d" if half_life else "N/A",
            "Cointegrated": "✅ YES" if pval < 0.05 else "❌ NO",
            "_spread":     spread,
            "_ticker":     ticker,
        })
    return results

# ── FinBERT Sentiment ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model…")
def load_finbert():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=-1,            # CPU
        truncation=True,
        max_length=512,
    )

def finbert_sentiment(pipe, texts):
    """Return list of (label, score) for each text."""
    results = []
    for t in texts:
        try:
            out = pipe(t[:512])[0]
            label = out["label"].lower()   # positive / negative / neutral
            score = out["score"]
            # map to polarity: positive→+score, negative→-score, neutral→0
            polarity = score if label == "positive" else (-score if label == "negative" else 0.0)
            results.append((label, polarity))
        except Exception:
            results.append(("neutral", 0.0))
    return results

# ── Risk metrics ──────────────────────────────────────────────────────────────
def compute_risk_metrics(returns, rf_annual=0.05):
    r  = returns.dropna()
    rf = rf_annual / 252
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = (r.mean() - rf) / r.std() * np.sqrt(252)
    down    = r[r < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf_annual) / down if down > 0 else np.nan
    cum     = (1 + r).cumprod()
    peak    = cum.cummax()
    dd      = (cum - peak) / peak
    max_dd  = dd.min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    var95   = np.percentile(r, 5)
    var99   = np.percentile(r, 1)
    cvar95  = r[r <= var95].mean()
    _, p_normal = stats.shapiro(r.tail(100).values)
    skew    = float(stats.skew(r))
    kurt    = float(stats.kurtosis(r))
    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
        max_dd=max_dd, calmar=calmar, var95=var95, var99=var99,
        cvar95=cvar95, p_normal=p_normal, skew=skew, kurt=kurt,
        cum=cum, dd=dd,
    )

# ── Feature importance ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def compute_feature_importance(_df):
    feat = _df[["RSI", "MACD", "MACD_Signal", "BB_width", "MA_ratio",
                 "Vol_ratio", "Volatility", "Returns"]].copy()
    feat["Price_vs_MA30"] = (_df["Close"] - _df["MA30"]) / _df["MA30"]
    feat["Price_vs_BB"]   = (_df["Close"] - _df["BB_mid"]) / (_df["BB_upper"] - _df["BB_lower"] + 1e-9)
    feat["Target"]        = _df["Close"].shift(-1)
    feat = feat.dropna()
    X = feat.drop("Target", axis=1)
    y = feat["Target"]
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values()

# ── Header ────────────────────────────────────────────────────────────────────
cur     = df["Close"].iloc[-1]
prev    = df["Close"].iloc[-2]
day_chg = (cur - prev) / prev * 100
high52  = df["Close"].rolling(252).max().iloc[-1]
low52   = df["Close"].rolling(252).min().iloc[-1]
ytd_s   = df.loc[df.index >= f"{df.index[-1].year}-01-01", "Close"]
ytd_ret = (cur / ytd_s.iloc[0] - 1) * 100 if len(ytd_s) > 0 else 0.0
vol30   = df["Volatility"].iloc[-1]
chg_cls = "t-change-up" if day_chg >= 0 else "t-change-down"
chg_sym = "▲" if day_chg >= 0 else "▼"

st.markdown(f"""
<div class="t-header">
  <div>
    <div class="t-ticker-block">
      <span class="t-symbol">AMZN</span>
      <span class="t-name">Amazon.com, Inc.</span>
      <span class="t-exch">NASDAQ · USD</span>
    </div>
    <div class="t-meta">
      <span>ML FORECASTING</span>
      <span>RISK ANALYTICS</span>
      <span>MONTE CARLO</span>
      <span>PEER COMPARISON</span>
      <span style="color:#1e2e4e">·  EDUCATIONAL — NOT FINANCIAL ADVICE</span>
    </div>
  </div>
  <div class="t-right">
    <div class="live-pill"><div class="live-dot"></div>LIVE DATA</div>
    <div class="t-price-main">${cur:.2f}</div>
    <div class="{chg_cls}">{chg_sym} {abs(day_chg):.2f}% today</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
kpis = [
    ("52-Week High",      f"${high52:.2f}",   f"{(cur/high52-1)*100:.1f}% from high",  "pos" if cur >= high52*0.97 else "neg"),
    ("52-Week Low",       f"${low52:.2f}",    f"{(cur/low52-1)*100:.1f}% above low",   "pos"),
    ("YTD Return",        f"{ytd_ret:+.1f}%", "year-to-date",                          "pos" if ytd_ret >= 0 else "neg"),
    ("30D Volatility",    f"{vol30:.1f}%",    "annualised",                            "neu" if vol30 < 30 else "neg"),
    ("RSI (14)",          f"{df['RSI'].iloc[-1]:.1f}",
                          "overbought" if df['RSI'].iloc[-1] > 70 else "oversold" if df['RSI'].iloc[-1] < 30 else "neutral",
                          "neg" if df['RSI'].iloc[-1] > 70 else "bull" if df['RSI'].iloc[-1] < 30 else "neu"),
]
card_html = '<div class="kpi-grid">'
for label, value, delta, variant in kpis:
    delta_cls = "up" if variant == "pos" else "down" if variant == "neg" else "neu"
    card_html += f"""
    <div class="kpi-card {variant}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta {delta_cls}">{delta}</div>
    </div>"""
card_html += "</div>"
st.markdown(card_html, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fc, tab_tech, tab_risk, tab_mkt, tab_wf, tab_sig, tab_news, tab_about = st.tabs([
    "🔮 Forecast", "📉 Technical Analysis", "📊 Risk Analytics",
    "🌐 Market Context", "🔁 Walk-Forward Validation", "📡 Signals & Anomalies",
    "📰 News & Sentiment", "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fc:
    ctrl, chart_col = st.columns([1, 3])
    with ctrl:
        st.markdown(_sig("Seq2Seq Encoder-Decoder LSTM — direct multi-step output, no error accumulation.", "bull"), unsafe_allow_html=True)
        n_days       = st.slider("Horizon (days)", 1, 7, 7)
        if n_days != 7:
            st.info("Seq2Seq model is trained for a 7-day horizon — setting to 7.")
            n_days = 7
        show_ci      = st.checkbox("Show 95% confidence band", value=True)
        show_mc      = st.checkbox("Show Monte Carlo simulation", value=False)
        context_days = st.slider("Historical context (days)", 30, 120, 60)

    with st.spinner("Running Seq2Seq forecast…"):
        mu, lo, hi = forecast_seq2seq(df, SEQ_LEN, n_days)

    current_price = df["Close"].iloc[-1]
    end_price     = mu.iloc[-1]
    pct_chg       = (end_price - current_price) / current_price * 100
    trend_up      = end_price > current_price

    with chart_col:
        fig = go.Figure()
        hist = df["Close"].iloc[-context_days:]
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Historical",
                                 line=dict(color="#5c6bc0", width=2)))

        if show_mc:
            mc_dates, mc_paths = monte_carlo_gbm(df, n_days=n_days, n_sims=500)
            p10 = np.percentile(mc_paths, 10, axis=1)
            p25 = np.percentile(mc_paths, 25, axis=1)
            p75 = np.percentile(mc_paths, 75, axis=1)
            p90 = np.percentile(mc_paths, 90, axis=1)
            for path in mc_paths[:, ::50].T:
                fig.add_trace(go.Scatter(x=mc_dates, y=path, mode="lines",
                                         line=dict(color="rgba(255,183,77,0.08)", width=1),
                                         showlegend=False))
            fig.add_trace(go.Scatter(
                x=list(mc_dates) + list(mc_dates[::-1]),
                y=list(p90) + list(p10[::-1]),
                fill="toself", fillcolor="rgba(255,183,77,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="MC P10–P90"
            ))
            fig.add_trace(go.Scatter(x=mc_dates, y=np.median(mc_paths, axis=1),
                                     name="MC Median", line=dict(color="#ffb300", width=2, dash="dot")))

        fig.add_trace(go.Scatter(x=mu.index, y=mu, name="Seq2Seq Forecast",
                                 line=dict(color="#26a69a", width=2.5, dash="dash")))
        if show_ci and lo is not None:
            fig.add_trace(go.Scatter(
                x=list(mu.index) + list(mu.index[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself", fillcolor="rgba(38,166,154,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI"
            ))
        _c(fig, legend_h=True)
        fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, width='stretch')

    direction = "upward" if trend_up else "downward"
    variant   = "bull" if trend_up else "bear"
    st.markdown(_sig(
        f"<strong>Seq2Seq LSTM</strong> projects a <strong>{direction} trend</strong> "
        f"over {n_days} day(s): <strong>${current_price:.2f} → ${end_price:.2f} ({pct_chg:+.2f}%)</strong>",
        variant
    ), unsafe_allow_html=True)

    if show_mc:
        mc_final = mc_paths[-1]
        prob_up   = (mc_final > current_price).mean() * 100
        mc_med    = np.median(mc_final)
        mc_p5     = np.percentile(mc_final, 5)
        mc_p95    = np.percentile(mc_final, 95)
        ca, cb, cc, cd = st.columns(4)
        ca.metric("MC Median (end)",    f"${mc_med:.2f}")
        cb.metric("P5 (downside)",      f"${mc_p5:.2f}")
        cc.metric("P95 (upside)",       f"${mc_p95:.2f}")
        cd.metric("P(price rises)",     f"{prob_up:.1f}%")

    st.markdown(_sec("Forecast Table", "teal"), unsafe_allow_html=True)
    fdf = mu.reset_index()
    fdf.columns = ["Date", "Forecast ($)"]
    fdf["Date"] = pd.to_datetime(fdf["Date"]).dt.strftime("%Y-%m-%d")
    if lo is not None:
        fdf["Lower 95%"] = [f"${v:.2f}" for v in lo.values]
        fdf["Upper 95%"] = [f"${v:.2f}" for v in hi.values]
    fdf["Forecast ($)"] = fdf["Forecast ($)"].map("${:.2f}".format)
    st.dataframe(fdf, width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    period_opt = st.selectbox("Time range", ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"], index=2, key="tech_period")
    period_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "2 Years": 504}
    dv = df.iloc[-period_map[period_opt]:]

    st.markdown(_sec("Price · Candlestick · VWAP"), unsafe_allow_html=True)
    # VWAP
    dv = dv.copy()
    dv["VWAP"] = (dv["Close"] * dv["Volume"]).cumsum() / dv["Volume"].cumsum()
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=dv.index, open=dv["Open"], high=dv["High"], low=dv["Low"], close=dv["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#10b981"), fillcolor="rgba(16,185,129,0.3)"),
        decreasing=dict(line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.3)"),
    ))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["VWAP"], name="VWAP",
                                    line=dict(color="#f59e0b", width=1.5, dash="dot")))
    _c(fig_price, height=420)
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Price (USD)",
                             xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, width='stretch')

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.markdown(_sec("RSI · 14 Period", "teal"), unsafe_allow_html=True)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI", line=dict(color="#26a69a", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.07)", line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.07)", line_width=0)
        _c(fig_rsi)
        fig_rsi.update_layout(yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, width='stretch')
        rsi_val = dv["RSI"].iloc[-1]
        if rsi_val > 70:
            st.markdown(_sig(f"RSI = {rsi_val:.1f} — potentially <strong>overbought</strong>. Watch for reversal.", "bear"), unsafe_allow_html=True)
        elif rsi_val < 30:
            st.markdown(_sig(f"RSI = {rsi_val:.1f} — potentially <strong>oversold</strong>. Potential entry signal.", "bull"), unsafe_allow_html=True)
        else:
            st.markdown(_sig(f"RSI = {rsi_val:.1f} — neutral zone, no extreme signal."), unsafe_allow_html=True)

    with col_macd:
        st.markdown(_sec("MACD · Signal · Histogram", "amber"), unsafe_allow_html=True)
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=dv.index, y=dv["MACD"],        name="MACD",   line=dict(color="#5c6bc0", width=2)))
        fig_macd.add_trace(go.Scatter(x=dv.index, y=dv["MACD_Signal"], name="Signal", line=dict(color="#ffb300", width=1.5)))
        fig_macd.add_trace(go.Bar(x=dv.index, y=dv["MACD_Hist"], name="Histogram",
                                  marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in dv["MACD_Hist"]], opacity=0.7))
        _c(fig_macd)
        st.plotly_chart(fig_macd, width='stretch')
        if dv["MACD"].iloc[-1] > dv["MACD_Signal"].iloc[-1]:
            st.markdown(_sig("MACD above Signal line — <strong>bullish momentum</strong>.", "bull"), unsafe_allow_html=True)
        else:
            st.markdown(_sig("MACD below Signal line — <strong>bearish momentum</strong>.", "bear"), unsafe_allow_html=True)

    st.markdown(_sec("Volume Analysis"), unsafe_allow_html=True)
    avg_vol = dv["Volume"].mean()
    fig_vol = go.Figure(go.Bar(x=dv.index, y=dv["Volume"],
                                marker_color=["#26a69a" if v >= avg_vol else "#ef5350" for v in dv["Volume"]]))
    fig_vol.add_hline(y=avg_vol, line_dash="dash", line_color="#ffb300",
                      annotation_text=f"20-day avg {avg_vol/1e6:.0f}M")
    _c(fig_vol)
    fig_vol.update_layout(yaxis_title="Shares")
    st.plotly_chart(fig_vol, width='stretch')

    st.markdown(_sec("RSI Strategy vs Buy &amp; Hold · Backtest", "teal"), unsafe_allow_html=True)
    bt       = backtest_rsi(dv)
    buy_hold = (1 + dv["Returns"].fillna(0)).cumprod()
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt,        name="RSI Strategy", line=dict(color="#26a69a", width=2)))
    fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name="Buy & Hold", line=dict(color="#5c6bc0", width=2, dash="dot")))
    _c(fig_bt, legend_h=True)
    fig_bt.update_layout(yaxis_title="Cumulative Return")
    st.plotly_chart(fig_bt, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    rm = compute_risk_metrics(df["Returns"])

    st.markdown(_sec("Risk & Return Metrics · 2Y Window · 5% Risk-Free Rate", "red"), unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Annualised Return",   f"{rm['ann_ret']*100:+.1f}%")
    r2.metric("Annualised Volatility", f"{rm['ann_vol']*100:.1f}%")
    r3.metric("Sharpe Ratio",         f"{rm['sharpe']:.3f}")
    r4.metric("Sortino Ratio",        f"{rm['sortino']:.3f}")

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Max Drawdown",         f"{rm['max_dd']*100:.1f}%")
    r6.metric("Calmar Ratio",         f"{rm['calmar']:.3f}")
    r7.metric("VaR 95% (daily)",      f"{rm['var95']*100:.2f}%")
    r8.metric("CVaR 95% (daily)",     f"{rm['cvar95']*100:.2f}%")

    sharpe_note = "above 1.0 — good risk-adjusted return" if rm["sharpe"] > 1 else "below 1.0 — return does not fully compensate risk"
    sv = "bull" if rm["sharpe"] > 1 else "bear"
    st.markdown(_sig(
        f"Sharpe ratio <strong>{rm['sharpe']:.3f}</strong> is {sharpe_note}. "
        f"CVaR 95% = <strong>{rm['cvar95']*100:.2f}%</strong> — expected loss on the worst 5% of trading days.", sv
    ), unsafe_allow_html=True)

    st.markdown(_sec("Underwater Chart · Drawdown from Peak", "red"), unsafe_allow_html=True)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=rm["dd"].index, y=rm["dd"] * 100,
                                 fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                                 line=dict(color="#ef5350", width=1.5), name="Drawdown %"))
    _c(fig_dd)
    fig_dd.update_layout(yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, width='stretch')

    st.markdown(_sec("Rolling 90-Day Sharpe Ratio"), unsafe_allow_html=True)
    roll_ret = df["Returns"].rolling(90).mean()
    roll_std = df["Returns"].rolling(90).std()
    roll_sharpe = (roll_ret - 0.05/252) / roll_std * np.sqrt(252)
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe,
                                 line=dict(color="#5c6bc0", width=2), name="Rolling Sharpe"))
    fig_rs.add_hline(y=1, line_dash="dash", line_color="#26a69a", annotation_text="Sharpe = 1")
    fig_rs.add_hline(y=0, line_dash="dash", line_color="#ef5350")
    _c(fig_rs)
    fig_rs.update_layout(yaxis_title="Sharpe Ratio")
    st.plotly_chart(fig_rs, width='stretch')

    st.markdown(_sec("Return Distribution · Normality Analysis", "amber"), unsafe_allow_html=True)
    col_dist, col_stats = st.columns([2, 1])
    rets_clean = df["Returns"].dropna().values
    with col_dist:
        x_range = np.linspace(rets_clean.min(), rets_clean.max(), 200)
        normal_fit = stats.norm.pdf(x_range, rets_clean.mean(), rets_clean.std())
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=rets_clean, nbinsx=80, histnorm="probability density",
                                         name="Daily Returns", marker_color="#5c6bc0", opacity=0.7))
        fig_dist.add_trace(go.Scatter(x=x_range, y=normal_fit, name="Normal Fit",
                                       line=dict(color="#ffb300", width=2)))
        fig_dist.add_vline(x=rm["var95"], line_dash="dash", line_color="#ef5350",
                           annotation_text="VaR 95%")
        _c(fig_dist)
        fig_dist.update_layout(xaxis_title="Daily Return", yaxis_title="Density")
        st.plotly_chart(fig_dist, width='stretch')
    with col_stats:
        norm_flag = rm["p_normal"] < 0.05
        kurt_flag = rm["kurt"] > 0
        st.markdown(f"""
        <div class="info-block">
          <span class="ib-title">Distribution Statistics</span>
          Skewness &nbsp;&nbsp;<strong style="color:#d8e3f5">{rm['skew']:.4f}</strong><br>
          Kurtosis &nbsp;&nbsp;<strong style="color:#d8e3f5">{rm['kurt']:.4f}</strong>
          <span style="color:#3d5070"> (Normal = 0)</span><br>
          Shapiro-Wilk p &nbsp;&nbsp;<strong style="color:#d8e3f5">{rm['p_normal']:.4f}</strong><br><br>
          {'<span style="color:#ef4444">✗ Non-normal distribution detected — fat tails present</span>' if norm_flag else '<span style="color:#10b981">✓ Cannot reject normality (p > 0.05)</span>'}<br>
          {'<span style="color:#f59e0b">⚠ Positive excess kurtosis — extreme moves more likely than Gaussian</span>' if kurt_flag else ''}
        </div>""", unsafe_allow_html=True)

    st.markdown(_sec("Rolling 30-Day VaR 95% · Value at Risk Over Time", "red"), unsafe_allow_html=True)
    st.caption("Daily VaR 95% computed on rolling 30-day window — shows how tail risk evolves.")
    roll_var = df["Returns"].rolling(30).quantile(0.05) * 100
    fig_rvar = go.Figure()
    fig_rvar.add_trace(go.Scatter(
        x=roll_var.index, y=roll_var,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
        line=dict(color="#ef4444", width=1.5), name="VaR 95%"
    ))
    fig_rvar.add_hline(y=roll_var.mean(), line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Mean VaR {roll_var.mean():.2f}%")
    _c(fig_rvar)
    fig_rvar.update_layout(yaxis_title="VaR 95% (%)")
    st.plotly_chart(fig_rvar, width='stretch')

    st.markdown(_sec("Volatility Regime · 30D vs 90D Rolling Annualised Vol", "amber"), unsafe_allow_html=True)
    st.caption("Regime detection via rolling volatility comparison — high-vol vs low-vol periods.")
    vol30 = df["Returns"].rolling(30).std() * np.sqrt(252) * 100
    vol90 = df["Returns"].rolling(90).std() * np.sqrt(252) * 100
    vol_threshold = vol90.mean()
    fig_regime = go.Figure()
    # colour background by regime
    high_vol = vol30 > vol_threshold
    for i in range(1, len(vol30)):
        if high_vol.iloc[i]:
            fig_regime.add_vrect(
                x0=vol30.index[i-1], x1=vol30.index[i],
                fillcolor="rgba(239,68,68,0.07)", line_width=0
            )
    fig_regime.add_trace(go.Scatter(x=vol30.index, y=vol30, name="Vol 30D",
                                     line=dict(color="#ef4444", width=2)))
    fig_regime.add_trace(go.Scatter(x=vol90.index, y=vol90, name="Vol 90D",
                                     line=dict(color="#f59e0b", width=1.5, dash="dot")))
    fig_regime.add_hline(y=vol_threshold, line_dash="dash", line_color="#7d93b8",
                          annotation_text="Regime Threshold")
    _c(fig_regime, legend_h=True)
    fig_regime.update_layout(yaxis_title="Annualised Volatility (%)")
    st.plotly_chart(fig_regime, width='stretch')
    curr_regime = "HIGH VOLATILITY" if vol30.iloc[-1] > vol_threshold else "LOW VOLATILITY"
    rv = "warn" if curr_regime == "HIGH VOLATILITY" else "bull"
    st.markdown(_sig(f"Current regime: <strong>{curr_regime}</strong> — 30D vol ({vol30.iloc[-1]:.1f}%) vs 90D baseline ({vol_threshold:.1f}%). Red shading = high-vol periods.", rv), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mkt:
    # ── AMZN-only sections (instant, no network) ──────────────────────────────
    def _pr(s, days):
        s = s.dropna()
        if len(s) < days: return None
        try:
            v = (s.iloc[-1] / s.iloc[-days] - 1) * 100
            return v if np.isfinite(v) else None
        except Exception:
            return None

    amzn_s = df["Close"]
    st.markdown(_sec("AMZN Return Metrics · Multi-Period"), unsafe_allow_html=True)
    am1, am2, am3, am4, am5 = st.columns(5)
    def _fmt(v): return f"{v:+.2f}%" if v is not None else "N/A"
    am1.metric("1W Return",  _fmt(_pr(amzn_s, 5)))
    am2.metric("1M Return",  _fmt(_pr(amzn_s, 21)))
    am3.metric("3M Return",  _fmt(_pr(amzn_s, 63)))
    am4.metric("6M Return",  _fmt(_pr(amzn_s, 126)))
    am5.metric("1Y Return",  _fmt(_pr(amzn_s, 252)))

    st.markdown(_sec("AMZN Price · Full History"), unsafe_allow_html=True)
    fig_amzn = go.Figure()
    fig_amzn.add_trace(go.Scatter(x=amzn_s.index, y=amzn_s,
                                   line=dict(color="#5c6bc0", width=1.5), name="AMZN Close"))
    _c(fig_amzn)
    fig_amzn.update_layout(yaxis_title="Price (USD)")
    st.plotly_chart(fig_amzn, width='stretch')

    st.markdown(_sec("Monthly Returns Heatmap", "amber"), unsafe_allow_html=True)
    try:
        monthly = df["Close"].resample("ME").last().pct_change().dropna() * 100
        monthly_raw = pd.DataFrame({
            "Year":  monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        })
        _month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                      7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        pivot = monthly_raw.pivot_table(index="Year", columns="Month", values="Return", aggfunc="mean")
        pivot.columns = [_month_map.get(m, str(m)) for m in pivot.columns]
        fig_heat = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn",
                              zmin=-20, zmax=20, aspect="auto")
        _c(fig_heat)
        st.plotly_chart(fig_heat, width='stretch')
    except Exception as _he:
        st.warning(f"Monthly heatmap unavailable: {_he}")

    # ── Peer Comparison (lazy — user clicks to load) ──────────────────────────
    st.markdown(_sec("Peer Comparison · MSFT GOOGL META AAPL SPY", "teal"), unsafe_allow_html=True)
    st.markdown(_sig("Peer data requires live yfinance fetch. Click below to load — may take 10–20 seconds.", "warn"), unsafe_allow_html=True)

    if st.button("▶ Load Peer Comparison", key="load_peers_btn"):
        with st.spinner("Fetching peer data from yfinance…"):
            peers_raw = load_peers()

        if peers_raw.empty:
            st.markdown(_sig("yfinance rate-limited — try again in a few minutes.", "bear"), unsafe_allow_html=True)
        else:
            # Align AMZN from main df
            common_idx = peers_raw.index.intersection(df.index)
            if len(common_idx) > 10:
                peers_df = peers_raw.loc[common_idx].copy()
                peers_df["AMZN"] = df["Close"].loc[common_idx]
            else:
                peers_df = peers_raw.copy()
            peers_df = peers_df.ffill().dropna(how="any")

            if not peers_df.empty:
                norm_peers = peers_df.div(peers_df.iloc[0]) * 100
                colors_p = ["#5c6bc0", "#26a69a", "#ffb300", "#ef5350", "#ab47bc", "#78909c"]
                fig_peers = go.Figure()
                for i, col in enumerate(norm_peers.columns):
                    w = 2.5 if col == "AMZN" else 1.5
                    fig_peers.add_trace(go.Scatter(x=norm_peers.index, y=norm_peers[col],
                                                    name=col, line=dict(color=colors_p[i % len(colors_p)], width=w)))
                fig_peers.add_hline(y=100, line_dash="dash", line_color="#7b8ab8")
                _c(fig_peers, legend_h=True)
                fig_peers.update_layout(yaxis_title="Indexed Price (Base = 100)")
                st.plotly_chart(fig_peers, width='stretch')

                # Returns table
                ret_rows = []
                for t in peers_df.columns:
                    s = peers_df[t]
                    ret_rows.append({"Ticker": t,
                        "1M (%)": period_return(s,21), "3M (%)": period_return(s,63),
                        "6M (%)": period_return(s,126), "1Y (%)": period_return(s,252),
                        "Vol (%)": s.pct_change().dropna().std()*np.sqrt(252)*100})
                ret_df = pd.DataFrame(ret_rows).set_index("Ticker")
                def color_returns(v):
                    if pd.isna(v): return ""
                    return "color:#10b981" if v >= 0 else "color:#ef4444"
                st.dataframe(ret_df.style.format("{:.1f}").map(color_returns,
                    subset=["1M (%)","3M (%)","6M (%)","1Y (%)"]), width='stretch')

                # Correlation
                if len(peers_df.columns) > 1:
                    st.markdown(_sec("Correlation Matrix", "blue"), unsafe_allow_html=True)
                    fig_corr = px.imshow(peers_df.pct_change().dropna().corr(),
                                          text_auto=".2f", color_continuous_scale="RdBu_r",
                                          zmin=-1, zmax=1, aspect="auto")
                    _c(fig_corr)
                    st.plotly_chart(fig_corr, width='stretch')

                # Beta
                if "SPY" in peers_df.columns:
                    st.markdown(_sec("Beta vs SPY", "red"), unsafe_allow_html=True)
                    spy_r = peers_df["SPY"].pct_change().dropna()
                    beta_rows = []
                    for t in [c for c in peers_df.columns if c != "SPY"]:
                        sr = peers_df[t].pct_change().dropna()
                        ci = sr.index.intersection(spy_r.index)
                        cov = np.cov(sr[ci], spy_r[ci])[0,1]
                        var = spy_r[ci].var()
                        beta_rows.append({"Ticker": t, "Beta": cov/var if var>0 else np.nan})
                    bdf = pd.DataFrame(beta_rows)
                    fig_b = go.Figure(go.Bar(x=bdf["Ticker"], y=bdf["Beta"],
                                              marker_color=["#10b981" if b<1 else "#ef4444" for b in bdf["Beta"]]))
                    fig_b.add_hline(y=1, line_dash="dash", line_color="#f59e0b")
                    _c(fig_b)
                    st.plotly_chart(fig_b, width='stretch')

                # Cointegration
                st.markdown(_sec("Cointegration & Pair Trading · Engle-Granger", "purple"), unsafe_allow_html=True)
                try:
                    coint_results = run_cointegration(peers_df)
                    if coint_results:
                        coint_display = [{k:v for k,v in r.items() if not k.startswith("_")} for r in coint_results]
                        st.dataframe(pd.DataFrame(coint_display), width='stretch', hide_index=True)
                        best = min(coint_results, key=lambda x: x["Coint p-val"])
                        spread = best["_spread"]
                        z_spread = (spread - spread.mean()) / spread.std()
                        fig_sp = go.Figure()
                        fig_sp.add_trace(go.Scatter(x=z_spread.index, y=z_spread,
                                                     line=dict(color="#5c6bc0", width=1.5),
                                                     name=f"Spread Z (AMZN/{best['_ticker']})"))
                        fig_sp.add_hline(y=2,  line_dash="dash", line_color="#ef4444", annotation_text="+2σ")
                        fig_sp.add_hline(y=-2, line_dash="dash", line_color="#10b981", annotation_text="-2σ")
                        fig_sp.add_hline(y=0,  line_dash="dot",  line_color="#7d93b8")
                        _c(fig_sp)
                        fig_sp.update_layout(yaxis_title="Z-Score")
                        st.plotly_chart(fig_sp, width='stretch')
                except Exception as ce:
                    st.markdown(_sig(f"Cointegration unavailable: {str(ce)[:100]}", "warn"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WALK-FORWARD VALIDATION + XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_wf:
    import traceback as _tb_wf
    st.markdown(_sec("Walk-Forward Validation · Rolling 252-Day Train / 21-Day Step", "teal"), unsafe_allow_html=True)
    st.caption("Realistic simulation of live deployment — model refitted on each expanding window, never sees future data.")

    st.markdown("""
    <div style="background:#0d1220;border:1px solid #192540;border-radius:4px;padding:16px 20px;margin:8px 0 16px">
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#3d5070;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Methodology</div>
      <ul style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#7d93b8;line-height:1.9;margin:0;padding-left:1.2em">
        <li>Training window: <strong style="color:#d8e3f5">252 trading days</strong> (1 year) expanding with each step</li>
        <li>Step size: <strong style="color:#d8e3f5">21 days</strong> — model retrained monthly on fresh data</li>
        <li>Models compared: <strong style="color:#d8e3f5">Linear Regression</strong> vs <strong style="color:#10b981">XGBoost</strong></li>
        <li>Metric: RMSE and MAPE on out-of-sample (test) windows only</li>
        <li>No data leakage — each prediction uses only past data available at that point in time</li>
      </ul>
    </div>""", unsafe_allow_html=True)

    # ── XGBoost with RandomizedSearchCV (lazy) ───────────────────────────────
    st.markdown(_sec("XGBoost · RandomizedSearchCV Hyperparameter Tuning", "blue"), unsafe_allow_html=True)
    st.caption("30 iterations · 5-fold CV · optimised for RMSE · 11 features including technical indicators")
    st.markdown(_sig("Training runs RandomizedSearchCV (30 iter × 5 fold). Click below to train — takes ~30–60 seconds.", "warn"), unsafe_allow_html=True)

    if st.button("▶ Train XGBoost + Walk-Forward", key="run_wf_btn"):
        try:
            with st.spinner("Training XGBoost with RandomizedSearchCV (30 iters × 5 folds)…"):
                xgb_model, best_params, xgb_rmse, xgb_mape, xgb_r2, xgb_fi, xgb_idx, xgb_actual, xgb_pred = train_xgboost(df)

            x1, x2, x3 = st.columns(3)
            x1.metric("XGBoost RMSE ($)", f"{xgb_rmse:.2f}")
            x2.metric("XGBoost MAPE (%)", f"{xgb_mape:.2f}")
            x3.metric("XGBoost R²",       f"{xgb_r2:.4f}")

            # Best hyperparameters
            params_html = "".join([
                f'<tr><td style="padding:6px 14px;font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#7d93b8;border-bottom:1px solid #192540">{k}</td>'
                f'<td style="padding:6px 14px;font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#d8e3f5;border-bottom:1px solid #192540;text-align:right">{v}</td></tr>'
                for k, v in best_params.items()
            ])
            st.markdown(f"""
            <div style="margin:12px 0">
              <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#3d5070;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Best Hyperparameters Found</div>
              <table style="width:50%;border-collapse:collapse;background:#0d1220;border:1px solid #192540;border-radius:3px">
                <tbody>{params_html}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            fig_xgb = go.Figure()
            fig_xgb.add_trace(go.Scatter(x=xgb_idx, y=xgb_actual, name="Actual",
                                          line=dict(color="#d8e3f5", width=2)))
            fig_xgb.add_trace(go.Scatter(x=xgb_idx, y=xgb_pred,   name="XGBoost",
                                          line=dict(color="#10b981", width=1.5, dash="dot")))
            _c(fig_xgb, legend_h=True)
            fig_xgb.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig_xgb, width='stretch')

            st.markdown(_sec("XGBoost Feature Importance", "teal"), unsafe_allow_html=True)
            fig_xfi = go.Figure(go.Bar(
                x=xgb_fi.values, y=xgb_fi.index, orientation="h",
                marker_color=px.colors.sequential.Viridis[:len(xgb_fi)],
            ))
            _c(fig_xfi, height=360)
            fig_xfi.update_layout(xaxis_title="Importance (Gain)")
            st.plotly_chart(fig_xfi, width='stretch')

            # Walk-forward validation
            st.markdown(_sec("Walk-Forward Validation · LR vs XGBoost", "amber"), unsafe_allow_html=True)
            with st.spinner("Running rolling walk-forward validation…"):
                wf_dates, wf_actual, wf_lr, wf_xgb = walk_forward_validation(df)

            if len(wf_dates) > 0:
                wf1, wf2, wf3, wf4 = st.columns(4)
                wf1.metric("LR RMSE ($)",  f"{np.sqrt(mean_squared_error(wf_actual, wf_lr)):.2f}")
                wf2.metric("XGB RMSE ($)", f"{np.sqrt(mean_squared_error(wf_actual, wf_xgb)):.2f}")
                wf3.metric("LR MAPE (%)",  f"{mape(wf_actual, wf_lr):.2f}")
                wf4.metric("XGB MAPE (%)", f"{mape(wf_actual, wf_xgb):.2f}")

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(x=wf_dates, y=wf_actual, name="Actual",
                                             line=dict(color="#d8e3f5", width=2)))
                fig_wf.add_trace(go.Scatter(x=wf_dates, y=wf_lr, name="LR (Walk-Fwd)",
                                             line=dict(color="#f59e0b", width=1.5, dash="dot")))
                fig_wf.add_trace(go.Scatter(x=wf_dates, y=wf_xgb, name="XGBoost (Walk-Fwd)",
                                             line=dict(color="#10b981", width=1.5, dash="dot")))
                _c(fig_wf, legend_h=True)
                fig_wf.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(fig_wf, width='stretch')

                wf_df_tmp = pd.DataFrame({"actual": wf_actual, "lr": wf_lr, "xgb": wf_xgb},
                                          index=pd.to_datetime(wf_dates))
                wf_df_tmp["lr_err2"]  = (wf_df_tmp["actual"] - wf_df_tmp["lr"])**2
                wf_df_tmp["xgb_err2"] = (wf_df_tmp["actual"] - wf_df_tmp["xgb"])**2
                roll_rmse_lr  = wf_df_tmp["lr_err2"].rolling(63).mean().apply(np.sqrt)
                roll_rmse_xgb = wf_df_tmp["xgb_err2"].rolling(63).mean().apply(np.sqrt)
                fig_rrmse = go.Figure()
                fig_rrmse.add_trace(go.Scatter(x=roll_rmse_lr.index, y=roll_rmse_lr,
                                                name="LR RMSE",  line=dict(color="#f59e0b", width=1.5)))
                fig_rrmse.add_trace(go.Scatter(x=roll_rmse_xgb.index, y=roll_rmse_xgb,
                                                name="XGB RMSE", line=dict(color="#10b981", width=1.5)))
                _c(fig_rrmse, legend_h=True)
                fig_rrmse.update_layout(yaxis_title="RMSE ($)")
                st.plotly_chart(fig_rrmse, width='stretch')

                better = "XGBoost" if mape(wf_actual, wf_xgb) < mape(wf_actual, wf_lr) else "Linear Regression"
                st.markdown(_sig(
                    f"<strong>{better}</strong> achieves lower MAPE across walk-forward windows. "
                    f"Walk-forward is more rigorous than a single train-test split — it simulates live retraining.", "bull"
                ), unsafe_allow_html=True)

        except Exception as _wf_err:
            st.error(f"Walk-Forward error: {_wf_err}")
            st.code(_tb_wf.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SIGNALS & ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sig:
    import traceback as _tb_sig
    # ── Signal Scorecard ──────────────────────────────────────────────────────
    try:
        sigs = signal_scorecard(df)
    except Exception as _sg_e:
        st.error(f"Signal scorecard error: {_sg_e}")
        st.code(_tb_sig.format_exc())
        sigs = []
    st.markdown(_sec("Technical Signal Scorecard · All Indicators"), unsafe_allow_html=True)
    bull_count = sum(1 for _, _, _, d in sigs if d == "bull")
    bear_count = sum(1 for _, _, _, d in sigs if d == "bear")
    warn_count = sum(1 for _, _, _, d in sigs if d == "warn")
    overall    = "BULLISH" if bull_count > bear_count else "BEARISH" if bear_count > bull_count else "MIXED"
    ov_var     = "bull" if overall == "BULLISH" else "bear" if overall == "BEARISH" else "warn"

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Bullish Signals", f"{bull_count} / {len(sigs)}")
    sc2.metric("Bearish Signals", f"{bear_count} / {len(sigs)}")
    sc3.metric("Caution Signals", f"{warn_count} / {len(sigs)}")
    sc4.metric("Overall Bias",    overall)
    st.markdown(_sig(
        f"<strong>{bull_count} bullish</strong> vs <strong>{bear_count} bearish</strong> signals across 7 indicators — overall bias is <strong>{overall}</strong>.",
        ov_var
    ), unsafe_allow_html=True)

    # Signal table
    sig_color = {"bull": "#10b981", "bear": "#ef4444", "warn": "#f59e0b", "": "#7d93b8"}
    sig_icon  = {"bull": "▲", "bear": "▼", "warn": "◆", "": "◉"}
    rows_html = ""
    for indicator, label, value, direction in sigs:
        color = sig_color[direction]
        icon  = sig_icon[direction]
        rows_html += f"""
        <tr>
          <td style="padding:10px 14px;font-family:'Space Mono',monospace;font-size:0.72rem;color:#7d93b8;border-bottom:1px solid #192540">{indicator}</td>
          <td style="padding:10px 14px;font-family:'Space Mono',monospace;font-size:0.72rem;border-bottom:1px solid #192540">
            <span style="color:{color};font-weight:700">{icon} {label}</span>
          </td>
          <td style="padding:10px 14px;font-family:'Space Mono',monospace;font-size:0.72rem;color:#d8e3f5;border-bottom:1px solid #192540;text-align:right">{value}</td>
        </tr>"""
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:#0d1220;border:1px solid #192540;border-radius:3px">
      <thead>
        <tr>
          <th style="padding:10px 14px;font-family:'Space Mono',monospace;font-size:0.6rem;color:#3d5070;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #1e2e4e;text-align:left">Indicator</th>
          <th style="padding:10px 14px;font-family:'Space Mono',monospace;font-size:0.6rem;color:#3d5070;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #1e2e4e;text-align:left">Signal</th>
          <th style="padding:10px 14px;font-family:'Space Mono',monospace;font-size:0.6rem;color:#3d5070;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid #1e2e4e;text-align:right">Value</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── Support & Resistance ──────────────────────────────────────────────────
    st.markdown(_sec("Support & Resistance · Classic Pivot Points", "teal"), unsafe_allow_html=True)
    st.caption("Floor-trader pivots computed from last full session's High / Low / Close.")
    pvt = pivot_points(df)
    cur_price = df["Close"].iloc[-1]

    pv_cols = st.columns(7)
    pv_labels = ["S3", "S2", "S1", "PP", "R1", "R2", "R3"]
    pv_colors = ["#ef4444","#f87171","#fca5a5","#f59e0b","#6ee7b7","#34d399","#10b981"]
    for i, lbl in enumerate(pv_labels):
        val  = pvt[lbl]
        dist = (val - cur_price) / cur_price * 100
        pv_cols[i].markdown(f"""
        <div style="text-align:center;background:#0d1220;border:1px solid #192540;border-radius:3px;padding:10px 4px">
          <div style="font-family:'Space Mono',monospace;font-size:0.55rem;color:{pv_colors[i]};font-weight:700;letter-spacing:.1em;margin-bottom:5px">{lbl}</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.9rem;color:#d8e3f5;font-weight:700">${val:.2f}</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#3d5070;margin-top:3px">{dist:+.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # Pivot chart
    recent = df["Close"].iloc[-60:]
    fig_pvt = go.Figure()
    fig_pvt.add_trace(go.Scatter(x=recent.index, y=recent, name="Close",
                                  line=dict(color="#5c6bc0", width=2)))
    pvt_style = {
        "PP":  ("#f59e0b", "dash"), "R1": ("#6ee7b7","dot"), "R2": ("#34d399","dot"), "R3": ("#10b981","dot"),
        "S1": ("#fca5a5","dot"), "S2": ("#f87171","dot"), "S3": ("#ef4444","dot"),
    }
    for name, (col, dash) in pvt_style.items():
        fig_pvt.add_hline(y=pvt[name], line_color=col, line_dash=dash, line_width=1,
                           annotation_text=f"{name} ${pvt[name]:.2f}",
                           annotation_font_color=col, annotation_font_size=9)
    _c(fig_pvt, height=340)
    fig_pvt.update_layout(yaxis_title="Price (USD)", xaxis_title="Date")
    st.plotly_chart(fig_pvt, width='stretch')

    # nearest S/R signal
    nearest_r = min([pvt["R1"], pvt["R2"], pvt["R3"]], key=lambda x: abs(x - cur_price))
    nearest_s = max([pvt["S1"], pvt["S2"], pvt["S3"]], key=lambda x: abs(x - cur_price) if x < cur_price else 99999)
    st.markdown(_sig(
        f"Nearest resistance: <strong>${nearest_r:.2f}</strong> ({(nearest_r/cur_price-1)*100:+.1f}%)  ·  "
        f"Nearest support: <strong>${nearest_s:.2f}</strong> ({(nearest_s/cur_price-1)*100:+.1f}%)",
        "warn"
    ), unsafe_allow_html=True)

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    st.markdown(_sec("Anomaly Detection · Statistical Price Move Flagging", "amber"), unsafe_allow_html=True)
    st.caption("Returns > 2.5 standard deviations from 20-day rolling mean are flagged as anomalous.")
    anom_df = detect_anomalies(df, window=20, z_thresh=2.5)

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=df["Close"].index, y=df["Close"],
                                   name="Close", line=dict(color="#5c6bc0", width=1.5)))
    if not anom_df.empty:
        up_anom   = anom_df[anom_df["Returns"] > 0]
        down_anom = anom_df[anom_df["Returns"] < 0]
        fig_anom.add_trace(go.Scatter(
            x=up_anom.index, y=up_anom["Close"], mode="markers",
            marker=dict(color="#10b981", size=9, symbol="triangle-up"),
            name=f"Positive Anomaly ({len(up_anom)})"
        ))
        fig_anom.add_trace(go.Scatter(
            x=down_anom.index, y=down_anom["Close"], mode="markers",
            marker=dict(color="#ef4444", size=9, symbol="triangle-down"),
            name=f"Negative Anomaly ({len(down_anom)})"
        ))
    _c(fig_anom, legend_h=True)
    fig_anom.update_layout(yaxis_title="Price (USD)")
    st.plotly_chart(fig_anom, width='stretch')

    if not anom_df.empty:
        anom_show = anom_df[["Close", "Returns", "z_score"]].copy()
        anom_show.index = anom_show.index.strftime("%Y-%m-%d")
        anom_show["Returns"] = anom_show["Returns"].map("{:+.2%}".format)
        anom_show["z_score"] = anom_show["z_score"].map("{:+.2f}σ".format)
        anom_show["Close"]   = anom_show["Close"].map("${:.2f}".format)
        anom_show.columns    = ["Close", "Daily Return", "Z-Score"]
        st.dataframe(anom_show.tail(15), width='stretch')
        st.markdown(_sig(f"<strong>{len(anom_df)}</strong> anomalous sessions detected over the full history — {len(up_anom)} positive, {len(down_anom)} negative.", "warn"), unsafe_allow_html=True)

    # ── Portfolio Simulator ───────────────────────────────────────────────────
    st.markdown(_sec("Portfolio Simulator · P&L Calculator", "blue"), unsafe_allow_html=True)
    st.caption("Estimate unrealised P&L and project value using model forecasts.")
    ps1, ps2, ps3 = st.columns(3)
    with ps1:
        shares   = st.number_input("Shares held", min_value=0.0, value=10.0, step=1.0)
    with ps2:
        avg_cost = st.number_input("Avg cost basis ($/share)", min_value=0.0, value=float(f"{cur_price:.2f}"), step=1.0)
    with ps3:
        fc_horizon = st.slider("Forecast horizon (days)", 1, 30, 7, key="ps_horizon")

    current_val = shares * cur_price
    cost_basis  = shares * avg_cost
    unrealised  = current_val - cost_basis
    unrealised_pct = (unrealised / cost_basis * 100) if cost_basis > 0 else 0

    # Quick LR forecast for the simulator
    mu_ps, _, _ = forecast_lr(df, fc_horizon)
    proj_price  = mu_ps.iloc[-1]
    proj_val    = shares * proj_price
    proj_pnl    = proj_val - cost_basis
    proj_pnl_pct = (proj_pnl / cost_basis * 100) if cost_basis > 0 else 0

    pa, pb, pc, pd_ = st.columns(4)
    pa.metric("Current Value",      f"${current_val:,.2f}")
    pb.metric("Unrealised P&L",     f"${unrealised:+,.2f}", f"{unrealised_pct:+.1f}%")
    pc.metric(f"Projected Price ({fc_horizon}d)", f"${proj_price:.2f}")
    pd_.metric("Projected P&L",     f"${proj_pnl:+,.2f}", f"{proj_pnl_pct:+.1f}%")

    pnl_var = "bull" if unrealised >= 0 else "bear"
    st.markdown(_sig(
        f"Holding <strong>{shares:g} shares</strong> at avg cost <strong>${avg_cost:.2f}</strong> — "
        f"current unrealised P&amp;L: <strong>${unrealised:+,.2f}</strong> ({unrealised_pct:+.1f}%). "
        f"LR model projects <strong>${proj_price:.2f}</strong> in {fc_horizon} days → projected P&amp;L: <strong>${proj_pnl:+,.2f}</strong>.",
        pnl_var
    ), unsafe_allow_html=True)

    # ── Options Greeks (Black-Scholes) ────────────────────────────────────────
    st.markdown(_sec("Options Pricing · Black-Scholes Model & Greeks", "amber"), unsafe_allow_html=True)
    st.caption("European option pricing with all five Greeks — Delta, Gamma, Theta, Vega, Rho.")
    og1, og2, og3, og4, og5 = st.columns(5)
    S_bs   = df["Close"].iloc[-1]
    K_bs   = og1.number_input("Strike (K)", value=round(S_bs), step=5.0)
    T_bs   = og2.number_input("Days to Expiry", value=30, min_value=1, max_value=365) / 365
    r_bs   = og3.number_input("Risk-Free Rate (%)", value=5.0, step=0.25) / 100
    sig_bs = og4.number_input("IV / Vol (%)", value=float(f"{df['Volatility'].iloc[-1]:.1f}"), step=1.0) / 100
    opt_t  = og5.selectbox("Type", ["call", "put"])

    price_bs, delta, gamma, theta, vega, rho = black_scholes(S_bs, K_bs, T_bs, r_bs, sig_bs, opt_t)

    greek_cols = st.columns(6)
    greek_data = [
        ("Option Price", f"${price_bs:.4f}", ""),
        ("Delta Δ",      f"{delta:.4f}",     "Sensitivity to ΔS"),
        ("Gamma Γ",      f"{gamma:.6f}",     "Rate of change of Δ"),
        ("Theta Θ",      f"${theta:.4f}/day","Time decay"),
        ("Vega V",       f"${vega:.4f}/%",   "Sensitivity to Δvol"),
        ("Rho ρ",        f"${rho:.4f}/%",    "Sensitivity to Δr"),
    ]
    for col, (name, val, sub) in zip(greek_cols, greek_data):
        col.markdown(f"""
        <div style="background:#0d1220;border:1px solid #192540;border-radius:3px;padding:12px 10px;text-align:center">
          <div style="font-family:'Space Mono',monospace;font-size:0.55rem;color:#3d5070;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px">{name}</div>
          <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#d8e3f5;font-weight:700">{val}</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.55rem;color:#3d5070;margin-top:4px">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # IV surface: price across strikes
    st.markdown(_sec("Option Price Surface · Strike vs IV", "blue"), unsafe_allow_html=True)
    strikes_range = np.linspace(S_bs * 0.7, S_bs * 1.3, 30)
    iv_range      = np.linspace(0.10, 0.70, 20)
    Z_surf = np.array([[black_scholes(S_bs, k, T_bs, r_bs, iv, opt_t)[0]
                         for k in strikes_range] for iv in iv_range])
    fig_surf = go.Figure(go.Surface(
        x=strikes_range, y=iv_range * 100, z=Z_surf,
        colorscale="Viridis", opacity=0.85,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#10b981", project_z=True))
    ))
    fig_surf.update_layout(
        scene=dict(
            xaxis=dict(title="Strike ($)", backgroundcolor="#05080f", gridcolor="#192540",
                       tickfont=dict(color="#3d5070", size=9)),
            yaxis=dict(title="IV (%)",    backgroundcolor="#05080f", gridcolor="#192540",
                       tickfont=dict(color="#3d5070", size=9)),
            zaxis=dict(title="Option Price ($)", backgroundcolor="#05080f", gridcolor="#192540",
                       tickfont=dict(color="#3d5070", size=9)),
            bgcolor="#05080f",
        ),
        paper_bgcolor="#05080f", font=dict(color="#7d93b8", family="Space Mono, monospace"),
        margin=dict(l=0, r=0, t=20, b=0), height=460,
    )
    st.plotly_chart(fig_surf, width='stretch')
    moneyness = "IN-THE-MONEY" if (S_bs > K_bs and opt_t == "call") or (S_bs < K_bs and opt_t == "put") else "OUT-OF-THE-MONEY" if (S_bs < K_bs and opt_t == "call") or (S_bs > K_bs and opt_t == "put") else "AT-THE-MONEY"
    st.markdown(_sig(
        f"{opt_t.upper()} option — S=${S_bs:.2f}, K=${K_bs:.2f} → <strong>{moneyness}</strong>. "
        f"Delta <strong>{delta:.4f}</strong> means a $1 move in AMZN → ~${abs(delta)*shares:.2f} P&amp;L on {shares:.0f} shares. "
        f"Theta <strong>${theta:.4f}/day</strong> — time decay cost per day held.",
        "bull" if moneyness == "IN-THE-MONEY" else "warn"
    ), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — NEWS & SENTIMENT (FinBERT)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    import traceback as _tb_news
    st.markdown(_sec("News Sentiment · FinBERT · Financial Domain NLP", "teal"), unsafe_allow_html=True)
    st.caption("ProsusAI/FinBERT — BERT fine-tuned on 10,000+ financial news articles. Far more accurate than general-purpose NLP for financial text.")

    if not news:
        st.info("No recent AMZN news articles found.")
    else:
        with st.spinner("Running FinBERT inference on headlines…"):
            try:
                finbert_pipe = load_finbert()
                fb_results   = finbert_sentiment(finbert_pipe, [e.title for e in news[:15]])
                fb_labels    = [r[0] for r in fb_results]
                fb_scores    = [r[1] for r in fb_results]
                model_name   = "FinBERT"
            except Exception:
                # Graceful fallback if transformers unavailable
                from textblob import TextBlob
                fb_labels = []
                fb_scores = []
                for e in news[:15]:
                    p = TextBlob(e.title).sentiment.polarity
                    fb_scores.append(p)
                    fb_labels.append("positive" if p > 0.05 else "negative" if p < -0.05 else "neutral")
                model_name = "TextBlob (fallback)"

        pos_count  = fb_labels.count("positive")
        neg_count  = fb_labels.count("negative")
        neu_count  = fb_labels.count("neutral")
        avg_pol    = np.mean(fb_scores)
        overall_lb = "Positive" if avg_pol > 0.05 else "Negative" if avg_pol < -0.05 else "Neutral"
        sent_color = "#10b981" if avg_pol > 0.05 else "#ef4444" if avg_pol < -0.05 else "#f59e0b"

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Model", model_name)
        m2.metric("Articles", len(news[:15]))
        m3.metric("🟢 Positive", pos_count)
        m4.metric("🔴 Negative", neg_count)
        m5.metric("Overall", overall_lb)

        sv = "bull" if avg_pol > 0.05 else "bear" if avg_pol < -0.05 else ""
        st.markdown(_sig(
            f"<strong>{model_name}</strong> classifies <strong>{pos_count} positive</strong>, "
            f"<strong>{neg_count} negative</strong>, <strong>{neu_count} neutral</strong> headlines. "
            f"Aggregate sentiment: <strong style='color:{sent_color}'>{overall_lb}</strong> (mean polarity {avg_pol:+.3f}).",
            sv
        ), unsafe_allow_html=True)

        # Sentiment distribution donut
        dc, bc = st.columns([1, 2])
        with dc:
            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[pos_count, neg_count, neu_count],
                hole=0.6,
                marker=dict(colors=["#10b981", "#ef4444", "#f59e0b"]),
                textfont=dict(family="Space Mono, monospace", size=10, color="#d8e3f5"),
            ))
            fig_pie.update_traces(hovertemplate="%{label}: %{value} articles")
            _c(fig_pie, height=220)
            fig_pie.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pie, width='stretch')

        with bc:
            titles_short = [e.title[:60] + "…" if len(e.title) > 60 else e.title for e in news[:10]]
            fig_sent = go.Figure(go.Bar(
                x=fb_scores[:10], y=titles_short, orientation="h",
                marker_color=["#10b981" if s > 0.05 else "#ef4444" if s < -0.05 else "#f59e0b"
                               for s in fb_scores[:10]],
            ))
            fig_sent.add_vline(x=0, line_dash="dash", line_color="#7b8ab8")
            _c(fig_sent, height=330)
            fig_sent.update_layout(xaxis_title="FinBERT Polarity Score")
            st.plotly_chart(fig_sent, width='stretch')

        filt = st.selectbox("Filter articles", ["All", "Positive", "Neutral", "Negative"])
        st.markdown("---")
        for entry, label, score in zip(news[:15], fb_labels, fb_scores):
            if filt == "Positive" and label != "positive": continue
            if filt == "Negative" and label != "negative": continue
            if filt == "Neutral"  and label != "neutral":  continue
            badge = "🟢 Positive" if label == "positive" else "🔴 Negative" if label == "negative" else "🟡 Neutral"
            col_badge = "#10b981" if label == "positive" else "#ef4444" if label == "negative" else "#f59e0b"
            st.markdown(f"**[{entry.title}]({entry.link})**")
            st.caption(f"<span style='color:{col_badge}'>{badge}</span>  ·  FinBERT score: {score:+.4f}  ·  {entry.get('published', '')}", unsafe_allow_html=True)
            st.markdown(f"{entry.get('summary', '')[:220]}…")
            st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(_sec("Project Overview"), unsafe_allow_html=True)
    st.markdown('<p style="color:#7d93b8;font-size:0.9rem;line-height:1.7;margin-bottom:20px">An end-to-end data science pipeline applied to financial time-series — combining statistical modelling, deep learning architectures, stochastic simulation, and quantitative risk analytics.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-block">
          <span class="ib-title">Data Pipeline</span>
          Live 2-year OHLCV data via <code>yfinance</code><br>
          12 engineered features: RSI, MACD, Bollinger Bands, MAs, Volatility, Volume ratio<br>
          MinMax normalisation — scaler fit on train split only (no leakage)<br>
          80/20 chronological train-test split
        </div>
        <div class="info-block">
          <span class="ib-title">Linear Regression Baseline</span>
          Date-ordinal feature, fits linear trend on last 120 days<br>
          Analytical 95% prediction interval widens with √(1 + h/n)<br>
          Used as baseline for deep learning comparison
        </div>
        <div class="info-block">
          <span class="ib-title">Monte Carlo · GBM</span>
          Geometric Brownian Motion with μ and σ from last 252 trading days<br>
          500 simulation paths — reports P10/P50/P90 and P(price rises)
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-block">
          <span class="ib-title">Seq2Seq LSTM</span>
          2-layer encoder reads the 30-day input window × 1 feature (Close)<br>
          2-layer decoder unrolls 7-step output directly — no autoregressive error accumulation<br>
          95% CI band available; direct multi-step horizon forecasting
        </div>
        <div class="info-block">
          <span class="ib-title">Random Forest · Explainability</span>
          200 trees trained on 10 technical indicator features<br>
          Feature importance via mean decrease in impurity<br>
          Reveals which signals drive next-day price prediction
        </div>
        <div class="info-block">
          <span class="ib-title">Signals & Anomaly Detection</span>
          7-indicator signal scorecard — RSI, MACD, Bollinger, MA crossovers, Volume, Volatility<br>
          Classic floor-trader pivot points (PP, R1/2/3, S1/2/3)<br>
          Z-score anomaly detection on rolling 20-day window (±2.5σ threshold)<br>
          Volatility regime detection — 30D vs 90D rolling vol comparison<br>
          Portfolio P&L simulator with LR-projected price
        </div>
        """, unsafe_allow_html=True)

    st.markdown(_sec("Tech Stack", "amber"), unsafe_allow_html=True)
    st.markdown("""
    | Layer | Tools |
    |---|---|
    | Data ingestion | `yfinance`, `feedparser`, CSV fallback |
    | Feature engineering | `ta`, `pandas`, `numpy` |
    | Deep learning | `PyTorch` — LSTM, Seq2Seq Encoder-Decoder |
    | Classical / ensemble ML | `scikit-learn` — LR, Random Forest |
    | Statistics | `scipy` — normality tests, distribution fitting, ACF |
    | Risk analytics | VaR/CVaR, Sharpe, Sortino, Calmar, Max Drawdown, Rolling VaR |
    | Simulation | Monte Carlo GBM — 500 paths, P10/P50/P90 |
    | Anomaly detection | Z-score rolling window flagging |
    | Technical signals | Pivot points, signal scorecard, regime detection |
    | NLP / Sentiment | `TextBlob` |
    | Visualisation | `Plotly` |
    | App framework | `Streamlit` 1.31+ |
    | Deployment | Streamlit Cloud (free, serverless) |
    """)

    st.markdown(_sec("Connect"), unsafe_allow_html=True)
    st.markdown('<p style="color:#7d93b8;font-size:0.9rem">👨‍💻 <a href="https://www.linkedin.com/in/karthikmulugu/" style="color:#3b82f6">Karthik Mulugu</a> &nbsp;·&nbsp; 🐙 <a href="https://github.com/Karthik0809/Amazon-Stock-Dashboard" style="color:#3b82f6">GitHub Repository</a></p>', unsafe_allow_html=True)
