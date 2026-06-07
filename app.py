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
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from seq2seq_lstm import Seq2SeqLSTM, make_multi_sequences

SEQ_LEN = 30
PEERS   = ["AMZN", "MSFT", "GOOGL", "META", "AAPL", "SPY"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMZN Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .kpi-box {
        background: #1c2130; border: 1px solid #2a3352;
        border-radius: 10px; padding: 18px; text-align: center;
    }
    .kpi-label {
        color: #7b8ab8; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.07em;
    }
    .kpi-value { color: #eef0f8; font-size: 1.65rem; font-weight: 800; margin: 6px 0 2px; }
    .kpi-up   { color: #26a69a; font-size: 0.82rem; }
    .kpi-down { color: #ef5350; font-size: 0.82rem; }
    .insight  {
        background: #151b2d; border-left: 3px solid #5c6bc0;
        border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
        font-size: 0.95rem;
    }
    .arch-block {
        background: #1c2130; border: 1px solid #2a3352;
        border-radius: 8px; padding: 16px; margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = yf.Ticker("AMZN").history(period="2y")
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    df.index = df.index.tz_localize(None)
    return df

@st.cache_data(ttl=3600)
def load_peers():
    frames = {}
    for t in PEERS:
        try:
            d = yf.Ticker(t).history(period="1y")
            d.index = pd.to_datetime(d.index).tz_localize(None)
            frames[t] = d["Close"]
        except Exception:
            pass
    return pd.DataFrame(frames).dropna()

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
df["Vol_ratio"]   = df["Volume"] / df["Volume"].rolling(20).mean()

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

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

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
st.title("📊 Amazon (AMZN) Stock Intelligence Dashboard")
st.caption(
    "Live market data · ML forecasting · Monte Carlo simulation · Risk analytics · "
    "Peer comparison · ⚠️ Educational project — not financial advice"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
cur     = df["Close"].iloc[-1]
prev    = df["Close"].iloc[-2]
day_chg = (cur - prev) / prev * 100
high52  = df["Close"].rolling(252).max().iloc[-1]
low52   = df["Close"].rolling(252).min().iloc[-1]
ytd_s   = df.loc[df.index >= f"{df.index[-1].year}-01-01", "Close"]
ytd_ret = (cur / ytd_s.iloc[0] - 1) * 100 if len(ytd_s) > 0 else 0.0
vol30   = df["Volatility"].iloc[-1]

kpis = [
    ("Current Price",     f"${cur:.2f}",      f"{'▲' if day_chg >= 0 else '▼'} {day_chg:+.2f}% today",  day_chg >= 0),
    ("52-Week High",      f"${high52:.2f}",   f"{(cur/high52-1)*100:.1f}% from high",                     cur >= high52 * 0.97),
    ("52-Week Low",       f"${low52:.2f}",    f"{(cur/low52-1)*100:.1f}% from low",                       True),
    ("YTD Return",        f"{ytd_ret:+.1f}%", "year-to-date",                                             ytd_ret >= 0),
    ("30-Day Volatility", f"{vol30:.1f}%",    "annualised",                                               vol30 < 30),
]
cols = st.columns(5)
for col, (label, value, delta, is_pos) in zip(cols, kpis):
    css = "kpi-up" if is_pos else "kpi-down"
    col.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="{css}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fc, tab_tech, tab_risk, tab_mkt, tab_models, tab_news, tab_about = st.tabs([
    "🔮 Forecast", "📉 Technical Analysis", "📊 Risk Analytics",
    "🌐 Market Context", "🤖 Model Performance", "📰 News & Sentiment", "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_fc:
    ctrl, chart_col = st.columns([1, 3])
    with ctrl:
        model_opt    = st.selectbox("Forecast model", ["Linear Regression", "One-Step LSTM", "Seq2Seq LSTM"])
        n_days       = st.slider("Horizon (days)", 1, 30, 7)
        if model_opt == "Seq2Seq LSTM" and n_days != 7:
            st.info("Seq2Seq is trained for 7-day output — setting to 7.")
            n_days = 7
        show_ci      = st.checkbox("Show 95% confidence band", value=True)
        show_mc      = st.checkbox("Show Monte Carlo simulation", value=False)
        context_days = st.slider("Historical context (days)", 30, 120, 60)

    with st.spinner("Running forecast…"):
        if model_opt == "Linear Regression":
            mu, lo, hi = forecast_lr(df, n_days)
        elif model_opt == "One-Step LSTM":
            mu, lo, hi = forecast_lstm(df, SEQ_LEN, n_days)
        else:
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

        fig.add_trace(go.Scatter(x=mu.index, y=mu, name=f"{model_opt} Forecast",
                                 line=dict(color="#26a69a", width=2.5, dash="dash")))
        if show_ci and lo is not None:
            fig.add_trace(go.Scatter(
                x=list(mu.index) + list(mu.index[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself", fillcolor="rgba(38,166,154,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI"
            ))
        fig.update_layout(template="plotly_dark", hovermode="x unified",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          legend=dict(orientation="h", y=1.02),
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    color     = "#26a69a" if trend_up else "#ef5350"
    direction = "upward 📈" if trend_up else "downward 📉"
    st.markdown(f"""
    <div class="insight">
        <strong>{model_opt}</strong> projects a
        <strong style="color:{color}">{direction} trend</strong>
        over {n_days} day(s):
        <strong>${current_price:.2f} → ${end_price:.2f} ({pct_chg:+.2f}%)</strong>
    </div>""", unsafe_allow_html=True)

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

    with st.expander("View forecast table"):
        fdf = mu.reset_index()
        fdf.columns = ["Date", "Forecast ($)"]
        fdf["Date"] = pd.to_datetime(fdf["Date"]).dt.strftime("%Y-%m-%d")
        if lo is not None:
            fdf["Lower 95%"] = [f"${v:.2f}" for v in lo.values]
            fdf["Upper 95%"] = [f"${v:.2f}" for v in hi.values]
        fdf["Forecast ($)"] = fdf["Forecast ($)"].map("${:.2f}".format)
        st.dataframe(fdf, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    period_opt = st.selectbox("Time range", ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"], index=2, key="tech_period")
    period_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "2 Years": 504}
    dv = df.iloc[-period_map[period_opt]:]

    st.subheader("Price, Moving Averages & Bollinger Bands")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["Close"], name="Close", line=dict(color="#5c6bc0", width=2)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["MA7"],   name="MA 7",  line=dict(color="#ffb300", width=1.2)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["MA30"],  name="MA 30", line=dict(color="#ef5350", width=1.2)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["MA90"],  name="MA 90", line=dict(color="#ab47bc", width=1.2)))
    fig_price.add_trace(go.Scatter(
        x=list(dv.index) + list(dv.index[::-1]),
        y=list(dv["BB_upper"]) + list(dv["BB_lower"][::-1]),
        fill="toself", fillcolor="rgba(92,107,192,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Bollinger Bands"
    ))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["BB_upper"], name="BB Upper", line=dict(color="#5c6bc0", dash="dot", width=1)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["BB_lower"], name="BB Lower", line=dict(color="#5c6bc0", dash="dot", width=1)))
    fig_price.update_layout(template="plotly_dark", hovermode="x unified",
                             xaxis_title="Date", yaxis_title="Price (USD)", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.subheader("RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI", line=dict(color="#26a69a", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.07)", line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.07)", line_width=0)
        fig_rsi.update_layout(template="plotly_dark", yaxis=dict(range=[0, 100]),
                               margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
        st.plotly_chart(fig_rsi, use_container_width=True)
        rsi_val = dv["RSI"].iloc[-1]
        if rsi_val > 70:
            st.markdown('<div class="insight">RSI > 70 — potentially <strong style="color:#ef5350">overbought</strong>.</div>', unsafe_allow_html=True)
        elif rsi_val < 30:
            st.markdown('<div class="insight">RSI < 30 — potentially <strong style="color:#26a69a">oversold</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight">RSI = {rsi_val:.1f} — neutral zone.</div>', unsafe_allow_html=True)

    with col_macd:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=dv.index, y=dv["MACD"],        name="MACD",   line=dict(color="#5c6bc0", width=2)))
        fig_macd.add_trace(go.Scatter(x=dv.index, y=dv["MACD_Signal"], name="Signal", line=dict(color="#ffb300", width=1.5)))
        fig_macd.add_trace(go.Bar(x=dv.index, y=dv["MACD_Hist"], name="Histogram",
                                  marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in dv["MACD_Hist"]], opacity=0.7))
        fig_macd.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_macd, use_container_width=True)
        if dv["MACD"].iloc[-1] > dv["MACD_Signal"].iloc[-1]:
            st.markdown('<div class="insight">MACD above Signal — <strong style="color:#26a69a">bullish momentum</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight">MACD below Signal — <strong style="color:#ef5350">bearish momentum</strong>.</div>', unsafe_allow_html=True)

    # Volume
    st.subheader("Volume")
    avg_vol = dv["Volume"].mean()
    fig_vol = go.Figure(go.Bar(x=dv.index, y=dv["Volume"],
                                marker_color=["#26a69a" if v >= avg_vol else "#ef5350" for v in dv["Volume"]]))
    fig_vol.add_hline(y=avg_vol, line_dash="dash", line_color="#ffb300",
                      annotation_text=f"20-day avg {avg_vol/1e6:.0f}M")
    fig_vol.update_layout(template="plotly_dark", yaxis_title="Shares",
                           hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_vol, use_container_width=True)

    # Autocorrelation of returns
    st.subheader("Return Autocorrelation (ACF)")
    st.caption("Tests whether past returns predict future returns (efficient market hypothesis check).")
    rets_clean = dv["Returns"].dropna().values
    max_lag    = 30
    acf_vals   = [pd.Series(rets_clean).autocorr(lag=i) for i in range(1, max_lag + 1)]
    conf_bound = 1.96 / np.sqrt(len(rets_clean))
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(1, max_lag + 1)), y=acf_vals,
                              marker_color=["#ef5350" if abs(v) > conf_bound else "#5c6bc0" for v in acf_vals],
                              name="ACF"))
    fig_acf.add_hline(y=conf_bound,  line_dash="dash", line_color="#ffb300", annotation_text="95% CI")
    fig_acf.add_hline(y=-conf_bound, line_dash="dash", line_color="#ffb300")
    fig_acf.update_layout(template="plotly_dark", xaxis_title="Lag (days)", yaxis_title="Autocorrelation",
                           margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_acf, use_container_width=True)
    sig_lags = [i+1 for i, v in enumerate(acf_vals) if abs(v) > conf_bound]
    if sig_lags:
        st.markdown(f'<div class="insight">Significant autocorrelation at lags {sig_lags} — returns are not fully independent, suggesting some predictability.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight">No significant autocorrelation detected — consistent with the Efficient Market Hypothesis.</div>', unsafe_allow_html=True)

    # RSI Backtest vs Buy & Hold
    st.subheader("RSI < 30 Strategy vs Buy & Hold")
    bt       = backtest_rsi(dv)
    buy_hold = (1 + dv["Returns"].fillna(0)).cumprod()
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt,        name="RSI Strategy", line=dict(color="#26a69a", width=2)))
    fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name="Buy & Hold", line=dict(color="#5c6bc0", width=2, dash="dot")))
    fig_bt.update_layout(template="plotly_dark", yaxis_title="Cumulative Return",
                          hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_bt, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    rm = compute_risk_metrics(df["Returns"])

    st.subheader("Risk & Return Metrics  (2-year window, 5% risk-free rate)")
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

    sharpe_note = "above 1.0 — good risk-adjusted return" if rm["sharpe"] > 1 else "below 1.0 — return doesn't fully compensate risk"
    st.markdown(f'<div class="insight">Sharpe ratio of <strong>{rm["sharpe"]:.3f}</strong> is {sharpe_note}. '
                f'CVaR 95% of <strong>{rm["cvar95"]*100:.2f}%</strong> means on the worst 5% of days, average loss is that amount.</div>',
                unsafe_allow_html=True)

    # Drawdown chart
    st.subheader("Underwater Chart (Drawdown from Peak)")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=rm["dd"].index, y=rm["dd"] * 100,
                                 fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
                                 line=dict(color="#ef5350", width=1.5), name="Drawdown %"))
    fig_dd.update_layout(template="plotly_dark", yaxis_title="Drawdown (%)",
                          hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_dd, use_container_width=True)

    # Rolling Sharpe
    st.subheader("Rolling 90-Day Sharpe Ratio")
    roll_ret = df["Returns"].rolling(90).mean()
    roll_std = df["Returns"].rolling(90).std()
    roll_sharpe = (roll_ret - 0.05/252) / roll_std * np.sqrt(252)
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe,
                                 line=dict(color="#5c6bc0", width=2), name="Rolling Sharpe"))
    fig_rs.add_hline(y=1, line_dash="dash", line_color="#26a69a", annotation_text="Sharpe = 1")
    fig_rs.add_hline(y=0, line_dash="dash", line_color="#ef5350")
    fig_rs.update_layout(template="plotly_dark", yaxis_title="Sharpe Ratio",
                          hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_rs, use_container_width=True)

    # Return distribution
    st.subheader("Return Distribution")
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
        fig_dist.update_layout(template="plotly_dark", xaxis_title="Daily Return",
                                yaxis_title="Density", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_stats:
        st.markdown("**Distribution Statistics**")
        st.markdown(f"- **Skewness:** {rm['skew']:.4f}")
        st.markdown(f"- **Kurtosis:** {rm['kurt']:.4f}  (Normal = 0)")
        st.markdown(f"- **Shapiro-Wilk p:** {rm['p_normal']:.4f}")
        if rm["p_normal"] < 0.05:
            st.markdown("- ❗ Returns are **not normally distributed** (fat tails / heavy skew)")
        else:
            st.markdown("- ✅ Cannot reject normality (p > 0.05)")
        if rm["kurt"] > 0:
            st.markdown("- ⚠️ Positive excess kurtosis — **fat tails**, extreme moves more likely than normal distribution predicts")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mkt:
    with st.spinner("Loading peer data…"):
        peers_df = load_peers()

    if peers_df.empty:
        st.error("Could not load peer data. Try refreshing.")
    else:
        # Normalised performance
        st.subheader("Relative Performance (Normalised to 100)")
        norm_peers = peers_df / peers_df.iloc[0] * 100
        fig_peers = go.Figure()
        colors_p = ["#5c6bc0", "#26a69a", "#ffb300", "#ef5350", "#ab47bc", "#78909c"]
        for i, col in enumerate(norm_peers.columns):
            width = 3 if col == "AMZN" else 1.5
            fig_peers.add_trace(go.Scatter(x=norm_peers.index, y=norm_peers[col],
                                            name=col, line=dict(color=colors_p[i % len(colors_p)], width=width)))
        fig_peers.add_hline(y=100, line_dash="dash", line_color="#7b8ab8")
        fig_peers.update_layout(template="plotly_dark", yaxis_title="Indexed Price (Base=100)",
                                 hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_peers, use_container_width=True)

        # Returns table
        st.subheader("Return Comparison")
        def period_return(s, days):
            s = s.dropna()
            if len(s) < days: return np.nan
            return (s.iloc[-1] / s.iloc[-days] - 1) * 100

        ret_rows = []
        for t in peers_df.columns:
            s = peers_df[t]
            ret_rows.append({
                "Ticker": t,
                "1M (%)":  period_return(s, 21),
                "3M (%)":  period_return(s, 63),
                "6M (%)":  period_return(s, 126),
                "1Y (%)":  period_return(s, 252),
                "Volatility (%)": s.pct_change().dropna().std() * np.sqrt(252) * 100,
            })
        ret_df = pd.DataFrame(ret_rows).set_index("Ticker")

        def color_returns(val):
            if pd.isna(val): return ""
            return "color: #26a69a" if val >= 0 else "color: #ef5350"

        st.dataframe(
            ret_df.style
                .format("{:.1f}", subset=["1M (%)", "3M (%)", "6M (%)", "1Y (%)", "Volatility (%)"])
                .applymap(color_returns, subset=["1M (%)", "3M (%)", "6M (%)", "1Y (%)"]),
            use_container_width=True
        )

        # Correlation heatmap
        st.subheader("Correlation Matrix (Daily Returns)")
        ret_corr = peers_df.pct_change().dropna().corr()
        fig_corr = px.imshow(
            ret_corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        fig_corr.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Beta vs SPY
        st.subheader("Beta vs S&P 500 (SPY)")
        if "SPY" in peers_df.columns:
            spy_rets  = peers_df["SPY"].pct_change().dropna()
            beta_rows = []
            for t in [c for c in peers_df.columns if c != "SPY"]:
                stk_rets = peers_df[t].pct_change().dropna()
                common   = stk_rets.index.intersection(spy_rets.index)
                cov  = np.cov(stk_rets[common], spy_rets[common])[0, 1]
                var  = spy_rets[common].var()
                beta = cov / var if var > 0 else np.nan
                beta_rows.append({"Ticker": t, "Beta": beta})
            beta_df = pd.DataFrame(beta_rows)
            fig_beta = go.Figure(go.Bar(
                x=beta_df["Ticker"], y=beta_df["Beta"],
                marker_color=["#26a69a" if b < 1 else "#ef5350" for b in beta_df["Beta"]],
            ))
            fig_beta.add_hline(y=1, line_dash="dash", line_color="#ffb300", annotation_text="Market Beta = 1")
            fig_beta.update_layout(template="plotly_dark", yaxis_title="Beta",
                                   margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_beta, use_container_width=True)
            amzn_beta = beta_df.loc[beta_df["Ticker"] == "AMZN", "Beta"].values
            if len(amzn_beta) > 0:
                b = amzn_beta[0]
                direction = "more" if b > 1 else "less"
                st.markdown(f'<div class="insight">AMZN Beta = <strong>{b:.2f}</strong> — the stock is <strong>{direction} volatile</strong> than the S&P 500. A 1% market move historically corresponds to a ~{b:.2f}% move in AMZN.</div>', unsafe_allow_html=True)

        # Rolling 60-day correlation with SPY
        if "SPY" in peers_df.columns:
            st.subheader("Rolling 60-Day Correlation: AMZN vs SPY")
            roll_corr = peers_df["AMZN"].pct_change().rolling(60).corr(peers_df["SPY"].pct_change())
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr,
                                         fill="tozeroy", fillcolor="rgba(92,107,192,0.15)",
                                         line=dict(color="#5c6bc0", width=2), name="Rolling Corr"))
            fig_rc.update_layout(template="plotly_dark", yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
                                  hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_rc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("Out-of-Sample Evaluation  (80/20 train-test split)")
    st.caption("All metrics computed on the held-out 20% test set — data never seen during training.")

    split    = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    # LR test predictions
    df_all      = df.copy()
    df_all["ord"] = df_all.index.map(pd.Timestamp.toordinal)
    lr_mdl      = LinearRegression().fit(
        df_all["ord"].iloc[:split].values.reshape(-1, 1),
        df_all["Close"].iloc[:split].values
    )
    lr_pred_test = lr_mdl.predict(df_all["ord"].iloc[split:].values.reshape(-1, 1))

    # LSTM test predictions
    feats       = df[["Open", "High", "Low", "Close", "Volume"]].values
    scaler_lstm = MinMaxScaler().fit(feats[:split])
    norm_lstm   = scaler_lstm.transform(feats)
    lstm_mdl    = OneStepLSTM(inp_size=5).cpu()
    lstm_mdl.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu", weights_only=False))
    lstm_mdl.eval()
    X_lstm, y_lstm = make_multi_sequences(norm_lstm, seq_len=SEQ_LEN, horizon=1)
    with torch.no_grad():
        ph_lstm = lstm_mdl(X_lstm.float()).numpy().flatten()
    dummy4      = np.zeros((len(ph_lstm), 4))
    inv_lstm    = scaler_lstm.inverse_transform(np.hstack([dummy4, ph_lstm.reshape(-1, 1)]))[:, -1]
    inv_y_lst   = scaler_lstm.inverse_transform(np.hstack([dummy4, y_lstm.numpy().reshape(-1, 1)]))[:, -1]
    idx_lstm    = df.index[SEQ_LEN: SEQ_LEN + len(inv_lstm)]
    test_mask_lstm = (idx_lstm >= df_test.index[0]).values

    # Seq2Seq test predictions
    closes_all  = df[["Close"]].values
    scaler_seq  = MinMaxScaler().fit(closes_all[:split])
    norm_seq    = scaler_seq.transform(closes_all)
    X_seq, y_seq = make_multi_sequences(norm_seq, seq_len=SEQ_LEN, horizon=1)
    seq_mdl     = Seq2SeqLSTM(input_dim=1).cpu()
    seq_mdl.load_state_dict(torch.load("seq2seq_lstm.pth", map_location="cpu", weights_only=False))
    seq_mdl.eval()
    with torch.no_grad():
        ph_seq = seq_mdl(X_seq.float()).numpy().flatten()
    inv_seq     = scaler_seq.inverse_transform(ph_seq.reshape(-1, 1)).flatten()
    inv_y_seq   = scaler_seq.inverse_transform(y_seq.numpy().reshape(-1, 1)).flatten()
    idx_seq     = df.index[SEQ_LEN: SEQ_LEN + len(inv_seq)]
    test_mask_seq = (idx_seq >= df_test.index[0]).values

    def metrics(actual, pred):
        n   = min(len(actual), len(pred))
        a, p = actual[:n], pred[:n]
        return (np.sqrt(mean_squared_error(a, p)),
                np.mean(np.abs(a - p)),
                mape(a, p),
                r2_score(a, p), n)

    act_lr = df_test["Close"].values
    pr_lr  = lr_pred_test
    pr_ls  = inv_lstm[test_mask_lstm]
    act_ls = inv_y_lst[test_mask_lstm]
    pr_sq  = inv_seq[test_mask_seq]
    act_sq = inv_y_seq[test_mask_seq]

    rows = []
    for name, act, pred in [("Linear Regression", act_lr, pr_lr),
                              ("One-Step LSTM",     act_ls, pr_ls),
                              ("Seq2Seq LSTM",      act_sq, pr_sq)]:
        rm_v, ma_v, mp_v, r2_v, n_v = metrics(act, pred)
        rows.append({"Model": name, "RMSE ($)": rm_v, "MAE ($)": ma_v, "MAPE (%)": mp_v, "R²": r2_v, "Test n": n_v})

    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.style
            .format({"RMSE ($)": "{:.2f}", "MAE ($)": "{:.2f}", "MAPE (%)": "{:.2f}", "R²": "{:.4f}"})
            .highlight_min(subset=["RMSE ($)", "MAE ($)", "MAPE (%)"], color="#1e3a2f")
            .highlight_max(subset=["R²"], color="#1e3a2f"),
        use_container_width=True, hide_index=True,
    )

    best_model = metrics_df.loc[metrics_df["RMSE ($)"].idxmin(), "Model"]
    st.markdown(f'<div class="insight">🏆 <strong>{best_model}</strong> achieves the lowest test-set RMSE.</div>', unsafe_allow_html=True)

    # Visual comparison
    st.subheader("Predicted vs Actual — Test Period")
    min_lr = min(len(act_lr), len(pr_lr))
    min_ls = min(len(act_ls), len(pr_ls))
    min_sq = min(len(act_sq), len(pr_sq))
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=df_test.index[:min_lr], y=act_lr[:min_lr],
                                  name="Actual", line=dict(color="#eef0f8", width=2)))
    fig_cmp.add_trace(go.Scatter(x=df_test.index[:min_lr], y=pr_lr[:min_lr],
                                  name="Linear Regression", line=dict(color="#ffb300", width=1.5, dash="dot")))
    fig_cmp.add_trace(go.Scatter(x=idx_lstm[test_mask_lstm][:min_ls], y=pr_ls[:min_ls],
                                  name="One-Step LSTM", line=dict(color="#26a69a", width=1.5, dash="dot")))
    fig_cmp.add_trace(go.Scatter(x=idx_seq[test_mask_seq][:min_sq], y=pr_sq[:min_sq],
                                  name="Seq2Seq LSTM", line=dict(color="#5c6bc0", width=1.5, dash="dot")))
    fig_cmp.update_layout(template="plotly_dark", hovermode="x unified",
                           xaxis_title="Date", yaxis_title="Price (USD)",
                           margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Residuals
    st.subheader("Residual Analysis")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df_test.index[:min_lr],
                                  y=(act_lr[:min_lr] - pr_lr[:min_lr]),
                                  name="LR Residuals", line=dict(color="#ffb300")))
    fig_res.add_trace(go.Scatter(x=idx_lstm[test_mask_lstm][:min_ls],
                                  y=(act_ls[:min_ls] - pr_ls[:min_ls]),
                                  name="LSTM Residuals", line=dict(color="#26a69a")))
    fig_res.add_hline(y=0, line_dash="dash", line_color="#7b8ab8")
    fig_res.update_layout(template="plotly_dark", hovermode="x unified",
                           yaxis_title="Residual ($)", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_res, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance (Random Forest — predicting next-day Close)")
    st.caption("Trained on all available data. Importance = mean decrease in impurity across 200 trees.")
    with st.spinner("Training Random Forest for feature importance…"):
        fi = compute_feature_importance(df)
    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation="h",
        marker_color=px.colors.sequential.Plasma_r[:len(fi)],
    ))
    fig_fi.update_layout(template="plotly_dark", xaxis_title="Importance",
                          margin=dict(l=0, r=0, t=20, b=0), height=350)
    st.plotly_chart(fig_fi, use_container_width=True)
    top_feat = fi.index[-1]
    st.markdown(f'<div class="insight"><strong>{top_feat}</strong> is the most predictive feature of next-day closing price according to the Random Forest model.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — NEWS & SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    if not news:
        st.info("No recent AMZN news articles found.")
    else:
        scores    = [TextBlob(e.title).sentiment.polarity for e in news[:15]]
        avg_sent  = np.mean(scores)
        sent_label = "Positive 🟢" if avg_sent > 0.05 else "Negative 🔴" if avg_sent < -0.05 else "Neutral 🟡"
        sent_color = "#26a69a" if avg_sent > 0.05 else "#ef5350" if avg_sent < -0.05 else "#ffb300"

        m1, m2, m3 = st.columns(3)
        m1.metric("Articles Analysed", len(news[:15]))
        m2.metric("Avg Sentiment Score", f"{avg_sent:+.3f}")
        m3.metric("Overall Signal", sent_label)

        st.markdown(f'<div class="insight">Overall news sentiment is <strong style="color:{sent_color}">{sent_label}</strong> ({avg_sent:+.3f}). Polarity derived from TextBlob NLP on headline text.</div>', unsafe_allow_html=True)

        titles_short = [e.title[:55] + "…" if len(e.title) > 55 else e.title for e in news[:10]]
        fig_sent = go.Figure(go.Bar(
            x=scores[:10], y=titles_short, orientation="h",
            marker_color=["#26a69a" if s > 0.05 else "#ef5350" if s < -0.05 else "#ffb300" for s in scores[:10]],
        ))
        fig_sent.add_vline(x=0, line_dash="dash", line_color="#7b8ab8")
        fig_sent.update_layout(template="plotly_dark", xaxis_title="Sentiment Polarity",
                                margin=dict(l=0, r=0, t=20, b=0), height=340)
        st.plotly_chart(fig_sent, use_container_width=True)

        filt = st.selectbox("Filter", ["All", "Positive", "Neutral", "Negative"])
        st.markdown("---")
        for entry, score in zip(news[:15], scores):
            if filt == "Positive" and score <= 0.05: continue
            if filt == "Negative" and score >= -0.05: continue
            if filt == "Neutral"  and abs(score) > 0.05: continue
            badge = "🟢 Positive" if score > 0.05 else "🔴 Negative" if score < -0.05 else "🟡 Neutral"
            st.markdown(f"**[{entry.title}]({entry.link})**")
            st.caption(f"{badge}  ·  Score: {score:+.3f}  ·  {entry.get('published', '')}")
            st.markdown(f"{entry.get('summary', '')[:220]}…")
            st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Project Overview")
    st.markdown("""
    An end-to-end **data science pipeline** applied to financial time-series,
    combining statistical modelling, deep learning, stochastic simulation,
    and risk analytics to analyse Amazon stock.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="arch-block">
        <strong>🔢 Data Pipeline</strong><br><br>
        • Live 2-year OHLCV via <code>yfinance</code><br>
        • 12 engineered features: RSI, MACD, Bollinger Bands, MAs, Volatility, Volume ratio<br>
        • MinMax normalisation with no data leakage (fit on train split only)<br>
        • 80/20 chronological train-test split
        </div>
        <div class="arch-block">
        <strong>📐 Linear Regression</strong><br><br>
        • Date-ordinal feature; fits linear trend on last 120 days<br>
        • Analytical 95% prediction interval: widens with √(1 + h/n)<br>
        • Baseline model for comparison
        </div>
        <div class="arch-block">
        <strong>🎲 Monte Carlo (GBM)</strong><br><br>
        • Geometric Brownian Motion with μ and σ from last 252 trading days<br>
        • 500 simulation paths; reports P10/P50/P90 and P(price rises)
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="arch-block">
        <strong>🧠 One-Step LSTM</strong><br><br>
        • Input: 30-step sliding window × 5 features (OHLCV)<br>
        • Architecture: LSTM(64 hidden) → Linear(1)<br>
        • Autoregressive rollout for multi-day forecast<br>
        • 95% CI from training residual std × √h
        </div>
        <div class="arch-block">
        <strong>🔁 Seq2Seq LSTM</strong><br><br>
        • 2-layer encoder reads 30-day input window<br>
        • 2-layer decoder unrolls 7-step output directly<br>
        • Avoids error accumulation of autoregressive approach
        </div>
        <div class="arch-block">
        <strong>🌲 Random Forest (Explainability)</strong><br><br>
        • 200 trees on 10 technical indicator features<br>
        • Feature importance via mean decrease in impurity<br>
        • Reveals which signals drive next-day price prediction
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Tech Stack")
    st.markdown("""
    | Layer | Tools |
    |---|---|
    | Data ingestion | `yfinance`, `feedparser` |
    | Feature engineering | `ta`, `pandas`, `numpy` |
    | Deep learning | `PyTorch` — LSTM, Seq2Seq |
    | Classical / ensemble ML | `scikit-learn` — LR, Random Forest |
    | Statistics | `scipy` — normality tests, distribution fitting |
    | NLP / Sentiment | `TextBlob` |
    | Visualisation | `Plotly` |
    | App framework | `Streamlit` |
    | Deployment | Streamlit Cloud (free, serverless) |
    """)

    st.subheader("Connect")
    st.markdown("""
    👨‍💻 [Karthik Mulugu](https://www.linkedin.com/in/karthikmulugu/)  ·
    🐙 [GitHub Repo](https://github.com/Karthik0809/Amazon-Stock-Dashboard)
    """)
