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
from seq2seq_lstm import Seq2SeqLSTM, make_multi_sequences

SEQ_LEN = 30

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

@st.cache_data(ttl=1800)
def load_news():
    feed = feedparser.parse("https://finance.yahoo.com/rss/headline?s=AMZN")
    return [e for e in feed.entries if "amazon" in e.title.lower() or "amzn" in e.title.lower()]

df = load_data()
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

# ── Model definition ──────────────────────────────────────────────────────────
class OneStepLSTM(nn.Module):
    def __init__(self, inp_size=5):
        super().__init__()
        self.lstm = nn.LSTM(inp_size, 64, batch_first=True)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Forecast functions ────────────────────────────────────────────────────────
def forecast_lr(df, n_days):
    data = df[-120:].copy()
    data["ord"] = data.index.map(pd.Timestamp.toordinal)
    X, y = data["ord"].values.reshape(-1, 1), data["Close"].values
    mdl = LinearRegression().fit(X, y)
    resid_std = (y - mdl.predict(X)).std()
    future = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=n_days)
    Xf = future.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    mu = mdl.predict(Xf)
    # CI widens with horizon distance
    ci = resid_std * np.sqrt(1 + np.arange(1, n_days + 1) / len(X))
    return (
        pd.Series(mu, index=future),
        pd.Series(mu - 1.96 * ci, index=future),
        pd.Series(mu + 1.96 * ci, index=future),
    )

def forecast_lstm(df, seq_len, n_days):
    feats  = df[["Open", "High", "Low", "Close", "Volume"]].values
    scaler = MinMaxScaler().fit(feats)
    norm   = scaler.transform(feats)

    mdl = OneStepLSTM(inp_size=5).cpu()
    mdl.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu", weights_only=False))
    mdl.eval()

    # Residual std from in-sample predictions (used for CI)
    X_h, y_h = make_multi_sequences(norm, seq_len=seq_len, horizon=1)
    with torch.no_grad():
        ph = mdl(X_h.float()).numpy().flatten()
    dummy4  = np.zeros((len(ph), 4))
    inv_ph  = scaler.inverse_transform(np.hstack([dummy4, ph.reshape(-1, 1)]))[:, -1]
    inv_yh  = scaler.inverse_transform(np.hstack([dummy4, y_h.numpy().reshape(-1, 1)]))[:, -1]
    resid_std = (inv_yh - inv_ph).std()

    # Autoregressive multi-step forecast
    # FIX: update Close (index 3), not Volume (index 4)
    preds, seq = [], norm[-seq_len:].copy()
    for _ in range(n_days):
        with torch.no_grad():
            p = mdl(torch.tensor(seq).unsqueeze(0).float()).item()
        preds.append(p)
        row    = seq[-1].copy()
        row[3] = p  # Close is feature index 3
        seq    = np.vstack([seq[1:], row])

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

def backtest_rsi(df):
    sig  = (df["RSI"] < 30).astype(int)
    rets = df["Close"].pct_change().shift(-1)
    return (1 + (sig * rets).fillna(0)).cumprod()

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Amazon (AMZN) Stock Intelligence Dashboard")
st.caption(
    "Live market data · ML price forecasting · Technical analysis · News sentiment analysis  "
    "· ⚠️ Educational project only — not financial advice"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
cur     = df["Close"].iloc[-1]
prev    = df["Close"].iloc[-2]
day_chg = (cur - prev) / prev * 100
high52  = df["Close"].rolling(252).max().iloc[-1]
low52   = df["Close"].rolling(252).min().iloc[-1]
ytd_start = df.loc[df.index >= f"{df.index[-1].year}-01-01", "Close"]
ytd_ret = (cur / ytd_start.iloc[0] - 1) * 100 if len(ytd_start) > 0 else 0.0
vol30   = df["Volatility"].iloc[-1]
rsi_now = df["RSI"].iloc[-1]

kpis = [
    ("Current Price",      f"${cur:.2f}",   f"{'▲' if day_chg >= 0 else '▼'} {day_chg:+.2f}% today",   day_chg >= 0),
    ("52-Week High",       f"${high52:.2f}", f"{(cur/high52-1)*100:.1f}% from high",                      cur >= high52 * 0.98),
    ("52-Week Low",        f"${low52:.2f}",  f"{(cur/low52-1)*100:.1f}% from low",                        True),
    ("YTD Return",         f"{ytd_ret:+.1f}%", "year-to-date",                                            ytd_ret >= 0),
    ("30-Day Volatility",  f"{vol30:.1f}%",  "annualised",                                                 vol30 < 30),
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
tab_fc, tab_tech, tab_models, tab_news, tab_about = st.tabs([
    "🔮 Forecast", "📉 Technical Analysis", "🤖 Model Performance", "📰 News & Sentiment", "ℹ️ About"
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
            st.info("Seq2Seq is trained for 7-day output — setting horizon to 7.")
            n_days = 7
        show_ci      = st.checkbox("Show 95% confidence band", value=True)
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
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist, name="Historical Close",
            line=dict(color="#5c6bc0", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=mu.index, y=mu, name=f"{model_opt} Forecast",
            line=dict(color="#26a69a", width=2.5, dash="dash")
        ))
        if show_ci and lo is not None:
            fig.add_trace(go.Scatter(
                x=list(mu.index) + list(mu.index[::-1]),
                y=list(hi) + list(lo[::-1]),
                fill="toself", fillcolor="rgba(38,166,154,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            ))
        fig.update_layout(
            template="plotly_dark", hovermode="x unified",
            xaxis_title="Date", yaxis_title="Price (USD)",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
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

    with st.expander("View forecast table"):
        fdf = mu.reset_index().rename(columns={"index": "Date", 0: "Forecast ($)"})
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
    period_opt = st.selectbox(
        "Time range",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
        index=2,
        key="tech_period"
    )
    period_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "2 Years": 504}
    n = period_map[period_opt]
    dv = df.iloc[-n:]

    # Price + MAs + Bollinger Bands
    st.subheader("Price & Moving Averages")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["Close"], name="Close", line=dict(color="#5c6bc0", width=2)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["MA7"],  name="MA 7",  line=dict(color="#ffb300", width=1.2)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["MA30"], name="MA 30", line=dict(color="#ef5350", width=1.2)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["MA90"], name="MA 90", line=dict(color="#ab47bc", width=1.2)))
    fig_price.add_trace(go.Scatter(
        x=list(dv.index) + list(dv.index[::-1]),
        y=list(dv["BB_upper"]) + list(dv["BB_lower"][::-1]),
        fill="toself", fillcolor="rgba(92,107,192,0.08)",
        line=dict(color="rgba(255,255,255,0)"), name="Bollinger Bands"
    ))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["BB_upper"], name="BB Upper", line=dict(color="#5c6bc0", dash="dot", width=1)))
    fig_price.add_trace(go.Scatter(x=dv.index, y=dv["BB_lower"], name="BB Lower", line=dict(color="#5c6bc0", dash="dot", width=1)))
    fig_price.update_layout(template="plotly_dark", hovermode="x unified",
                             xaxis_title="Date", yaxis_title="Price (USD)",
                             margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.subheader("RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI", line=dict(color="#26a69a", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold (30)")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.07)", line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.07)", line_width=0)
        fig_rsi.update_layout(template="plotly_dark", yaxis=dict(range=[0, 100]),
                               margin=dict(l=0, r=0, t=20, b=0), hovermode="x unified")
        st.plotly_chart(fig_rsi, use_container_width=True)
        rsi_val = dv["RSI"].iloc[-1]
        if rsi_val > 70:
            st.markdown('<div class="insight">RSI > 70 — stock may be <strong style="color:#ef5350">overbought</strong>. Watch for potential pullback.</div>', unsafe_allow_html=True)
        elif rsi_val < 30:
            st.markdown('<div class="insight">RSI < 30 — stock may be <strong style="color:#26a69a">oversold</strong>. Potential buying opportunity.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight">RSI = {rsi_val:.1f} — neutral zone. No extreme signal.</div>', unsafe_allow_html=True)

    with col_macd:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=dv.index, y=dv["MACD"],        name="MACD",   line=dict(color="#5c6bc0", width=2)))
        fig_macd.add_trace(go.Scatter(x=dv.index, y=dv["MACD_Signal"], name="Signal", line=dict(color="#ffb300", width=1.5)))
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in dv["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=dv.index, y=dv["MACD_Hist"], name="Histogram", marker_color=colors, opacity=0.7))
        fig_macd.update_layout(template="plotly_dark", hovermode="x unified",
                                margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_macd, use_container_width=True)
        macd_val = dv["MACD"].iloc[-1]
        sig_val  = dv["MACD_Signal"].iloc[-1]
        if macd_val > sig_val:
            st.markdown('<div class="insight">MACD above Signal — <strong style="color:#26a69a">bullish momentum</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight">MACD below Signal — <strong style="color:#ef5350">bearish momentum</strong>.</div>', unsafe_allow_html=True)

    # Volume
    st.subheader("Volume")
    avg_vol = dv["Volume"].mean()
    vol_colors = ["#26a69a" if v >= avg_vol else "#ef5350" for v in dv["Volume"]]
    fig_vol = go.Figure(go.Bar(x=dv.index, y=dv["Volume"], marker_color=vol_colors, name="Volume"))
    fig_vol.add_hline(y=avg_vol, line_dash="dash", line_color="#ffb300", annotation_text=f"Avg {avg_vol/1e6:.0f}M")
    fig_vol.update_layout(template="plotly_dark", yaxis_title="Shares", hovermode="x unified",
                           margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_vol, use_container_width=True)

    # RSI Backtest
    st.subheader("RSI < 30 Strategy — Backtest")
    bt = backtest_rsi(dv)
    buy_hold = (1 + dv["Returns"].fillna(0)).cumprod()
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt,        name="RSI Strategy",  line=dict(color="#26a69a", width=2)))
    fig_bt.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold, name="Buy & Hold", line=dict(color="#5c6bc0", width=2, dash="dot")))
    fig_bt.update_layout(template="plotly_dark", yaxis_title="Cumulative Return",
                          hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_bt, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("Out-of-Sample Evaluation  (80 / 20 train-test split)")
    st.caption("Models are evaluated on the held-out 20% test set — data they never saw during training.")

    # ── Build test-set predictions ────────────────────────────────────────────
    split    = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    # Linear Regression
    df_all = df.copy()
    df_all["ord"] = df_all.index.map(pd.Timestamp.toordinal)
    lr_mdl = LinearRegression().fit(
        df_all["ord"].iloc[:split].values.reshape(-1, 1),
        df_all["Close"].iloc[:split].values
    )
    lr_pred_test = lr_mdl.predict(df_all["ord"].iloc[split:].values.reshape(-1, 1))

    # LSTM
    feats  = df[["Open", "High", "Low", "Close", "Volume"]].values
    scaler_lstm = MinMaxScaler().fit(feats[:split])
    norm_lstm   = scaler_lstm.transform(feats)
    lstm_mdl = OneStepLSTM(inp_size=5).cpu()
    lstm_mdl.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu", weights_only=False))
    lstm_mdl.eval()
    X_lstm, y_lstm = make_multi_sequences(norm_lstm, seq_len=SEQ_LEN, horizon=1)
    with torch.no_grad():
        ph_lstm = lstm_mdl(X_lstm.float()).numpy().flatten()
    dummy4    = np.zeros((len(ph_lstm), 4))
    inv_lstm  = scaler_lstm.inverse_transform(np.hstack([dummy4, ph_lstm.reshape(-1,1)]))[:,-1]
    inv_y_lst = scaler_lstm.inverse_transform(np.hstack([dummy4, y_lstm.numpy().reshape(-1,1)]))[:,-1]
    idx_lstm  = df.index[SEQ_LEN : SEQ_LEN + len(inv_lstm)]
    test_mask_lstm = (idx_lstm >= df_test.index[0]).values

    # Seq2Seq
    closes_all  = df[["Close"]].values
    scaler_seq  = MinMaxScaler().fit(closes_all[:split])
    norm_seq    = scaler_seq.transform(closes_all)
    X_seq, y_seq = make_multi_sequences(norm_seq, seq_len=SEQ_LEN, horizon=1)
    seq_mdl = Seq2SeqLSTM(input_dim=1).cpu()
    seq_mdl.load_state_dict(torch.load("seq2seq_lstm.pth", map_location="cpu", weights_only=False))
    seq_mdl.eval()
    with torch.no_grad():
        ph_seq = seq_mdl(X_seq.float()).numpy().flatten()
    inv_seq   = scaler_seq.inverse_transform(ph_seq.reshape(-1,1)).flatten()
    inv_y_seq = scaler_seq.inverse_transform(y_seq.numpy().reshape(-1,1)).flatten()
    idx_seq   = df.index[SEQ_LEN : SEQ_LEN + len(inv_seq)]
    test_mask_seq = (idx_seq >= df_test.index[0]).values

    # ── Metrics ───────────────────────────────────────────────────────────────
    def metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae  = np.mean(np.abs(actual - pred))
        mape_v = mape(actual, pred)
        r2   = r2_score(actual, pred)
        return rmse, mae, mape_v, r2

    act_lr  = df_test["Close"].values
    act_ls  = inv_y_lst[test_mask_lstm]
    act_sq  = inv_y_seq[test_mask_seq]
    pr_lr   = lr_pred_test
    pr_ls   = inv_lstm[test_mask_lstm]
    pr_sq   = inv_seq[test_mask_seq]

    min_lr = min(len(act_lr), len(pr_lr))
    min_ls = min(len(act_ls), len(pr_ls))
    min_sq = min(len(act_sq), len(pr_sq))

    rows = []
    for name, act, pred, n_ in [
        ("Linear Regression", act_lr[:min_lr], pr_lr[:min_lr], min_lr),
        ("One-Step LSTM",     act_ls[:min_ls], pr_ls[:min_ls], min_ls),
        ("Seq2Seq LSTM",      act_sq[:min_sq], pr_sq[:min_sq], min_sq),
    ]:
        rm, ma, mp, r2 = metrics(act, pred)
        rows.append({"Model": name, "RMSE ($)": rm, "MAE ($)": ma, "MAPE (%)": mp, "R²": r2, "Test samples": n_})

    metrics_df = pd.DataFrame(rows)
    best_rmse  = metrics_df["RMSE ($)"].idxmin()

    st.dataframe(
        metrics_df.style
            .format({"RMSE ($)": "{:.2f}", "MAE ($)": "{:.2f}", "MAPE (%)": "{:.2f}", "R²": "{:.4f}"})
            .highlight_min(subset=["RMSE ($)", "MAE ($)", "MAPE (%)"], color="#1e3a2f")
            .highlight_max(subset=["R²"], color="#1e3a2f"),
        use_container_width=True,
        hide_index=True,
    )

    best_model = metrics_df.loc[best_rmse, "Model"]
    st.markdown(f'<div class="insight">🏆 <strong>{best_model}</strong> achieves the lowest RMSE on the held-out test set.</div>', unsafe_allow_html=True)

    # ── Visual comparison on test window ─────────────────────────────────────
    st.subheader("Predicted vs Actual — Test Period")
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

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.subheader("Residuals — Test Period")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df_test.index[:min_lr],
                                  y=(act_lr[:min_lr] - pr_lr[:min_lr]),
                                  name="LR Residuals", mode="lines+markers",
                                  line=dict(color="#ffb300")))
    fig_res.add_trace(go.Scatter(x=idx_lstm[test_mask_lstm][:min_ls],
                                  y=(act_ls[:min_ls] - pr_ls[:min_ls]),
                                  name="LSTM Residuals", mode="lines+markers",
                                  line=dict(color="#26a69a")))
    fig_res.add_hline(y=0, line_dash="dash", line_color="#7b8ab8")
    fig_res.update_layout(template="plotly_dark", hovermode="x unified",
                           yaxis_title="Residual ($)", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_res, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NEWS & SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_news:
    if not news:
        st.info("No recent AMZN news articles found.")
    else:
        # Aggregate sentiment
        scores = [TextBlob(e.title).sentiment.polarity for e in news[:15]]
        avg_sent = np.mean(scores)
        sent_label = "Positive 🟢" if avg_sent > 0.05 else "Negative 🔴" if avg_sent < -0.05 else "Neutral 🟡"
        sent_color = "#26a69a" if avg_sent > 0.05 else "#ef5350" if avg_sent < -0.05 else "#ffb300"

        m1, m2, m3 = st.columns(3)
        m1.metric("Articles Analysed", len(news[:15]))
        m2.metric("Avg Sentiment Score", f"{avg_sent:+.3f}")
        m3.metric("Overall Signal", sent_label)

        st.markdown(f'<div class="insight">Overall news sentiment is <strong style="color:{sent_color}">{sent_label}</strong> (score {avg_sent:+.3f}). Sentiment is derived from headline text via TextBlob polarity analysis.</div>', unsafe_allow_html=True)

        # Sentiment bar chart
        titles_short = [e.title[:50] + "…" for e in news[:10]]
        fig_sent = go.Figure(go.Bar(
            x=scores[:10], y=titles_short, orientation="h",
            marker_color=["#26a69a" if s > 0 else "#ef5350" if s < 0 else "#ffb300" for s in scores[:10]],
        ))
        fig_sent.add_vline(x=0, line_dash="dash", line_color="#7b8ab8")
        fig_sent.update_layout(template="plotly_dark", xaxis_title="Sentiment Polarity",
                                margin=dict(l=0, r=0, t=20, b=0), height=320)
        st.plotly_chart(fig_sent, use_container_width=True)

        # Filter
        filt = st.selectbox("Filter by sentiment", ["All", "Positive", "Neutral", "Negative"])
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
# TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Project Overview")
    st.markdown("""
    This dashboard demonstrates an end-to-end **data science and MLOps pipeline** applied to
    financial time-series data. It combines classical statistical models with deep learning
    architectures to forecast stock prices and extract actionable signals.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="arch-block">
        <strong>🔢 Data Pipeline</strong><br><br>
        • Live OHLCV data via <code>yfinance</code> (2-year window)<br>
        • Feature engineering: RSI, MACD, Bollinger Bands, Moving Averages, Annualised Volatility<br>
        • MinMax normalisation per-feature before model input<br>
        • 80/20 train-test split with no data leakage
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="arch-block">
        <strong>📐 Linear Regression Baseline</strong><br><br>
        • Fits on last 120 trading days of Close prices<br>
        • Date ordinal as sole feature — captures linear trend<br>
        • 95% prediction interval widens with √(1 + h/n) factor<br>
        • RMSE / MAE / MAPE / R² evaluated on held-out test set
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="arch-block">
        <strong>🧠 One-Step LSTM</strong><br><br>
        • Input: 30-step sliding window × 5 features (OHLCV)<br>
        • Architecture: LSTM(64 hidden) → Linear(1)<br>
        • Trained to predict next-day Close (normalised)<br>
        • Multi-step forecasting via autoregressive rollout<br>
        • CI estimated from in-sample residual std × √h
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="arch-block">
        <strong>🔁 Seq2Seq LSTM</strong><br><br>
        • Encoder: 2-layer LSTM(64) reads 30-day input sequence<br>
        • Decoder: 2-layer LSTM(64) unrolls 7-step prediction<br>
        • Direct multi-horizon output — no error accumulation<br>
        • Trained end-to-end with MSE loss on 7-day target windows
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Tech Stack")
    st.markdown("""
    | Layer | Tools |
    |---|---|
    | Data ingestion | `yfinance`, `feedparser` |
    | Feature engineering | `ta` (technical analysis), `pandas`, `numpy` |
    | Deep learning | `PyTorch` — LSTM, Seq2Seq |
    | Classical ML | `scikit-learn` — Linear Regression, MinMaxScaler |
    | NLP / Sentiment | `TextBlob` |
    | Visualisation | `Plotly` |
    | App framework | `Streamlit` |
    | Deployment | Streamlit Cloud (free tier) |
    """)

    st.subheader("Connect")
    st.markdown("""
    👨‍💻 [Karthik Mulugu](https://www.linkedin.com/in/karthikmulugu/) ·
    🐙 [GitHub Repo](https://github.com/Karthik0809/Amazon-Stock-Dashboard)
    """)
