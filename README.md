# AMZN · Intelligence Terminal

> A production-grade, ML-powered stock analytics dashboard for **Amazon (AMZN)** — built with PyTorch, Streamlit, and quantitative finance techniques used in real trading desks.

<p align="center">
  <a href="https://amazonstock-dashboard.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-LSTM%20%7C%20Seq2Seq-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  &nbsp;
  <img src="https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit Cloud" />
</p>

---

## 🔗 Live App

**[▸ Launch Dashboard](https://amazonstock-dashboard.streamlit.app/)**

---

## What This Project Demonstrates

This dashboard showcases a full data-science pipeline — from raw market data ingestion through feature engineering, deep learning forecasting, quantitative risk modelling, and an interactive terminal-style UI.

Designed to be **resume-worthy for a Data Scientist / Quantitative Analyst role**, covering:

| Domain | Techniques Used |
|---|---|
| **ML / Deep Learning** | PyTorch LSTM, Seq2Seq Encoder-Decoder, Linear Regression |
| **Time-Series Forecasting** | Autoregressive multi-step prediction, 80/20 chronological split |
| **Quantitative Risk** | Sharpe, Sortino, Calmar ratio, VaR 95%/99%, CVaR, Max Drawdown |
| **Statistical Analysis** | Shapiro-Wilk normality test, ACF (EMH testing), skewness, kurtosis |
| **Simulation** | Monte Carlo GBM — 500 paths, P10/P50/P90 percentile fan |
| **Feature Engineering** | RSI, MACD, Bollinger Bands, EMA, ATR, volume signals |
| **NLP** | News sentiment analysis via TextBlob + Yahoo RSS feed |
| **Anomaly Detection** | Z-score rolling window flagging of unusual price events |
| **Signal Engineering** | 7-indicator scorecard, pivot points, volatility regime detection |
| **Portfolio Analytics** | P&L simulator, position sizing, projected value at forecast horizon |
| **Data Visualisation** | Plotly interactive charts, custom dark terminal theme |

---

## Features

### 🔮 Forecast Tab
- **Linear Regression** baseline with analytical confidence intervals
- **One-Step LSTM** — PyTorch LSTM trained on 5 features (OHLCV), residual-based confidence bands
- **Seq2Seq LSTM** — Encoder-Decoder architecture for direct multi-step horizon forecasting
- Adjustable forecast horizon (1–30 days) and lookback window

### 📉 Technical Analysis Tab
- Candlestick chart with 20/50/200-day moving averages + Bollinger Bands
- RSI with overbought/oversold zones · MACD histogram with crossover signals
- Volume bars with colour coding

### 📊 Risk Analytics Tab
- Full scorecard: Sharpe, Sortino, Max Drawdown, Calmar, VaR 95%, CVaR
- Return distribution with Shapiro-Wilk normality test, skewness & kurtosis
- ACF plot (Efficient Market Hypothesis test)
- **Rolling 30-Day VaR** — shows how tail risk evolves over time
- **Volatility Regime Detection** — 30D vs 90D rolling vol with regime shading (high/low vol periods)
- Random Forest feature importance (200 trees, mean decrease in impurity)

### 🌐 Market Context Tab
- Peer comparison: AMZN vs MSFT, GOOGL, META, AAPL, SPY (normalised to 100)
- Rolling 60-day correlation heatmap · Beta vs SPY
- Monthly returns heatmap (year × month grid)

### 🤖 Model Performance Tab
- RMSE and MAPE across all three models with visual bar comparison
- Actual vs Predicted overlay — strict train/test shading, no data leakage

### 📡 Signals & Anomalies Tab *(new)*
- **Signal Scorecard** — consolidated bull/bear verdict across 7 indicators (RSI, MACD, Bollinger Bands, MA crossover, Price vs MA90, Volume, Volatility)
- **Support & Resistance** — classic floor-trader pivot points (PP, R1/R2/R3, S1/S2/S3) plotted on last 60 days of price
- **Anomaly Detection** — Z-score on 20-day rolling window flags statistically unusual price moves (±2.5σ), plotted as triangles on price chart with a sortable event table
- **Portfolio P&L Simulator** — enter shares held + avg cost basis, see unrealised P&L and projected value at any forecast horizon

### 📰 News & Sentiment Tab
- Live Yahoo Finance RSS feed filtered for AMZN headlines
- Per-article TextBlob sentiment scores + aggregated chart

---

## Architecture

```
Amazon-Stock-Dashboard/
├── app.py                  # Streamlit app — all tabs, UI, ML inference
├── model.py                # One-Step LSTM definition (PyTorch)
├── seq2seq_lstm.py         # Seq2Seq Encoder-Decoder + make_multi_sequences()
├── amazon_lstm_model.pth   # Trained One-Step LSTM weights
├── seq2seq_lstm.pth        # Trained Seq2Seq weights
├── amazon_stock.csv        # Historical AMZN data (rate-limit fallback)
└── requirements.txt        # Python dependencies
```

**Data flow:**
```
yfinance API  ──▶  Feature Engineering  ──▶  MinMaxScaler  ──▶  LSTM / LR
     │                                                                │
     └── (rate-limit fallback) amazon_stock.csv              Risk · Charts · UI
```

---

## Local Setup

```bash
git clone https://github.com/Karthik0809/Amazon-Stock-Dashboard.git
cd Amazon-Stock-Dashboard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

App runs at `http://localhost:8501`

---

## Deploy on Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **New app**
2. Select repo `Karthik0809/Amazon-Stock-Dashboard` · branch `main` · file `app.py`
3. Click **Deploy** — no servers, no credit card, no config needed

> **Note**: yfinance is sometimes rate-limited on Streamlit Cloud's shared IPs. The app automatically falls back to the bundled `amazon_stock.csv` so it never crashes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.31+, custom CSS dark terminal theme |
| Charts | Plotly (fully interactive) |
| Deep Learning | PyTorch 2.x — LSTM, Seq2Seq |
| Classical ML | scikit-learn — Linear Regression, Random Forest |
| Market Data | yfinance + CSV fallback |
| Technical Indicators | `ta` library (RSI, MACD, Bollinger Bands, ATR) |
| NLP | TextBlob + feedparser |
| Statistics | SciPy, StatsModels, NumPy, Pandas |

---

## Key Engineering Decisions

- **No data leakage** — train/test split is strictly chronological (80/20); no future data seen during training
- **LSTM autoregressive fix** — multi-step rollout updates feature index 3 (Close), not index 4 (Volume), fixing a trend-direction inversion bug
- **Length-safe masking** — prediction arrays clipped before masking to handle variable yfinance response sizes on cloud deployments
- **CSS via `st.html()`** — style injection bypasses Streamlit Cloud's CSP stripping of `<style>` tags in `st.markdown()`
- **Graceful degradation** — all network calls (peers, news, live data) fail silently to cached or local state

---

## Connect

**Karthik Mulugu** — Data Scientist  
📧 karthikmulugu14@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/karthikmulugu/)  
🐙 [GitHub](https://github.com/Karthik0809)

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

---

> ⚠️ **Disclaimer**: Educational project only. Not financial advice. Past performance does not guarantee future results.
