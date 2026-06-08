# AMZN Stock Forecasting & Analytics Dashboard

> An end-to-end machine learning pipeline that benchmarks four forecasting approaches on Amazon (AMZN) price data, deployed as an interactive dashboard.

<p align="center">
  <a href="https://amazonstock-dashboard.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/PyTorch-LSTM%20%7C%20Seq2Seq-EE4C2C?logo=pytorch&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white" />
</p>

**[▸ Live App](https://amazonstock-dashboard.streamlit.app/)**

---

## Problem Statement

Stock price forecasting is a hard time-series problem — prices are non-stationary, noisy, and influenced by factors outside any feature set. The goal of this project was not to "beat the market" but to rigorously compare forecasting approaches, understand where each breaks down, and build production-quality tooling around them.

**Core question:** On a real equity time series, does a deep learning model (LSTM, Seq2Seq) actually outperform a well-tuned classical baseline (Linear Regression, XGBoost), and under what conditions?

---

## Approach & Methodology

### Data
- Source: Yahoo Finance via `yfinance`, with a local CSV fallback to handle API rate limits on cloud deployments
- ~5 years of daily OHLCV data for AMZN
- Train/test split: strictly chronological 80/20 — no shuffling, no future data leakage

### Feature Engineering
Built 10+ technical features from raw OHLCV:

| Feature | Description |
|---|---|
| RSI | Momentum oscillator — identifies overbought/oversold conditions |
| MACD / Signal | Trend-following momentum indicator |
| Bollinger Band Width | Measures volatility relative to recent price |
| MA Ratio (20/50) | Short vs medium-term trend relationship |
| Volume Ratio | Current vs 20-day average volume |
| 30D Rolling Volatility | Annualised standard deviation of returns |
| Price vs MA30/MA90 | Distance from moving averages (mean-reversion signal) |

### Models Compared

| Model | Type | Key Design Choice |
|---|---|---|
| Linear Regression | Baseline | Analytical confidence intervals via prediction std |
| One-Step LSTM | Deep Learning | PyTorch, trained on 5-feature sequences; residual-based confidence bands |
| Seq2Seq LSTM | Deep Learning | Encoder-Decoder architecture for direct multi-step prediction (avoids error accumulation) |
| XGBoost | Ensemble | RandomizedSearchCV hyperparameter tuning; walk-forward validation |

### Evaluation
- Metrics: RMSE, MAPE, R²
- Walk-forward validation on XGBoost to simulate real deployment conditions (rolling retrain on expanding window)
- Visual actual vs. predicted overlays with strict train/test shading

---

## Key Findings

- **Linear Regression is a surprisingly strong baseline** — on trending data, a simple regression on recent prices captures the direction well and is hard to beat on RMSE alone
- **LSTM models overfit on short sequences** — with ~5 years of daily data, the deep learning models have relatively little data to generalise; they tend to lag price turns
- **Seq2Seq outperforms one-step LSTM on multi-step horizons** — direct multi-step prediction avoids the compounding error of autoregressive rollout
- **XGBoost benefits most from feature engineering** — the 10+ technical features give it signal the regression models can't use, and it captures non-linear interactions well
- **No model reliably predicts turning points** — all models perform worse at market inflection points, consistent with the efficient market hypothesis

---

## Beyond Forecasting

The dashboard also includes supporting analyses that provide context for the forecasts:

- **Risk Analytics** — Sharpe, Sortino, Max Drawdown, VaR 95%/99%, CVaR; return distribution normality testing (Shapiro-Wilk); ACF for autocorrelation structure
- **Volatility Regime Detection** — 30D vs 90D rolling vol with high-vol period shading; helps contextualise model performance across market regimes
- **Anomaly Detection** — Z-score rolling window (20-day, ±2.5σ) flags statistically unusual price events
- **Signal Scorecard** — 7-indicator consensus (RSI, MACD, Bollinger Bands, MA crossover, Volume, Volatility) gives a quick directional read
- **Monte Carlo Simulation** — 500 GBM paths with P10/P50/P90 fan for probabilistic price range estimation
- **News Sentiment** — Live Yahoo Finance RSS feed with FinBERT (transformer-based) sentiment scoring per article
- **Peer Comparison** — AMZN vs MSFT, GOOGL, META, AAPL, SPY normalised performance and correlation matrix

---

## Architecture

```
Amazon-Stock-Dashboard/
├── app.py                  # Streamlit app — all tabs, UI, caching, ML inference
├── model.py                # One-Step LSTM definition (PyTorch)
├── seq2seq_lstm.py         # Seq2Seq Encoder-Decoder architecture
├── amazon_lstm_model.pth   # Pre-trained One-Step LSTM weights
├── seq2seq_lstm.pth        # Pre-trained Seq2Seq weights
├── amazon_stock.csv        # Historical AMZN data (rate-limit fallback)
└── requirements.txt
```

**Data flow:**
```
yfinance API  ──▶  Feature Engineering  ──▶  MinMaxScaler  ──▶  LSTM / LR / XGBoost
     │                                                                    │
     └── (rate-limit fallback) amazon_stock.csv           Risk · Charts · Dashboard
```

---

## Engineering Notes

A few non-obvious decisions made during development:

- **Chronological train/test split** — shuffling would leak future prices into training; all splits preserve time ordering
- **LSTM autoregressive rollout fix** — multi-step rollout updates the Close feature at the correct index; an off-by-one bug caused trend inversions in early versions
- **Volatility regime batching** — initial implementation called `add_vrect()` ~500 times (once per trading day), blocking all subsequent tab renders for 10–30s; fixed by merging consecutive high-vol days into single rectangles
- **Streamlit caching strategy** — heavy functions (LSTM inference, Monte Carlo, risk metrics) are `@st.cache_data(ttl=3600)` to avoid re-running on every tab click
- **XGBoost memory constraint** — `n_jobs=1` and reduced CV folds to fit within Streamlit Cloud's 1GB RAM limit

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

## Tech Stack

| Layer | Technology |
|---|---|
| App framework | Streamlit 1.58, custom CSS dark theme |
| Charts | Plotly (fully interactive) |
| Deep Learning | PyTorch — LSTM, Seq2Seq Encoder-Decoder |
| Classical ML | scikit-learn — Linear Regression, Random Forest; XGBoost |
| NLP | HuggingFace Transformers (FinBERT), feedparser |
| Statistics | SciPy, StatsModels |
| Market Data | yfinance + CSV fallback |
| Deployment | Streamlit Cloud (free tier) |

---

## Connect

**Karthik Mulugu**
📧 karthikmulugu14@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/karthikmulugu/)
🐙 [GitHub](https://github.com/Karthik0809)

---

> ⚠️ **Disclaimer**: Educational project only. Not financial advice.
