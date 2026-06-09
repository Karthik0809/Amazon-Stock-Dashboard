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
- **LSTM models are limited by data size** — ~5 years of daily OHLCV yields ~1,250 training sequences. LSTMs in NLP typically see millions. The models capture trend direction but lag at turning points; this is expected, not a bug
- **Seq2Seq outperforms one-step LSTM on multi-step horizons** — direct 7-step prediction avoids the compounding error of autoregressive rollout (~15% MAPE improvement on the test set)
- **XGBoost benefits most from feature engineering** — the 11 technical features give it signal the regression models can't use, and it captures non-linear interactions; consistently beats LR across AMZN, MSFT, and GOOGL in walk-forward validation
- **No model reliably predicts turning points** — all models perform worse at market inflection points, consistent with the efficient market hypothesis on large-cap stocks

### Why LSTM Underperforms Classical Models Here (and That's OK)

This is a known result in the quantitative finance literature. Daily OHLCV data for a single stock is:
- **Low signal-to-noise**: prices are close to a random walk (EMH)
- **Non-stationary**: volatility regimes shift; models trained on one regime struggle on another
- **Small sample for DL**: ~1,250 sequences vs millions needed to unlock LSTM capacity

A deep learning approach would be more competitive with: tick-level data, cross-sectional features (hundreds of stocks simultaneously), or alternative data (earnings transcripts, order flow). The project documents this honestly rather than cherry-picking favourable results.

### What Was Tried That Didn't Work

| Experiment | Result |
|---|---|
| 2-layer LSTM + dropout 0.3 | Higher val loss — too much regularisation for this data size |
| Bidirectional LSTM | Marginal improvement, 2× parameters, not worth RAM cost |
| Adding FinBERT sentiment as LSTM features | No improvement — daily news scores too noisy |
| Autoregressive 7-day rollout | ~15% worse MAPE vs Seq2Seq — error accumulates |
| Global MinMaxScaler (not train-only) | Data leakage — inflated test metrics; fixed to train-split-only scaling |
| XGBoost 30 CV iters on Streamlit Cloud | OOM crash on 1GB RAM; reduced to 5 iters × 3 folds |

---

## Beyond Forecasting

The dashboard also includes supporting analyses that provide context for the forecasts:

- **Risk Analytics** — Sharpe, Sortino, Max Drawdown, VaR 95%/99%, CVaR; return distribution normality testing (Shapiro-Wilk); ACF for autocorrelation structure
- **Volatility Regime Detection** — 30D vs 90D rolling vol with high-vol period shading; helps contextualise model performance across market regimes
- **Anomaly Detection** — Z-score rolling window (20-day, ±2.5σ) flags statistically unusual price events
- **Signal Backtesting** — each technical signal (RSI, MACD, Volume) backtested with t-test p-values and win rates against the unconditional mean return
- **Feature Correlation** — Pearson correlation of all indicators vs next-day returns; honestly shows near-zero correlations consistent with EMH
- **News Sentiment** — Live Yahoo Finance RSS feed with FinBERT (transformer-based) sentiment scoring per article
- **Peer Comparison** — AMZN vs MSFT, GOOGL, META, AAPL, SPY normalised performance and correlation matrix

---

## Reproducibility

The full training pipeline is documented in [`training_notebook.ipynb`](training_notebook.ipynb):
- Data loading and EDA (return distribution, volatility regimes)
- Feature engineering with correlation analysis
- One-Step LSTM training with loss curves and early stopping
- Seq2Seq LSTM training with loss curves
- Multi-ticker walk-forward validation (AMZN, MSFT, GOOGL)
- Explicit table of experiments that were tried and discarded

To retrain models locally:
```bash
jupyter notebook training_notebook.ipynb
```

## Architecture

```
Amazon-Stock-Dashboard/
├── app.py                      # Streamlit app — all tabs, UI, caching, ML inference
├── training_notebook.ipynb     # Full training pipeline with loss curves & experiments
├── model.py                    # One-Step LSTM definition (PyTorch)
├── seq2seq_lstm.py             # Seq2Seq Encoder-Decoder architecture
├── amazon_lstm_model.pth       # Pre-trained One-Step LSTM weights
├── seq2seq_lstm.pth            # Pre-trained Seq2Seq weights
├── amazon_stock.csv            # Historical AMZN data (auto-refreshed daily via CI)
├── .github/workflows/          # GitHub Action: refresh CSV after market close daily
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
| App framework | Streamlit 1.58, custom CSS light theme |
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
