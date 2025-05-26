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
from sklearn.metrics import mean_squared_error
from seq2seq_lstm import Seq2SeqLSTM, make_multi_sequences

# --- Streamlit config ---
st.set_page_config(page_title="Amazon Stock Dashboard", layout="wide")
st.title("📊 Amazon Stock Dashboard")
st.markdown("### A comprehensive dashboard for analyzing Amazon's stock performance using various models and indicators.")

st.warning("This dashboard is for educational purposes only. Not financial advice. Always do your own research.")

st.info("""
**Important Notes:**
- Models are trained on historical data and may not accurately predict future trends.
- Performance metrics reflect past performance only.
- News articles are sourced from Yahoo Finance and may be delayed.
- Sentiment scores are based on article titles and are not investment signals.
- RSI strategy is basic and should be combined with other indicators.
""")

SEQ_LEN = 30

# --- Load live stock data ---
@st.cache_data
def load_live_data():
    ticker = yf.Ticker("AMZN")
    df = ticker.history(period="1y")
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df

df = load_live_data()
df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
df["MACD"] = ta.trend.MACD(df["Close"]).macd()
df["Signal"] = ta.trend.MACD(df["Close"]).macd_signal()
df["BB_upper"] = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
df["BB_lower"] = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()

# --- Linear Regression Forecast ---
def forecast_lr(df, n_days):
    df = df[-90:].copy()
    df["Date_ordinal"] = df.index.map(pd.Timestamp.toordinal)
    X = df["Date_ordinal"].values.reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X, y)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    preds = model.predict(future_ordinals)
    return pd.Series(preds, index=future_dates)

# --- One-Step LSTM Model ---
class OneStepLSTM(nn.Module):
    def __init__(self, inp_size=5):
        super().__init__()
        self.lstm = nn.LSTM(inp_size, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- LSTM Forecast (autoregressive multi-day) ---
def forecast_lstm(df, seq_len, n_days):
    features = df[["Open", "High", "Low", "Close", "Volume"]].values
    scaler = MinMaxScaler().fit(features)
    norm = scaler.transform(features)

    model = OneStepLSTM(inp_size=5).cpu()
    model.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu"))
    model.eval()

    preds = []
    current_seq = norm[-seq_len:]

    for _ in range(n_days):
        with torch.no_grad():
            x = torch.tensor(current_seq).unsqueeze(0).float()
            pred = model(x).item()
        preds.append(pred)
        next_input = np.append(current_seq[1:], [[*current_seq[-1][:4], pred]], axis=0)
        current_seq = next_input

    dummy = np.zeros((n_days, 4))
    inv = scaler.inverse_transform(np.hstack([dummy, np.array(preds).reshape(-1, 1)]))[:, -1]
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    return pd.Series(inv, index=future_dates)

# --- Seq2Seq Forecast ---
def forecast_seq2seq(df, seq_len, n_days):
    closes = df[["Close"]].values
    scaler = MinMaxScaler().fit(closes)
    norm = scaler.transform(closes)
    X, _ = make_multi_sequences(norm, seq_len=seq_len, horizon=n_days)
    last = X[-1:]
    model = Seq2SeqLSTM(input_dim=1).cpu()
    model.load_state_dict(torch.load("seq2seq_lstm.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(last.float()).numpy().flatten()
    inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)
    return pd.Series(inv, index=dates)

# --- RSI Backtest ---
def backtest_rsi(df):
    signals = (df["RSI"] < 30).astype(int)
    rets = df["Close"].pct_change().shift(-1)
    strat = signals * rets
    return (1 + strat.fillna(0)).cumprod()

# --- News Loader ---
@st.cache_data
def load_news():
    feed = feedparser.parse("https://finance.yahoo.com/rss/headline?s=AMZN")
    return [entry for entry in feed.entries if "amazon" in entry.title.lower() or "amzn" in entry.title.lower()]

news = load_news()

# --- UI Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "📉 Historical", "🤖 Trained Models", "📰 News", "📚 Resources"])

with tab1:
    st.subheader("📅 Forecast Horizon")
    n_days = st.slider("Select Forecast Horizon (days)", 1, 30, 7)
    model_option = st.selectbox("Select Forecast Model", ["Linear Regression", "One-Step LSTM", "Seq2Seq LSTM"])
    if model_option == "Seq2Seq LSTM" and n_days != 7:
        st.warning("⚠️ Seq2Seq LSTM is trained for a 7-day forecast. Forecast horizon has been set to 7.")
        n_days = 7

    if model_option == "Linear Regression":
        forecast = forecast_lr(df, n_days)
    elif model_option == "One-Step LSTM":
        forecast = forecast_lstm(df, SEQ_LEN, n_days)
    else:
        forecast = forecast_seq2seq(df, SEQ_LEN, n_days)

    st.subheader(f"{model_option} {n_days}-Day Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-30:], y=df["Close"].iloc[-30:], name="Recent Close"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name=f"{model_option} Forecast"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    trend = "increasing 📈" if forecast.iloc[-1] > df["Close"].iloc[-1] else "decreasing 📉"
    st.markdown(f"### Conclusion: The selected model **{model_option}** predicts a **{trend}** trend over the next {n_days} days.")

# --- Tab 2: Historical ---
with tab2:
    st.subheader("📉 Historical Stock Indicators")
    st.markdown("""
    **Indicators & Strategy Overview:**
    - **RSI (Relative Strength Index)**: Oscillator that measures momentum and identifies overbought (>70) or oversold (<30) market conditions. Useful in spotting price reversals.
    - **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator that helps identify potential buy or sell signals by comparing short- and long-term moving averages.
    - **Bollinger Bands**: Volatility indicator using a moving average and standard deviation. Prices approaching upper/lower bands may indicate overbought or oversold conditions.
    - **7-Day MA / 30-Day MA**: Moving averages smooth out price data. 7-Day MA reacts faster; 30-Day MA highlights long-term trends.
    - **RSI Backtest Strategy**: A simple trading strategy where a buy signal is triggered if RSI < 30 (asset deemed undervalued). This backtest assumes holding for one day and re-evaluating RSI.
    """)
    

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(7).mean(), name="7-Day MA"))
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(30).mean(), name="30-Day MA"))
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", yaxis="y2"))
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", yaxis="y2"))
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))
    fig_hist.update_layout(
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(title="Indicators", overlaying="y", side="right"),
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("📈 RSI Backtest Strategy: Buy if RSI < 30")
    bt = backtest_rsi(df)
    fig_bt = px.line(bt, title="Cumulative Returns: RSI < 30 Strategy")
    fig_bt.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig_bt, use_container_width=True)
    st.markdown("### Conclusion: The RSI backtest strategy shows cumulative returns over time. A positive trend indicates the strategy's effectiveness.")
    st.markdown("### Note: The strategy is simplistic and should be combined with other indicators for better decision-making.")
    st.markdown("### Disclaimer: Past performance is not indicative of future results. Always do your own research before investing.")

# --- Tab 3: Trained Models Comparison ---
with tab3:
    st.subheader("📊 Trained Model Comparison (on Past Data)")

    df_train = df.copy()
    df_train["Date_ordinal"] = df_train.index.map(pd.Timestamp.toordinal)
    X = df_train["Date_ordinal"].values.reshape(-1, 1)
    y = df_train["Close"].values

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_pred_lr = lr_model.predict(X)

    # Plot LR
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(x=df_train.index, y=y, name="Actual"))
    fig_lr.add_trace(go.Scatter(x=df_train.index, y=y_pred_lr, name="LR Predicted", opacity=0.7))
    fig_lr.update_layout(title="Linear Regression Fit on Training Data", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_lr, use_container_width=True)

    # LSTM Fit on Past Data
    features = df_train[["Open", "High", "Low", "Close", "Volume"]].values
    scaler_lstm = MinMaxScaler().fit(features)
    norm = scaler_lstm.transform(features)
    X_lstm, y_lstm = make_multi_sequences(norm, seq_len=SEQ_LEN, horizon=1)

    lstm_model = OneStepLSTM(inp_size=5).cpu()
    lstm_model.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu"))
    lstm_model.eval()

    with torch.no_grad():
        preds_lstm = lstm_model(X_lstm.float()).numpy().flatten()

    dummy_pad = np.zeros((len(preds_lstm), 4))
    inv_preds_lstm = scaler_lstm.inverse_transform(np.hstack([dummy_pad, preds_lstm.reshape(-1, 1)]))[:, -1]
    inv_actual_lstm = scaler_lstm.inverse_transform(np.hstack([dummy_pad, y_lstm.reshape(-1, 1)]))[:, -1]
    idx_lstm = df_train.index[-len(inv_actual_lstm):]

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=idx_lstm, y=inv_actual_lstm, name="Actual"))
    fig_lstm.add_trace(go.Scatter(x=idx_lstm, y=inv_preds_lstm, name="LSTM Predicted", opacity=0.7))
    fig_lstm.update_layout(title="LSTM Fit on Training Data", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_lstm, use_container_width=True)

    # Seq2Seq Fit on Past Data
    closes = df_train[["Close"]].values
    scaler_seq = MinMaxScaler().fit(closes)
    norm_seq = scaler_seq.transform(closes)
    X_seq, y_seq = make_multi_sequences(norm_seq, seq_len=SEQ_LEN, horizon=1)

    seq2seq_model = Seq2SeqLSTM(input_dim=1).cpu()
    seq2seq_model.load_state_dict(torch.load("seq2seq_lstm.pth", map_location="cpu"))
    seq2seq_model.eval()

    with torch.no_grad():
        preds_seq = seq2seq_model(X_seq.float()).numpy().flatten()

    inv_preds_seq = scaler_seq.inverse_transform(preds_seq.reshape(-1, 1)).flatten()
    inv_actual_seq = scaler_seq.inverse_transform(y_seq.reshape(-1, 1)).flatten()
    idx_seq = df_train.index[-len(inv_actual_seq):]

    # Ensure lengths match before plotting
    inv_preds_seq = inv_preds_seq[-len(idx_seq):]
    inv_actual_seq = inv_actual_seq[-len(idx_seq):]
    if len(inv_preds_seq) != len(idx_seq) or len(inv_actual_seq) != len(idx_seq):
        min_len = min(len(inv_preds_seq), len(idx_seq), len(inv_actual_seq))
        inv_preds_seq = inv_preds_seq[:min_len]
        inv_actual_seq = inv_actual_seq[:min_len]
        idx_seq = idx_seq[:min_len]

    fig_seq = go.Figure()
    fig_seq.add_trace(go.Scatter(x=idx_seq, y=inv_actual_seq, name="Actual"))
    fig_seq.add_trace(go.Scatter(x=idx_seq, y=inv_preds_seq, name="Seq2Seq Predicted", opacity=0.7))
    fig_seq.update_layout(title="Seq2Seq Fit on Training Data", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_seq, use_container_width=True)

    st.markdown("### Conclusion: The models show varying performance on training data. LSTM and Seq2Seq models capture trends better than Linear Regression, but all models have room for improvement.")
    st.markdown("### Note: The models are trained on historical data and may not predict future trends accurately. Always consider multiple factors before making investment decisions.")
    st.markdown("### Disclaimer: The models are for educational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.")
    st.markdown("### Disclaimer: The performance metrics are based on historical data and may not reflect future performance. Always verify with reliable sources.")
    

    # --- Metrics Comparison ---
    st.markdown("### 📏 Metrics (RMSE, MAPE) on Training Data")
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
    mape_lr = mean_absolute_percentage_error(y, y_pred_lr)

    min_len_lstm = min(len(inv_actual_lstm), len(inv_preds_lstm))
    rmse_lstm = np.sqrt(mean_squared_error(inv_actual_lstm[:min_len_lstm], inv_preds_lstm[:min_len_lstm]))
    mape_lstm = mean_absolute_percentage_error(inv_actual_lstm[:min_len_lstm], inv_preds_lstm[:min_len_lstm])

    min_len_seq = min(len(inv_actual_seq), len(inv_preds_seq))
    rmse_seq = np.sqrt(mean_squared_error(inv_actual_seq[:min_len_seq], inv_preds_seq[:min_len_seq]))
    mape_seq = mean_absolute_percentage_error(inv_actual_seq[:min_len_seq], inv_preds_seq[:min_len_seq])

    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "One-Step LSTM", "Seq2Seq LSTM"],
        "RMSE": [rmse_lr, rmse_lstm, rmse_seq],
        "MAPE (%)": [mape_lr, mape_lstm, mape_seq]
    })

    st.dataframe(metrics_df.style.format({"RMSE": "{:.2f}", "MAPE (%)": "{:.2f}"}))

    st.markdown("### 📈 Model Performance Over Time")

    time_range = st.selectbox("Select Time Range", [
        "Last 1 Day", "Last 7 Days", "Last 15 Days", "Last 30 Days",
        "Last 3 Months", "Last 6 Months", "Last 1 Year", "Select Month"
    ])

    if time_range == "Last 1 Day":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.Timedelta(days=1)
        mask_seq = idx_seq >= idx_seq[-1] - pd.Timedelta(days=1)
    elif time_range == "Last 7 Days":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.Timedelta(days=7)
        mask_seq = idx_seq >= idx_seq[-1] - pd.Timedelta(days=7)
    elif time_range == "Last 15 Days":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.Timedelta(days=15)
        mask_seq = idx_seq >= idx_seq[-1] - pd.Timedelta(days=15)
    elif time_range == "Last 30 Days":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.Timedelta(days=30)
        mask_seq = idx_seq >= idx_seq[-1] - pd.Timedelta(days=30)
    elif time_range == "Last 3 Months":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.DateOffset(months=3)
        mask_seq = idx_seq >= idx_seq[-1] - pd.DateOffset(months=3)
    elif time_range == "Last 6 Months":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.DateOffset(months=6)
        mask_seq = idx_seq >= idx_seq[-1] - pd.DateOffset(months=6)
    elif time_range == "Last 1 Year":
        mask_lstm = idx_lstm >= idx_lstm[-1] - pd.DateOffset(years=1)
        mask_seq = idx_seq >= idx_seq[-1] - pd.DateOffset(years=1)
    else:
        all_months = sorted(set(pd.to_datetime(idx_lstm).to_series().dt.to_period("M").dt.to_timestamp()))
        all_months = [m.strftime("%Y-%m") for m in all_months]
        selected_month = st.selectbox("Select Month", all_months)
        selected_dt = pd.to_datetime(selected_month).tz_localize(idx_lstm.tz)
        mask_lstm = (idx_lstm >= selected_dt) & (idx_lstm < selected_dt + pd.offsets.MonthEnd(1))
        mask_seq = (idx_seq >= selected_dt) & (idx_seq < selected_dt + pd.offsets.MonthEnd(1))

    fig_performance_zoom = go.Figure()
    fig_performance_zoom.add_trace(go.Scatter(x=idx_lstm[mask_lstm], y=inv_actual_lstm[mask_lstm], name="Actual (LSTM)"))
    fig_performance_zoom.add_trace(go.Scatter(x=idx_lstm[mask_lstm], y=inv_preds_lstm[mask_lstm], name="LSTM Predicted", opacity=0.7))
    fig_performance_zoom.add_trace(go.Scatter(x=idx_seq[mask_seq], y=inv_actual_seq[mask_seq], name="Actual (Seq2Seq)"))
    fig_performance_zoom.add_trace(go.Scatter(x=idx_seq[mask_seq], y=inv_preds_seq[mask_seq], name="Seq2Seq Predicted", opacity=0.7))
    fig_performance_zoom.update_layout(title=f"Model Performance Over Time ({time_range})", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_performance_zoom, use_container_width=True)

with tab4:
    st.subheader("📰 Amazon News")
    st.markdown("### Disclaimer: This dashboard is for educational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.")
    st.markdown("### Disclaimer: The news articles are sourced from Yahoo Finance and may not reflect the latest updates. Always verify with reliable sources.")
    st.markdown("### Note: Sentiment analysis is based on the title of the news article. A score closer to 1 indicates positive sentiment, while a score closer to -1 indicates negative sentiment.")

    sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])

    for entry in news[:10]:
        sentiment = TextBlob(entry.title).sentiment.polarity
        label = "🟢 Positive" if sentiment > 0 else "🔴 Negative" if sentiment < 0 else "🟡 Neutral"

        if sentiment_filter == "Positive" and sentiment <= 0:
            continue
        elif sentiment_filter == "Negative" and sentiment >= 0:
            continue
        elif sentiment_filter == "Neutral" and sentiment != 0:
            continue

        st.markdown(f"**[{entry.title}]({entry.link})**\n\n*{label}*\n\n{entry.summary[:200]}...\n")
        st.markdown(f"---")
        st.markdown(f"**Published on:** {entry.published}")
        st.markdown(f"**Sentiment Score:** {sentiment:.2f}")
        st.markdown(f"---")

with tab5:
    st.subheader("Additional Resources")
    st.markdown("Explore more about trading strategies and financial analysis:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- [RSI Trading Strategy](https://www.investopedia.com/terms/r/rsi.asp)")
        st.markdown("- [MACD Trading Strategy](https://www.investopedia.com/terms/m/macd.asp)")
        st.markdown("- [Bollinger Bands Strategy](https://www.investopedia.com/terms/b/bollingerbands.asp)")
        st.markdown("- [Moving Averages Guide](https://www.investopedia.com/terms/m/movingaverage.asp)")
        st.markdown("- [Backtesting Explained](https://www.investopedia.com/terms/b/backtesting.asp)")
        st.markdown("- [Investopedia](https://www.investopedia.com/)")
    with col2:
        st.markdown("- [TradingView Platform](https://www.tradingview.com/)")
        st.markdown("- [Yahoo Finance](https://finance.yahoo.com/)")
        st.markdown("- [StockCharts](https://stockcharts.com/)")
        st.markdown("- [MarketWatch](https://www.marketwatch.com/)")
        st.markdown("- [Seeking Alpha](https://seekingalpha.com/)")
        st.markdown("- [Finviz Screener](https://finviz.com/)")