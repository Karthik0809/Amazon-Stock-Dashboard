import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from seq2seq_lstm import Seq2SeqLSTM, make_multi_sequences

# --------- Page Config ----------
st.set_page_config(page_title="Amazon Stock Dashboard", layout="wide")

# --------- Sidebar Controls ----------
st.sidebar.header("Controls")
seq_len = st.sidebar.slider("Lookback Window (days)", 5, 60, 30)
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)
show_boll = st.sidebar.checkbox("Show Bollinger Bands", True)
show_backtest = st.sidebar.checkbox("Show Backtest", True)

# --------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_stock.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

df = load_data()

# --------- Linear Regression Model ----------
def get_lr(df):
    f = df.copy()
    f["return_pct"] = f["close"].pct_change() * 100
    f["vol_ma7"] = f["volume"].rolling(7).mean()
    f["volatility_7"] = f["return_pct"].rolling(7).std()
    f["MA10"] = f["close"].rolling(10).mean()
    f["MA50"] = f["close"].rolling(50).mean()
    f.dropna(inplace=True)
    f["target"] = f["close"].shift(-1)
    f.dropna(inplace=True)

    split = int(0.8 * len(f))
    train, test = f.iloc[:split], f.iloc[split:]
    X_train = train[["return_pct", "vol_ma7", "volatility_7", "MA10", "MA50"]].values
    y_train = train["target"].values
    X_test = test[["return_pct", "vol_ma7", "volatility_7", "MA10", "MA50"]].values
    y_test = test["target"].values

    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    return test.index, y_test, y_pred

dates_lr, y_true_lr, y_pred_lr = get_lr(df)

# --------- One-Step LSTM (Your Architecture) ----------
class OneStepLSTM(nn.Module):
    def __init__(self, inp_size):
        super().__init__()
        self.lstm = nn.LSTM(inp_size, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use only last timestep

def get_onestep_lstm(df, seq_len):
    features = df[["open", "high", "low", "close", "volume"]].values
    scaler = MinMaxScaler().fit(features)
    norm = scaler.transform(features)

    X, y = make_multi_sequences(norm, seq_len=seq_len, horizon=1)

    model = OneStepLSTM(inp_size=5).cpu()
    model.load_state_dict(torch.load("amazon_lstm_model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(X.float()).numpy().flatten()

    pred_inv = scaler.inverse_transform(
        np.hstack([np.zeros((len(preds), 4)), preds.reshape(-1, 1)])
    )[:, -1]
    true_inv = scaler.inverse_transform(
        np.hstack([np.zeros((len(y), 4)), y.reshape(-1, 1)])
    )[:, -1]

    idx = df.index[-len(true_inv):]
    return idx, true_inv, pred_inv

dates_lstm, y_true_lstm, y_pred_lstm = get_onestep_lstm(df, seq_len)

# --------- Backtesting ----------
def backtest_lr(dts, y_true, y_pred):
    sig = (y_pred > y_true).astype(int)[:-1]
    rets = np.diff(y_true) / y_true[:-1]
    strat = sig * rets
    cum = (1 + strat).cumprod()
    return pd.Series(cum, index=dts[1:])

bt_series = backtest_lr(dates_lr, y_true_lr, y_pred_lr)

# --------- Seq2Seq Multi-step Forecast ----------
def get_seq2seq(df, seq_len, horizon):
    closes = df[["close"]].values
    scaler = MinMaxScaler().fit(closes)
    norm = scaler.transform(closes)
    X, _ = make_multi_sequences(norm, seq_len=seq_len, horizon=horizon)
    last = X[-1:].float()

    model = Seq2SeqLSTM(input_dim=norm.shape[1]).cpu()
    model.load_state_dict(torch.load("seq2seq_lstm.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        fc_norm = model(last).numpy().flatten()

    fc = scaler.inverse_transform(fc_norm.reshape(-1, 1)).flatten()
    mean, std = fc.mean(), fc.std()
    idx = pd.date_range(df.index[-1] + pd.Timedelta(1, "D"), periods=horizon)
    return pd.Series(fc, index=idx), mean - std, mean + std

# --------- RMSE Function ----------
def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

# --------- Metrics ----------
c1, c2, c3 = st.columns(3)
c1.metric("LR RMSE", f"{rmse(y_true_lr, y_pred_lr):.2f}")
c2.metric("LSTM RMSE", f"{rmse(y_true_lstm, y_pred_lstm):.2f}")
c3.metric("Data Points", len(df))

# --------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Historical", "Forecasts", "Models"])

with tab1:
    st.subheader("Historical Close Price" + (" + Bollinger" if show_boll else ""))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
    if show_boll:
        m7 = df["close"].rolling(7).mean()
        m30 = df["close"].rolling(30).mean()
        m20 = df["close"].rolling(20).mean()
        s20 = df["close"].rolling(20).std()
        fig.add_trace(go.Scatter(x=df.index, y=m7, name="MA7", opacity=0.7))
        fig.add_trace(go.Scatter(x=df.index, y=m30, name="MA30", opacity=0.7))
        fig.add_trace(go.Scatter(x=df.index, y=m20 - 2 * s20, name="BB Lower", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=m20 + 2 * s20, name="BB Upper", line=dict(dash="dot"), fill="tonexty", fillcolor="rgba(200,200,200,0.2)"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"{horizon}-Day Seq2Seq Forecast")
    fc_series, fc_lower, fc_upper = get_seq2seq(df, seq_len, horizon)
    lower_series = pd.Series(fc_lower, index=fc_series.index)
    upper_series = pd.Series(fc_upper, index=fc_series.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc_series.index, y=fc_series, name="Forecast"))
    fig.add_trace(go.Scatter(x=lower_series.index, y=lower_series, name="Lower 1σ", line=dict(color="orange"), opacity=0.3))
    fig.add_trace(go.Scatter(x=upper_series.index, y=upper_series, name="Upper 1σ", line=dict(color="orange"), opacity=0.3, fill="tonexty"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if show_backtest:
        st.subheader("Backtest: LR Strategy Cumulative Returns")
        fig2 = px.line(bt_series, title="LR Strategy")
        fig2.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Actual vs Linear Regression")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_lr, y=y_true_lr, name="Actual"))
    fig.add_trace(go.Scatter(x=dates_lr, y=y_pred_lr, name="LR Pred", opacity=0.7))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Actual vs One-Step LSTM")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates_lstm, y=y_true_lstm, name="Actual"))
    fig2.add_trace(go.Scatter(x=dates_lstm, y=y_pred_lstm, name="LSTM Pred", opacity=0.7))
    fig2.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
