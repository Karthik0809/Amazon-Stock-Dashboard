import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, warnings
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import mlflow
import mlflow.pytorch
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("AMZN-Stock-Forecasting")

os.makedirs('notebook_outputs', exist_ok=True)
print("=== Step 1: EDA + LR + LSTM (full history, return prediction) ===")

# Load FULL history (all 7300+ rows, not just 2yr)
df = pd.read_csv('amazon_stock.csv')
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True); df.sort_index(inplace=True)
print(f"Full history: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

# EDA plot
returns = df['close'].pct_change().dropna()
vol30   = returns.rolling(30).std() * np.sqrt(252) * 100
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes[0,0].plot(df.index, df['close'], color='#2563eb', linewidth=1)
axes[0,0].set_title(f'AMZN Close Price (full history, {len(df)} days)')
axes[0,1].plot(returns.index, returns*100, color='#64748b', lw=0.6, alpha=0.8)
axes[0,1].set_title('Daily Returns (%)')
axes[1,0].hist(returns*100, bins=80, color='#2563eb', alpha=0.8, edgecolor='white')
axes[1,0].set_title(f'Return Distribution  skew={returns.skew():.2f}  kurt={returns.kurtosis():.2f}')
axes[1,1].plot(vol30.index, vol30, color='#d97706')
axes[1,1].set_title('30D Rolling Volatility (annualised %)')
plt.tight_layout()
plt.savefig('notebook_outputs/01_eda.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"EDA: mean={returns.mean()*100:.3f}%  vol={returns.std()*np.sqrt(252)*100:.1f}%  skew={returns.skew():.3f}  kurt={returns.kurtosis():.3f}")

# Feature engineering
delta = df['close'].diff()
df['RSI']      = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9))
df['MACD']     = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['vol_ratio']= df['volume'] / df['volume'].rolling(20).mean()
df['hl_ratio'] = (df['high'] - df['low']) / df['close']
df['Returns']  = df['close'].pct_change()
df['Volatility'] = df['Returns'].rolling(20).std()
df['MA20']     = df['close'].rolling(20).mean()
df['MA50']     = df['close'].rolling(50).mean()
df['MA_ratio'] = df['MA20'] / df['MA50']
df_feat = df.dropna()

# TARGET = next-day return (%, not absolute price)
# Naive baseline = predict 0% return (no move)
next_ret = df_feat['Returns'].shift(-1).dropna() * 100   # in %
df_feat  = df_feat.iloc[:-1]                             # align

closes = df_feat['close'].values
n      = len(df_feat)
split  = int(n * 0.8)
dates  = df_feat.index

print(f"After feature engineering: {n} rows  |  train={split}  test={n-split}")

# ── Naive baseline (predict 0% return every day) ──────────────────────────────
naive_preds = np.zeros(n - split)
actual_rets = next_ret.values[split:]
naive_mae  = float(np.mean(np.abs(actual_rets - naive_preds)))
naive_rmse = float(np.sqrt(np.mean((actual_rets - naive_preds)**2)))
naive_dir  = float(np.mean((naive_preds > 0) == (actual_rets > 0)) * 100)
print(f"Naive (0% return): MAE={naive_mae:.4f}%  RMSE={naive_rmse:.4f}%  Dir={naive_dir:.1f}%")

# ── Linear Regression on returns ──────────────────────────────────────────────
with mlflow.start_run(run_name="LinearRegression-Returns"):
    mlflow.set_tag("model_type", "classical")
    mlflow.set_tag("target", "next_day_return_%")
    mlflow.log_param("features", "RSI,MACD,vol_ratio,hl_ratio,MA_ratio,Volatility")
    mlflow.log_param("train_rows", split)

    feat_cols = ['RSI','MACD','vol_ratio','hl_ratio','Returns','Volatility','MA_ratio']
    X = df_feat[feat_cols].values
    y = next_ret.values
    lr = LinearRegression().fit(X[:split], y[:split])
    lr_pred = lr.predict(X[split:])
    lr_mae   = float(np.mean(np.abs(y[split:] - lr_pred)))
    lr_rmse  = float(np.sqrt(np.mean((y[split:] - lr_pred)**2)))
    lr_dir   = float(np.mean((lr_pred > 0) == (y[split:] > 0)) * 100)

    mlflow.log_metrics({"mae": lr_mae, "rmse": lr_rmse, "directional_accuracy_pct": lr_dir,
                        "mae_vs_naive": lr_mae - naive_mae})
    print(f"LR:    MAE={lr_mae:.4f}%  RMSE={lr_rmse:.4f}%  Dir={lr_dir:.1f}%")

    fig, ax = plt.subplots(figsize=(13, 4))
    test_dates = dates[split:]
    ax.plot(test_dates, y[split:], color='#0f172a', lw=0.8, alpha=0.7, label='Actual return')
    ax.plot(test_dates, lr_pred,  color='#2563eb', lw=0.8, alpha=0.7, label='LR predicted')
    ax.axhline(0, color='#94a3b8', ls='--', lw=0.5)
    ax.set_title(f'LR Return Prediction  MAE={lr_mae:.4f}%  Dir={lr_dir:.1f}%'); ax.legend()
    plt.tight_layout()
    plt.savefig('notebook_outputs/02_lr.png', dpi=120, bbox_inches='tight')
    mlflow.log_artifact('notebook_outputs/02_lr.png', artifact_path="plots")
    plt.close()

# ── One-Step LSTM on returns ───────────────────────────────────────────────────
SEQ        = 60    # longer lookback on full history
HIDDEN     = 128
LR_RATE    = 0.001
BATCH      = 64
MAX_EPOCHS = 80
PATIENCE   = 8

with mlflow.start_run(run_name="OneStepLSTM-Returns"):
    mlflow.set_tag("model_type", "deep_learning")
    mlflow.set_tag("target", "next_day_return_%")
    mlflow.log_params({
        "seq_len":    SEQ,
        "hidden_dim": HIDDEN,
        "lr":         LR_RATE,
        "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS,
        "patience":   PATIENCE,
        "features":   "open,high,low,close,volume,RSI,MACD,vol_ratio,hl_ratio,Returns,Volatility,MA_ratio",
        "train_rows": split,
        "dataset":    "full_history_7300+",
    })

    # Scale OHLCV + engineered features as input; target = standardised return
    input_cols = ['open','high','low','close','volume','RSI','MACD','vol_ratio','hl_ratio','Returns','Volatility','MA_ratio']
    feat_arr   = df_feat[input_cols].values.astype(np.float32)
    sc_x       = MinMaxScaler().fit(feat_arr[:split])
    norm_x     = sc_x.transform(feat_arr)

    y_arr      = next_ret.values.astype(np.float32)
    sc_y       = StandardScaler().fit(y_arr[:split].reshape(-1,1))
    norm_y     = sc_y.transform(y_arr.reshape(-1,1)).flatten()

    Xs = np.array([norm_x[i:i+SEQ] for i in range(len(norm_x)-SEQ)])
    Ys = np.array([norm_y[i+SEQ]   for i in range(len(norm_y)-SEQ)])
    ss = int(len(Xs) * 0.8)

    X_tr = torch.tensor(Xs[:ss], dtype=torch.float32)
    y_tr = torch.tensor(Ys[:ss], dtype=torch.float32)
    X_te = torch.tensor(Xs[ss:], dtype=torch.float32)
    y_te = torch.tensor(Ys[ss:], dtype=torch.float32)
    loader     = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH)

    IN_DIM = len(input_cols)

    class OneStepLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(IN_DIM, HIDDEN, num_layers=2, batch_first=True,
                                dropout=0.2)
            self.fc   = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    mdl  = OneStepLSTM()
    opt  = torch.optim.Adam(mdl.parameters(), lr=LR_RATE, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit = nn.MSELoss()
    tr_l, vl_l, best, patience_ct = [], [], float('inf'), 0

    print("Training LSTM on return prediction (full history, 2-layer)...")
    for ep in range(MAX_EPOCHS):
        mdl.train(); epl = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(mdl(xb).squeeze(), yb)
            loss.backward(); opt.step(); epl += loss.item()
        tr_l.append(epl / len(loader))
        mdl.eval()
        with torch.no_grad():
            vl = sum(crit(mdl(xb).squeeze(), yb).item() for xb, yb in val_loader) / len(val_loader)
        vl_l.append(vl)
        sched.step(vl)
        mlflow.log_metrics({"train_loss": tr_l[-1], "val_loss": vl}, step=ep)
        if vl < best:
            best = vl; patience_ct = 0
            torch.save(mdl.state_dict(), 'amazon_lstm_model.pth')
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stopping at epoch {ep+1}")
                mlflow.log_param("stopped_epoch", ep + 1)
                break
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1:3d} | Train: {tr_l[-1]:.5f} | Val: {vl:.5f}")

    mdl.load_state_dict(torch.load('amazon_lstm_model.pth', map_location='cpu', weights_only=False))
    mdl.eval()
    with torch.no_grad():
        ps_scaled = mdl(X_te).squeeze().numpy()

    # Inverse transform to % return
    inv_p = sc_y.inverse_transform(ps_scaled.reshape(-1,1)).flatten()
    inv_t = sc_y.inverse_transform(y_te.numpy().reshape(-1,1)).flatten()

    lstm_mae  = float(np.mean(np.abs(inv_t - inv_p)))
    lstm_rmse = float(np.sqrt(np.mean((inv_t - inv_p)**2)))
    lstm_dir  = float(np.mean((inv_p > 0) == (inv_t > 0)) * 100)

    mlflow.log_metrics({
        "mae":                  lstm_mae,
        "rmse":                 lstm_rmse,
        "directional_accuracy_pct": lstm_dir,
        "mae_vs_naive":         lstm_mae - naive_mae,
        "best_val_loss":        best,
    })
    mlflow.log_artifact('amazon_lstm_model.pth', artifact_path="models")

    print(f"LSTM:  MAE={lstm_mae:.4f}%  RMSE={lstm_rmse:.4f}%  Dir={lstm_dir:.1f}%")
    print(f"Naive: MAE={naive_mae:.4f}%  Dir={naive_dir:.1f}%")
    print(f"LSTM beats naive on MAE: {lstm_mae < naive_mae}  |  Dir: {lstm_dir:.1f}% vs {naive_dir:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(tr_l, label='Train', color='#2563eb')
    axes[0].plot(vl_l, label='Val',   color='#ef4444')
    axes[0].set_title('One-Step LSTM — Loss Curves (return prediction)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss'); axes[0].legend()
    td = dates[SEQ + ss + 1:][:len(inv_p)]
    axes[1].plot(td, inv_t[:len(td)], color='#0f172a', lw=0.8, alpha=0.7, label='Actual return %')
    axes[1].plot(td, inv_p[:len(td)], color='#2563eb', lw=0.8, alpha=0.7, label='LSTM predicted %')
    axes[1].axhline(0, color='#94a3b8', ls='--', lw=0.5)
    axes[1].set_title(f'LSTM Return Prediction  Dir={lstm_dir:.1f}%  MAE={lstm_mae:.4f}%')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('notebook_outputs/03_lstm_training.png', dpi=120, bbox_inches='tight')
    plt.savefig('training_validation_loss.png', dpi=120, bbox_inches='tight')
    mlflow.log_artifact('notebook_outputs/03_lstm_training.png', artifact_path="plots")
    plt.close()
    print("Plots saved.")

np.save('notebook_outputs/_step1_metrics.npy',
        [naive_mae, naive_rmse, naive_dir, lr_mae, lr_rmse, lr_dir, lstm_mae, lstm_rmse, lstm_dir])
print("Step 1 complete.")
