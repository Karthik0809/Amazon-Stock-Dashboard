import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, warnings
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

os.makedirs('notebook_outputs', exist_ok=True)
print("=== Step 1: EDA + LR + LSTM ===")

# Load 2yr data
df = pd.read_csv('amazon_stock.csv')
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True); df.sort_index(inplace=True)
df = df[df.index >= df.index[-1] - pd.DateOffset(years=2)]
print(f"Data: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

# EDA plot
returns = df['close'].pct_change().dropna()
vol30   = returns.rolling(30).std() * np.sqrt(252) * 100
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
axes[0,0].plot(df.index, df['close'], color='#2563eb', linewidth=1)
axes[0,0].set_title('AMZN Close Price (2yr)')
axes[0,1].plot(returns.index, returns*100, color='#64748b', lw=0.6, alpha=0.8)
axes[0,1].set_title('Daily Returns (%)')
axes[1,0].hist(returns*100, bins=60, color='#2563eb', alpha=0.8, edgecolor='white')
axes[1,0].set_title('Return Distribution')
axes[1,1].plot(vol30.index, vol30, color='#d97706')
axes[1,1].set_title('30D Rolling Volatility (%)')
plt.tight_layout()
plt.savefig('notebook_outputs/01_eda.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"EDA: mean={returns.mean()*100:.3f}%  vol={returns.std()*np.sqrt(252)*100:.1f}%  skew={returns.skew():.3f}  kurt={returns.kurtosis():.3f}")

# Feature engineering
delta = df['close'].diff()
df['RSI']  = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9))
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['Returns'] = df['close'].pct_change()
df_feat = df.dropna()
closes = df_feat['close'].values
dates  = df_feat.index
n, split = len(closes), int(len(closes) * 0.8)

# Naive + LR
naive_rmse = float(np.sqrt(np.mean((closes[split:] - closes[split-1:n-1])**2)))
naive_mape = float(np.mean(np.abs((closes[split:] - closes[split-1:n-1]) / (closes[split:]+1e-9))) * 100)
lr      = LinearRegression().fit(np.arange(split).reshape(-1,1), closes[:split])
lr_pred = lr.predict(np.arange(split, n).reshape(-1,1))
lr_rmse = float(np.sqrt(np.mean((closes[split:] - lr_pred)**2)))
lr_mape = float(np.mean(np.abs((closes[split:] - lr_pred) / (closes[split:]+1e-9))) * 100)
print(f"Naive: RMSE=${naive_rmse:.2f}  MAPE={naive_mape:.2f}%")
print(f"LR:    RMSE=${lr_rmse:.2f}  MAPE={lr_mape:.2f}%  ({(naive_rmse-lr_rmse)/naive_rmse*100:.1f}% vs naive)")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(dates[:split], closes[:split], color='#cbd5e1', lw=1, label='Train')
ax.plot(dates[split:], closes[split:], color='#0f172a', lw=1.5, label='Actual')
ax.plot(dates[split:], lr_pred, color='#2563eb', lw=1.5, ls='--', label='LR')
ax.axvline(dates[split], color='#94a3b8', ls=':')
ax.set_title('Linear Regression - Test Set'); ax.legend()
plt.tight_layout()
plt.savefig('notebook_outputs/02_lr.png', dpi=120, bbox_inches='tight')
plt.close()

# LSTM training
SEQ = 30
feats_arr   = df_feat[['open','high','low','close','volume']].values
sc_lstm     = MinMaxScaler().fit(feats_arr)
norm        = sc_lstm.transform(feats_arr)
Xs = np.array([norm[i:i+SEQ] for i in range(len(norm)-SEQ)])
Ys = np.array([norm[i+SEQ, 3] for i in range(len(norm)-SEQ)])  # close = col 3
ss = int(len(Xs) * 0.8)
X_tr = torch.tensor(Xs[:ss], dtype=torch.float32)
y_tr = torch.tensor(Ys[:ss], dtype=torch.float32)
X_te = torch.tensor(Xs[ss:], dtype=torch.float32)
y_te = torch.tensor(Ys[ss:], dtype=torch.float32)
loader     = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32)
val_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=32)

class OneStepLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, batch_first=True)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

mdl  = OneStepLSTM()
opt  = torch.optim.Adam(mdl.parameters(), lr=0.001)
crit = nn.MSELoss()
tr_l, vl_l, best, patience = [], [], float('inf'), 0

print("Training LSTM...")
for ep in range(50):
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
    if vl < best:
        best = vl; patience = 0
        torch.save(mdl.state_dict(), 'amazon_lstm_model.pth')
    else:
        patience += 1
        if patience >= 5:
            print(f"  Early stopping at epoch {ep+1}")
            break
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1:3d} | Train: {tr_l[-1]:.5f} | Val: {vl_l[-1]:.5f}")

mdl.load_state_dict(torch.load('amazon_lstm_model.pth', map_location='cpu', weights_only=False))
mdl.eval()
with torch.no_grad():
    ps = mdl(X_te).squeeze().numpy()
dummy = np.zeros((len(ps), 5)); dummy[:, 3] = ps
inv_p = sc_lstm.inverse_transform(dummy)[:, 3]
dummy2 = np.zeros((len(y_te), 5)); dummy2[:, 3] = y_te.numpy()
inv_t = sc_lstm.inverse_transform(dummy2)[:, 3]
lstm_rmse = float(np.sqrt(np.mean((inv_t - inv_p)**2)))
lstm_mape = float(np.mean(np.abs((inv_t - inv_p) / (inv_t+1e-9))) * 100)
print(f"LSTM:  RMSE=${lstm_rmse:.2f}  MAPE={lstm_mape:.2f}%  ({(naive_rmse-lstm_rmse)/naive_rmse*100:.1f}% vs naive)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(tr_l, label='Train', color='#2563eb')
axes[0].plot(vl_l, label='Val',   color='#ef4444')
axes[0].set_title('One-Step LSTM - Loss Curves')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss'); axes[0].legend()
td = dates[SEQ+ss:][:len(inv_p)]
axes[1].plot(td, inv_t, color='#0f172a', lw=1.5, label='Actual')
axes[1].plot(td, inv_p, color='#2563eb', lw=1.2, ls='--', label='LSTM')
axes[1].set_title(f'LSTM Test  RMSE={lstm_rmse:.2f}  MAPE={lstm_mape:.2f}%')
axes[1].legend()
plt.tight_layout()
plt.savefig('notebook_outputs/03_lstm_training.png', dpi=120, bbox_inches='tight')
plt.savefig('training_validation_loss.png', dpi=120, bbox_inches='tight')
plt.close()
print("LSTM plots saved.")
np.save('notebook_outputs/_step1_metrics.npy', [naive_rmse, naive_mape, lr_rmse, lr_mape, lstm_rmse, lstm_mape])
print("Step 1 complete.")

