import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, warnings, sys
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seq2seq_lstm import Seq2SeqLSTM, make_multi_sequences

os.makedirs('notebook_outputs', exist_ok=True)
print("=== Step 2: Seq2Seq LSTM ===")

df = pd.read_csv('amazon_stock.csv')
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True); df.sort_index(inplace=True)
df = df[df.index >= df.index[-1] - pd.DateOffset(years=2)]

# Features
delta = df['close'].diff()
df['RSI'] = 100 - 100/(1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0,1e-9))
df_feat = df.dropna()

HORIZON = 7
c_vals = df_feat[['close']].values
sc2    = MinMaxScaler().fit(c_vals)
nc     = sc2.transform(c_vals)
X_s2s, y_s2s = make_multi_sequences(nc, seq_len=30, horizon=HORIZON)
sp = int(len(X_s2s) * 0.8)
X_tr, y_tr = X_s2s[:sp], y_s2s[:sp]
X_te, y_te = X_s2s[sp:], y_s2s[sp:]

loader     = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16)
val_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=16)
model = Seq2SeqLSTM(input_dim=1)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.MSELoss()
tr_l, vl_l, best, patience = [], [], float('inf'), 0

print("Training Seq2Seq...")
for ep in range(50):
    model.train(); epl = 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); opt.step(); epl += loss.item()
    tr_l.append(epl / len(loader))
    model.eval()
    with torch.no_grad():
        vl = sum(crit(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
    vl_l.append(vl)
    if vl < best:
        best = vl; patience = 0
        torch.save(model.state_dict(), 'seq2seq_lstm.pth')
    else:
        patience += 1
        if patience >= 5:
            print(f"  Early stopping at epoch {ep+1}")
            break
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1:3d} | Train: {tr_l[-1]:.5f} | Val: {vl_l[-1]:.5f}")

model.load_state_dict(torch.load('seq2seq_lstm.pth', map_location='cpu', weights_only=False))
model.eval()
with torch.no_grad():
    preds = model(X_te).numpy()
pred1 = sc2.inverse_transform(preds[:, 0].reshape(-1,1)).flatten()
true1 = sc2.inverse_transform(y_te[:, 0].numpy().reshape(-1,1)).flatten()
s2s_rmse = float(np.sqrt(np.mean((true1-pred1)**2)))
s2s_mape = float(np.mean(np.abs((true1-pred1)/(true1+1e-9)))*100)
print(f"Seq2Seq: RMSE={s2s_rmse:.2f}  MAPE={s2s_mape:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(tr_l, label='Train', color='#059669')
axes[0].plot(vl_l, label='Val',   color='#ef4444')
axes[0].set_title('Seq2Seq LSTM - Loss Curves')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss'); axes[0].legend()
axes[1].plot(true1[:200], color='#0f172a', lw=1.5, label='Actual')
axes[1].plot(pred1[:200], color='#059669', lw=1.2, ls='--', label='Seq2Seq')
axes[1].set_title(f'Seq2Seq Test  RMSE={s2s_rmse:.2f}  MAPE={s2s_mape:.2f}%')
axes[1].legend()
plt.tight_layout()
plt.savefig('notebook_outputs/04_seq2seq_training.png', dpi=120, bbox_inches='tight')
plt.close()
print("Seq2Seq plots saved.")
np.save('notebook_outputs/_step2_metrics.npy', [s2s_rmse, s2s_mape])
print("Step 2 complete.")

