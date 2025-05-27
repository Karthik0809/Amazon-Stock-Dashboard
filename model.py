import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
df = pd.read_csv("amazon_stock.csv")
df.columns = df.columns.str.strip().str.lower()
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df.dropna(subset=["date"], inplace=True)
df.set_index("date", inplace=True)
df.index.name = "Date"

# Linear Regression
print("\n--- Linear Regression ---")
df["date_ordinal"] = df.index.map(pd.Timestamp.toordinal)
X_lr = df[["date_ordinal"]].values
y_lr = df["close"].values
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
pred_lr = lr_model.predict(X_test_lr)

rmse_lr = np.sqrt(mean_squared_error(y_test_lr, pred_lr))
mape_lr = np.mean(np.abs((y_test_lr - pred_lr) / y_test_lr)) * 100
print(f"Linear Regression RMSE: {rmse_lr:.2f}, MAPE: {mape_lr:.2f}%")

# LSTM Setup
print("\n--- LSTM Training ---")
features = df[['open', 'high', 'low', 'close', 'volume']].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

SEQ_LEN = 30
X_lstm, y_lstm = [], []
for i in range(len(scaled) - SEQ_LEN):
    X_lstm.append(scaled[i:i+SEQ_LEN])
    y_lstm.append(scaled[i+SEQ_LEN][3])  # 'close' index

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

X_train_t = torch.tensor(X_train_lstm, dtype=torch.float32)
y_train_t = torch.tensor(y_train_lstm, dtype=torch.float32)
X_val_t = torch.tensor(X_val_lstm, dtype=torch.float32)
y_val_t = torch.tensor(y_val_lstm, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32)

# LSTM Model
class OneStepLSTM(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = OneStepLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
best_val_loss = float('inf')
early_stop_patience = 3
patience_counter = 0

for epoch in range(20):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).squeeze()
            val_loss += criterion(pred, yb).item()
    val_loss_avg = val_loss / len(val_loader)
    val_losses.append(val_loss_avg)

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_loss_avg:.4f}")

    # Early stopping check
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

# Save LSTM model
torch.save(model.state_dict(), "amazon_lstm_model.pth")
print("✅ LSTM model saved to amazon_lstm_model.pth")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_val_t).squeeze().numpy()

pad_pred = np.zeros((len(preds), 5))
pad_pred[:, 3] = preds
inv_preds = scaler.inverse_transform(pad_pred)[:, 3]

pad_true = np.zeros((len(y_val_lstm), 5))
pad_true[:, 3] = y_val_lstm
inv_true = scaler.inverse_transform(pad_true)[:, 3]

rmse_lstm = np.sqrt(mean_squared_error(inv_true, inv_preds))
mape_lstm = np.mean(np.abs((inv_true - inv_preds) / inv_true)) * 100

print(f"LSTM RMSE: {rmse_lstm:.2f}, MAPE: {mape_lstm:.2f}%")

# Plot loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_validation_loss.png")
plt.show()