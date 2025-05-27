# seq2seq_lstm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1) Define make_multi_sequences(...)
def make_multi_sequences(data, seq_len=30, horizon=7):
    Xs, Ys = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        Xs.append(data[i : i + seq_len, :])
        Ys.append(data[i + seq_len : i + seq_len + horizon, 0])
    return torch.tensor(Xs, dtype=torch.float32), torch.tensor(Ys, dtype=torch.float32)

# 2) Define your Seq2SeqLSTM model
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, horizon=7):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc      = nn.Linear(hidden_dim, 1)
        self.horizon = horizon

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_input = torch.zeros(x.size(0), self.horizon, h.size(2), device=x.device)
        dec_out, _ = self.decoder(dec_input, (h, c))
        return self.fc(dec_out).squeeze(-1)

# 3) Add the “main” guard at the bottom
if __name__ == "__main__":
    # — load & prepare your DataFrame —
    df = pd.read_csv("amazon_stock.csv", parse_dates=["date"]).set_index("date")
    # — feature engineering & normalization —
    features = df[["close"]].values  # for example
    scaler   = MinMaxScaler().fit(features)
    data_norm = scaler.transform(features)
    
    # 4) Build sequences
    X, Y = make_multi_sequences(data_norm, seq_len=30, horizon=7)

    # 5) Split into train/test
    split      = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_test,  Y_test  = X[split:], Y[split:]

    # 6) Instantiate model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Seq2SeqLSTM(input_dim=X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()

    # 7) Training loop
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}")

    # 8) Save the trained model
    torch.save(model.state_dict(), "seq2seq_lstm.pth")
