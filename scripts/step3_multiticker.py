import os, warnings
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import yfinance as yf
import mlflow
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("AMZN-Stock-Forecasting")

os.makedirs('notebook_outputs', exist_ok=True)
print("=== Step 3: Multi-ticker walk-forward ===")

def walk_forward(tdf, window=252, step=21):
    tdf = tdf.copy(); tdf.columns = tdf.columns.str.lower()
    tdf['rsi']  = 100 - 100/(1 + tdf['close'].diff().clip(lower=0).rolling(14).mean() /
                               (-tdf['close'].diff().clip(upper=0)).rolling(14).mean().replace(0,1e-9))
    tdf['macd'] = tdf['close'].ewm(span=12).mean() - tdf['close'].ewm(span=26).mean()
    tdf['vol_r']= tdf['volume'] / tdf['volume'].rolling(20).mean()
    tdf['hl_r'] = (tdf['high'] - tdf['low']) / tdf['close']
    tdf['ret']  = tdf['close'].pct_change()
    tdf['vola'] = tdf['ret'].rolling(20).std()
    tdf = tdf.dropna()
    closes = tdf['close'].values
    feat_cols = ['rsi','macd','vol_r','hl_r','ret','vola']
    actuals, lr_p, xgb_p = [], [], []
    idx = window
    while idx + step <= len(tdf):
        tc = closes[idx-window:idx]; te = closes[idx:idx+step]
        lp = LinearRegression().fit(np.arange(len(tc)).reshape(-1,1), tc).predict(
             np.arange(len(tc), len(tc)+step).reshape(-1,1))
        try:
            ft = tdf[feat_cols].iloc[idx-window:idx].copy()
            ft['t'] = closes[idx-window+1:idx+1]; ft = ft.dropna()
            xgb = XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, n_jobs=1, verbosity=0, random_state=42)
            xgb.fit(ft.drop('t',axis=1), ft['t'])
            xp = xgb.predict(tdf[feat_cols].iloc[idx:idx+step].ffill().values)
        except Exception:
            xp = lp
        actuals.extend(te); lr_p.extend(lp); xgb_p.extend(xp)
        idx += step
    a, l, x = np.array(actuals), np.array(lr_p), np.array(xgb_p)
    return {'LR RMSE':  float(np.sqrt(np.mean((a-l)**2))),
            'XGB RMSE': float(np.sqrt(np.mean((a-x)**2))),
            'LR MAPE':  float(np.mean(np.abs((a-l)/(a+1e-9)))*100),
            'XGB MAPE': float(np.mean(np.abs((a-x)/(a+1e-9)))*100)}

# Load AMZN
df_amzn = pd.read_csv('amazon_stock.csv')
df_amzn.columns = df_amzn.columns.str.strip().str.lower()
df_amzn['date'] = pd.to_datetime(df_amzn['date'], utc=True, errors='coerce').dt.tz_localize(None)
df_amzn.dropna(subset=['date'], inplace=True)
df_amzn.set_index('date', inplace=True); df_amzn.sort_index(inplace=True)
df_amzn = df_amzn[df_amzn.index >= df_amzn.index[-1] - pd.DateOffset(years=2)]

tickers = {'AMZN': df_amzn}
for t in ['MSFT', 'GOOGL']:
    try:
        raw = yf.Ticker(t).history(period='2y')
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        tickers[t] = raw
        print(f"  Fetched {t}: {len(raw)} rows")
    except Exception as e:
        print(f"  Could not fetch {t}: {e}")

rows = []
for ticker, tdf in tickers.items():
    print(f"  Walk-forward {ticker}...", end=' ', flush=True)
    r = walk_forward(tdf)
    r['Ticker'] = ticker; rows.append(r)
    print(f"LR={r['LR RMSE']:.2f}  XGB={r['XGB RMSE']:.2f}  XGB wins: {r['XGB RMSE'] < r['LR RMSE']}")

    # Log per-ticker run
    with mlflow.start_run(run_name=f"WalkForward-{ticker}"):
        mlflow.set_tag("model_type", "walk_forward")
        mlflow.set_tag("ticker", ticker)
        mlflow.log_params({
            "window_days": 252,
            "step_days":   21,
            "ticker":      ticker,
            "xgb_n_estimators": 80,
            "xgb_max_depth":    4,
            "xgb_lr":           0.1,
        })
        mlflow.log_metric("lr_rmse",   r['LR RMSE'])
        mlflow.log_metric("xgb_rmse",  r['XGB RMSE'])
        mlflow.log_metric("lr_mape",   r['LR MAPE'])
        mlflow.log_metric("xgb_mape",  r['XGB MAPE'])
        mlflow.log_metric("xgb_improvement_pct",
                          (r['LR RMSE'] - r['XGB RMSE']) / r['LR RMSE'] * 100)

mr_df = pd.DataFrame(rows).set_index('Ticker')

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(mr_df)); w = 0.35
ax.bar(x-w/2, mr_df['LR RMSE'],  w, label='Linear Regression', color='#2563eb')
ax.bar(x+w/2, mr_df['XGB RMSE'], w, label='XGBoost',           color='#059669')
ax.set_xticks(x); ax.set_xticklabels(mr_df.index)
ax.set_ylabel('Walk-Forward RMSE (USD)')
ax.set_title('Walk-Forward RMSE: AMZN vs MSFT vs GOOGL')
ax.legend()
plt.tight_layout()
plt.savefig('notebook_outputs/05_multi_ticker.png', dpi=120, bbox_inches='tight')
plt.savefig('multi_ticker_walkforward.png', dpi=120, bbox_inches='tight')
plt.close()

print("\nResults:")
print(mr_df[['LR RMSE','XGB RMSE','LR MAPE','XGB MAPE']].round(2).to_string())
print("Step 3 complete.")
