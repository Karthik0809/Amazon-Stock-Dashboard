import os, warnings
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("AMZN-Stock-Forecasting")

print("=== Step 4: Best-model search (honest, no leakage) ===")

df = pd.read_csv('amazon_stock.csv')
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True); df.sort_index(inplace=True)

# Rich feature set — all computed from PAST data only
delta = df['close'].diff()
df['rsi']       = 100 - 100/(1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0,1e-9))
df['macd']      = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['macd_sig']  = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_sig']
df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['hl_ratio']  = (df['high'] - df['low']) / df['close']
df['ret1']      = df['close'].pct_change()
df['ret5']      = df['close'].pct_change(5)
df['ret10']     = df['close'].pct_change(10)
df['ret21']     = df['close'].pct_change(21)
df['vola20']    = df['ret1'].rolling(20).std()
df['vola5']     = df['ret1'].rolling(5).std()
df['ma20']      = df['close'].rolling(20).mean()
df['ma50']      = df['close'].rolling(50).mean()
df['ma_ratio']  = df['ma20'] / df['ma50']
df['px_ma20']   = df['close'] / df['ma20'] - 1
df['px_ma50']   = df['close'] / df['ma50'] - 1
bb_mid = df['close'].rolling(20).mean(); bb_std = df['close'].rolling(20).std()
df['bb_pos']    = (df['close'] - bb_mid) / (2*bb_std + 1e-9)
df['mom_consistency'] = (df['ret1'] > 0).rolling(10).mean()  # fraction of up days in last 10
df['gap']       = df['open'] / df['close'].shift(1) - 1
df['dow']       = df.index.dayofweek

FEATS = ['rsi','macd','macd_hist','vol_ratio','hl_ratio','ret1','ret5','ret10','ret21',
         'vola20','vola5','ma_ratio','px_ma20','px_ma50','bb_pos','mom_consistency','gap','dow']

results = []

def evaluate(name, horizon, y_pred_dir, y_true_dir, extra_params=None):
    acc = float(np.mean(y_pred_dir == y_true_dir) * 100)
    base = float(max(np.mean(y_true_dir), 1-np.mean(y_true_dir)) * 100)  # majority class
    results.append({'Model': name, 'Horizon': f'{horizon}d', 'Dir Acc (%)': round(acc,1),
                    'Majority baseline (%)': round(base,1), 'Edge (pp)': round(acc-base,1)})
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_type", "direction_classifier")
        mlflow.log_param("horizon_days", horizon)
        if extra_params: mlflow.log_params(extra_params)
        mlflow.log_metric("directional_accuracy_pct", acc)
        mlflow.log_metric("majority_baseline_pct", base)
        mlflow.log_metric("edge_pp", acc - base)
    print(f"  {name:35s} h={horizon}d  acc={acc:.1f}%  baseline={base:.1f}%  edge={acc-base:+.1f}pp")
    return acc

for horizon in [1, 5, 21]:
    sub = df.copy()
    sub['target_ret'] = sub['close'].pct_change(horizon).shift(-horizon)
    sub = sub.dropna(subset=FEATS + ['target_ret'])
    X = sub[FEATS].values
    y_dir = (sub['target_ret'].values > 0).astype(int)
    split = int(len(sub) * 0.8)

    # XGBoost classifier on direction
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=42,
                        eval_metric='logloss')
    xgb.fit(X[:split], y_dir[:split])
    pred = xgb.predict(X[split:])
    evaluate(f"XGB-Direction-{horizon}d", horizon, pred, y_dir[split:],
             {"n_estimators":300,"max_depth":4,"lr":0.03})

    # High-confidence subset (probability filter) — only act when model is confident
    proba = xgb.predict_proba(X[split:])[:,1]
    conf_mask = (proba > 0.6) | (proba < 0.4)
    if conf_mask.sum() > 30:
        acc_conf = float(np.mean((proba[conf_mask] > 0.5).astype(int) == y_dir[split:][conf_mask]) * 100)
        cov = float(conf_mask.mean()*100)
        results.append({'Model': f'XGB-HighConf-{horizon}d', 'Horizon': f'{horizon}d',
                        'Dir Acc (%)': round(acc_conf,1), 'Majority baseline (%)': '-',
                        'Edge (pp)': f'coverage {cov:.0f}%'})
        with mlflow.start_run(run_name=f"XGB-HighConf-{horizon}d"):
            mlflow.set_tag("model_type", "direction_classifier")
            mlflow.log_param("horizon_days", horizon)
            mlflow.log_param("confidence_threshold", 0.6)
            mlflow.log_metric("directional_accuracy_pct", acc_conf)
            mlflow.log_metric("coverage_pct", cov)
        print(f"  {'XGB-HighConf-'+str(horizon)+'d':35s} h={horizon}d  acc={acc_conf:.1f}%  (coverage {cov:.0f}% of days)")

    # Logistic regression baseline
    logr = LogisticRegression(max_iter=2000).fit(X[:split], y_dir[:split])
    evaluate(f"Logistic-{horizon}d", horizon, logr.predict(X[split:]), y_dir[split:])

print()
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv('notebook_outputs/_step4_results.csv', index=False)
print("\nStep 4 complete.")
