# 📊 Amazon Stock Dashboard

A powerful and interactive dashboard for analyzing **Amazon (AMZN) stock performance** using:
- Historical stock data
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Predictive models: Linear Regression, One-Step LSTM, and Seq2Seq LSTM
- News sentiment analysis
- Backtest strategies

## Technical Indicators:

RSI (Relative Strength Index): Measures momentum by comparing recent gains vs. losses (scale of 0–100); helps spot overbought or oversold conditions.

MACD (Moving Average Convergence Divergence): Tracks the difference between short and long-term EMAs to identify trend shifts and momentum.

Bollinger Bands: Plots a moving average with upper/lower bands (based on standard deviation); indicates volatility and potential breakouts.

Moving Averages (MA): Smooths price data to show trends — e.g., 50-day or 200-day average helps identify bullish/bearish signals.


* **RSI (Relative Strength Index):**
  Think of this like a "temperature check" for a stock — if it’s too high (above 70), the stock might be overbought (too hot); too low (below 30), it might be oversold (too cold).

* **MACD (Moving Average Convergence Divergence):**
  This looks at the *speed* and *direction* of price movement using two moving averages. When one line crosses the other, it often signals a change in trend — kind of like traffic signals for stock momentum.

* **Bollinger Bands:**
  Imagine the stock price walking on a path with elastic bands above and below it. When the price hits or stretches these bands, it may be about to bounce back or break out — showing how “wild” the price is moving.

* **Moving Averages (MA):**
  This is like taking an average of past prices to smooth out short-term ups and downs — helping you see the overall trend (like a rising or falling path).

* **EMA (Exponential Moving Average):**
  A special type of moving average that gives more weight to recent prices — like paying more attention to what happened yesterday than a month ago. It reacts faster to price changes.

---


![image](https://github.com/user-attachments/assets/e481158b-3836-4e01-87f8-0a1610e3c0b9)


## 🚀 Features

- 📈 **Live Forecasting**: Choose from 3 models to forecast 1–30 days (7 days for Seq2Seq LSTM)
- 📉 **Historical Indicators**: Includes RSI, MACD, Bollinger Bands, Moving Averages
- 🧠 **Model Comparison**: Evaluate LSTM and Seq2Seq against Linear Regression using RMSE/MAPE
- 🧪 **Backtesting**: Simple RSI-based strategy simulation (Buy if RSI < 30)
- 📰 **Sentiment Analysis**: Live news with polarity scoring and sentiment filtering
- 📚 **Resources Tab**: Curated links to learning tools and platforms

---

## 📦 Project Structure

```
amazon_stock_dashboard/
│
├── app.py                  # Streamlit dashboard
├── model.py                # LSTM model architecture
├── seq2seq_lstm.py         # Seq2Seq model and utils
├── requirements.txt        # Python package dependencies
├── run_dashboard.bat       # Windows launcher
├── README.md               # Project overview
└── *.pth                   # Trained model weights (LSTM and Seq2Seq)
```

---

## ⚙️ Installation & Usage

### 1. Clone this repository

```bash
git clone https://github.com/Karthik0809/Amazon-Stock-Dashboard.git
cd Amazon-Stock-Dashboard
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Or use the included `.bat` file (Windows only):

## 🔗 Connect with Me

📇 [Karthik Mulugu](https://www.linkedin.com/in/karthikmulugu/) — Feel free to reach out for any **doubts or clarifications**.

---

## 📝 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.  
See the [LICENSE](LICENSE) file for details.

---

**Note**: This dashboard is for educational purposes only and should not be considered financial advice.
