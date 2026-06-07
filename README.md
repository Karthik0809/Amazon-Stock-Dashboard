# 📊 Amazon Stock Dashboard

A powerful and interactive dashboard for analyzing **Amazon (AMZN) stock performance**, deployed on **Streamlit Cloud**.

🔗 **Live App**: *(Deploy to Streamlit Cloud and update this link)*

---

## 🔧 Key Features

- **Live Forecasting** with Linear Regression, One-Step LSTM, and Seq2Seq LSTM
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and Moving Averages
- **News Sentiment Analysis** with live updates from Yahoo Finance
- **Backtesting** using RSI-based strategy
- **Model Comparison** using RMSE & MAPE metrics

---

## 📈 Technical Indicators Overview

- **RSI (Relative Strength Index)**: Detects overbought/oversold conditions
- **MACD**: Measures trend strength via EMAs
- **Bollinger Bands**: Captures price volatility and potential breakouts
- **Moving Averages**: Smoothens data to track overall trend

---

## 🚀 Deploy on Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app** → select `Karthik0809/Amazon-Stock-Dashboard` → set `app.py` as the main file
3. Click **Deploy** — Streamlit Cloud handles everything automatically, no servers needed

---

## ⚙️ Local Setup

```bash
git clone https://github.com/Karthik0809/Amazon-Stock-Dashboard.git
cd Amazon-Stock-Dashboard
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Project Structure

```
Amazon-Stock-Dashboard/
├── app.py                  # Streamlit app
├── model.py                # One-step LSTM model definition
├── seq2seq_lstm.py         # Seq2Seq LSTM model + utilities
├── requirements.txt        # Python dependencies
├── amazon_lstm_model.pth   # Trained One-Step LSTM weights
├── seq2seq_lstm.pth        # Trained Seq2Seq LSTM weights
├── amazon_stock.csv        # Historical stock data
└── README.md               # Project overview
```

---

## 🛠️ Future Enhancements

- Add more ticker symbols beyond AMZN
- Dockerize the app for portability
- Add transformer-based forecasting models

---

## 🔗 Connect

👨‍💻 [Karthik Mulugu](https://www.linkedin.com/in/karthikmulugu/)  
📬 Feel free to reach out for questions or collaboration!

---

## 📝 License

Licensed under the **MIT License** — free to use, modify, and distribute.  
See [LICENSE](LICENSE) for details.

---

⚠️ **Disclaimer**: This is an educational project and does not constitute financial advice.
