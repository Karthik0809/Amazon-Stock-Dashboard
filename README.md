# 📊 Amazon Stock Dashboard (Deployed on AWS EC2)

A powerful and interactive dashboard for analyzing **Amazon (AMZN) stock performance**, now deployed on an AWS EC2 instance using `Streamlit` and `systemd`.

---

## 🔧 Key Features:

- **Live Forecasting** with Linear Regression, One-Step LSTM, and Seq2Seq LSTM
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and Moving Averages
- **News Sentiment Analysis** with live updates
- **Backtesting** using RSI-based strategy
- **Model Comparison** using RMSE & MAPE
- **Deployed with Auto-Restart** on AWS EC2 using `systemd`

---

## 📈 Technical Indicators Overview

- **RSI (Relative Strength Index)**: Detects overbought/oversold conditions
- **MACD**: Measures trend strength via EMAs
- **Bollinger Bands**: Captures price volatility and potential breakouts
- **Moving Averages**: Smoothens data to track overall trend

---

## 🖥️ Deployment Steps (Done)

1. ✅ Created a `t2.micro` Ubuntu 24.04 AWS EC2 instance
2. ✅ Allowed TCP traffic on port `8501` in the EC2 security group
3. ✅ SSH’d into EC2 and installed Python, pip, and `virtualenv`
4. ✅ Transferred local project using `scp`
5. ✅ Installed project dependencies and `Streamlit`
6. ✅ Launched Streamlit app with:
   ```bash
   streamlit run app.py --server.port 8501 --server.enableXsrfProtection false
   ```
7. ✅ Made it **production-ready** by creating a `streamlit.service` systemd unit:
   ```ini
   # /etc/systemd/system/streamlit.service
   [Unit]
   Description=Amazon Stock Streamlit App
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/amazon_stock
   ExecStart=/home/ubuntu/venv/bin/streamlit run app.py --server.port 8501 --server.enableXsrfProtection false
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
8. ✅ Enabled the service to run at boot:
   ```bash
   sudo systemctl daemon-reexec
   sudo systemctl start streamlit
   sudo systemctl enable streamlit
   ```

🔗 **Live App URL**:  
[`http://<your-ec2-public-ip>:8501` ](http://50.17.153.51:8501/) 
*Example: http://18.217.XX.XX:8501*

---

## 📦 Project Structure

```
amazon_stock/
├── app.py                  # Streamlit app
├── model.py                # One-step LSTM model
├── seq2seq_lstm.py         # Seq2Seq LSTM model + utils
├── requirements.txt        # Python dependencies
├── run_dashboard.bat       # Windows launch script
├── README.md               # Project overview
└── *.pth                   # Trained PyTorch model weights
```

---

## ⚙️ Local Setup (If not deploying)

```bash
git clone https://github.com/Karthik0809/Amazon-Stock-Dashboard.git
cd Amazon-Stock-Dashboard
python -m venv venv
source venv/bin/activate   # Or venv\Scripts\activate for Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Future Enhancements

- Enable HTTPS with Let’s Encrypt + Nginx
- Dockerize the app for portability
- Setup custom domain for public access

---

## 🔗 Connect with Me

👨‍💻 [Karthik Mulugu](https://www.linkedin.com/in/karthikmulugu/)  
📬 Feel free to reach out for questions, deployment help, or collaboration!

---

## 📝 License

Licensed under the **MIT License** — free to use, modify, and distribute.  
See [LICENSE](LICENSE) for details.

---

⚠️ **Disclaimer**: This is an educational project and does not constitute financial advice.

