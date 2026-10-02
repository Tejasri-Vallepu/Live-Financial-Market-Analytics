# 📈 Live Financial Market Analytics & Forecasting System

**A Machine Learning-powered financial market analytics platform for market data analysis, technical indicator computation, and next-trading-day closing price forecasting.**

<p align="center">

  **Python · Machine Learning · Scikit-learn · Flask · Streamlit · Alpha Vantage**

</p>

---

## 📌 Overview

The **Live Financial Market Analytics & Forecasting System** is a data-driven application designed to collect daily stock market data, analyze historical price movements, calculate technical indicators, and forecast the next available trading day's closing price using machine learning.

The system integrates a market data pipeline, exploratory data analysis, feature engineering, multiple regression models, a Flask REST API, and an interactive Streamlit dashboard.

It is designed to demonstrate an end-to-end machine learning workflow, from data acquisition and model evaluation to prediction serving and visualization.

## 🎯 Objectives

- Retrieve daily stock market data using a financial market API.
- Perform data cleaning, validation, and exploratory analysis.
- Calculate technical indicators to analyze historical market trends.
- Engineer features using historical price, return, and volume data.
- Train and evaluate multiple machine learning regression models.
- Generate next-trading-day closing price estimates.
- Expose predictions through a Flask REST API.
- Visualize market analytics and model results in a Streamlit dashboard.

## 🏗️ System Architecture

```text
             Alpha Vantage API
                    │
                    ▼
             Data Collection
                    │
                    ▼
          Data Cleaning & Validation
                    │
                    ▼
       Exploratory Data Analysis (EDA)
                    │
                    ▼
          Technical Indicators
                    │
                    ▼
            Feature Engineering
                    │
                    ▼
          Time-Series Data Split
                    │
                    ▼
           Model Training
                    │
                    ▼
           Model Evaluation
                    │
                    ▼
           Trained ML Model
                    │
           ┌────────┴────────┐
           ▼                 ▼
       Flask API       Streamlit Dashboard
           │                 │
           └────────┬────────┘
                    ▼
          Market Prediction & Analytics
```

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Data Source | Alpha Vantage API |
| API Development | Flask, REST API |
| Dashboard | Streamlit |
| Environment Management | Python-dotenv |
| API Requests | Requests |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## ✨ Key Features

### 📊 Market Data Collection
- Fetches daily stock market data using the Alpha Vantage API.
- Supports stock symbols such as `AAPL`, subject to API availability.
- Collects Open, High, Low, Close, and Volume (OHLCV) data.
- Stores historical market data in CSV format.

### 🔍 Exploratory Data Analysis
- Historical closing price visualization.
- OHLC price trend analysis.
- Daily return distribution and summary statistics.
- Rolling volatility analysis.
- Technical indicator visualization.

### 📉 Technical Indicators

| Indicator | Description |
|---|---|
| Daily Return | Measures the percentage change in closing price |
| SMA (20) | Calculates the 20-period simple moving average |
| EMA (20) | Calculates the 20-period exponential moving average |
| RSI (14) | Measures recent price movement momentum |
| MACD | Represents the difference between 12-period and 26-period EMAs |
| MACD Signal | 9-period EMA of MACD |
| Volatility (20D) | Measures rolling standard deviation of daily returns |

### ⚙️ Feature Engineering

The system creates model features from historical market data, including:

- OHLCV attributes
- Daily returns
- Rolling volatility
- Moving averages
- RSI and MACD indicators
- Lagged closing prices
- Lagged returns
- Lagged trading volume

**Target variable:** Next available trading day's closing price.

The dataset is split chronologically into training and testing sets to preserve time order and avoid random shuffling.

### 🤖 Machine Learning Models

The following regression models were implemented and evaluated:

- Baseline Model
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor

Random Forest hyperparameter tuning was also performed using `TimeSeriesSplit` and `GridSearchCV`.

### 📈 Model Evaluation

Models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

#### Experimental Results

The following results were obtained on the current historical test dataset:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 3.6331 | 4.8319 | 0.5874 |
| Linear Regression | 8.1039 | 9.3552 | -0.5468 |
| Ridge Regression | 4.8134 | 5.9652 | 0.3711 |
| Random Forest | 4.3409 | 5.6984 | 0.4261 |
| Gradient Boosting | 4.8037 | 6.1617 | 0.3290 |
| Decision Tree | 7.4526 | 10.0079 | -0.7702 |

The baseline model outperformed the machine learning models on this particular test period. Random Forest had the lowest MAE among the evaluated machine learning models in this experiment. Hyperparameter tuning did not improve its held-out test performance.

These results are preliminary and are specific to the available dataset and evaluation period.

### 🔮 Next-Day Price Forecasting

- Retrieves the latest available daily market data.
- Applies the same feature engineering process used during training.
- Loads the saved Random Forest model.
- Predicts the next available trading day's closing price.
- Calculates the percentage difference between the latest close and predicted close.

The daily data endpoint provides the latest available daily observation, which may not be an intraday real-time quote.

### 🌐 Flask REST API

A Flask-based API serves market prediction results.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks API status |
| `/predict?symbol=AAPL` | GET | Generates a next-day closing price estimate |

Example request:

```http
GET http://127.0.0.1:5000/predict?symbol=AAPL
```

Example response structure:

```json
{
  "status": "success",
  "symbol": "AAPL",
  "latest_date": "YYYY-MM-DD",
  "current_close": 0.00,
  "predicted_next_close": 0.00,
  "expected_change_percent": 0.00
}
```


### 🖥️ Interactive Streamlit Dashboard

The dashboard provides a unified interface for market analysis and model outputs.

- Latest available closing price
- Predicted next closing price
- Expected percentage change
- Latest available market date
- Historical closing price chart
- OHLC visualization
- SMA and EMA analysis
- RSI and MACD charts
- Rolling volatility visualization
- Actual vs. predicted comparison
- Model evaluation metrics
- Recent market data table
### Start the Flask API

```bash
python api/app.py
```

The API will be available at:

```text
http://127.0.0.1:5000
```

###  Start the Streamlit Dashboard

Open another terminal, activate the same virtual environment, and run:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will provide a local URL in the terminal to open the dashboard.

## 📋 Requirements

The main dependencies include:

```text
pandas
numpy
matplotlib
scikit-learn
joblib
requests
python-dotenv
flask
streamlit
```

Install them using the provided `requirements.txt`.

## 🔐 Security & Data Handling

- API credentials are stored in environment variables.
- `.env` should be excluded using `.gitignore`.
- Model artifacts are loaded from the local `models/` directory.
- API errors are handled by the application.
- Avoid uploading credentials, private data, or sensitive files to GitHub.

## ⚠️ Limitations

- The current experiment uses a relatively small historical dataset, so the reported metrics may not generalize to other periods.
- Market conditions can change, and historical patterns may not persist.
- The daily market endpoint does not guarantee intraday real-time prices.
- The current baseline outperformed the trained machine learning models on the held-out test period.
- The forecast is a point estimate and does not represent a guaranteed future price.

This project is intended for educational and analytical purposes. It is **not financial, investment, or trading advice**.

## 🔮 Future Enhancements

- Expand historical data coverage.
- Automate periodic data collection and validation.
- Add rolling-window backtesting.
- Compare additional time-series forecasting methods.
- Explore LSTM, GRU, and transformer-based approaches where justified by data availability.
- Add automated model retraining and model monitoring.
- Introduce prediction uncertainty estimates.
- Containerize the application with Docker.
- Deploy the dashboard and API to a cloud platform.
- Add automated tests and continuous integration.





*Built as an end-to-end machine learning project exploring financial data analytics, forecasting, API development, and interactive visualization.*
"# Live-Financial-Market-Analytics" 
