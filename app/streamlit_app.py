# ============================================================
# Live Financial Market Analytics & Forecasting System
# ============================================================

import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

API_URL = "http://127.0.0.1:5000"


# ============================================================
# PROJECT PATH
# ============================================================


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# ============================================================
# PROJECT MODULES
# ============================================================

from src.data_loader import fetch_market_data

from src.features import (
    create_features,
    get_model_features
)

from src.prediction import (
    load_model,
    load_feature_list,
    predict_next_close,
    calculate_expected_change
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Market Analytics",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📈 Live Financial Market Analytics & Forecasting")

st.write(
    "Live market data, technical indicators and "
    "next-day closing price forecasting."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Market Settings")

symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL"
).upper().strip()

refresh = st.sidebar.button(
    "🔄 Refresh Market Data"
)


# ============================================================
# MAIN PIPELINE
# ============================================================

try:

    # --------------------------------------------------------
    # 1. FETCH LIVE MARKET DATA
    # --------------------------------------------------------

    live_df = fetch_market_data(symbol)

    if live_df.empty:
        st.error("No market data available.")
        st.stop()


    # --------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # --------------------------------------------------------

    
    featured_df = create_features(live_df)


    # --------------------------------------------------------
    # 3. GET MODEL FEATURES
    # --------------------------------------------------------

    X_live = get_model_features(featured_df)


    # --------------------------------------------------------
    # 4. LOAD SAVED MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # 5. LOAD TRAINING FEATURE LIST
    # --------------------------------------------------------

    feature_list = load_feature_list()


    # --------------------------------------------------------
    # 6. MAINTAIN SAME FEATURE ORDER
    # --------------------------------------------------------

    
    X_live = X_live[feature_list]


    # --------------------------------------------------------
    # 7. LATEST DATA FOR PREDICTION
    # --------------------------------------------------------

    latest_features = X_live.tail(1)


    # --------------------------------------------------------
    # 8. CURRENT CLOSE
    # --------------------------------------------------------

    current_close = live_df["Close"].iloc[-1]


    # --------------------------------------------------------
    # 9. NEXT-DAY PREDICTION
    # --------------------------------------------------------

    predicted_close = predict_next_close(
        model,
        latest_features
    )


    # --------------------------------------------------------
    # 10. EXPECTED PERCENTAGE CHANGE
    # --------------------------------------------------------

    expected_change = calculate_expected_change(
        current_close,
        predicted_close
    )


    # --------------------------------------------------------
    # 11. LATEST DATE
    # --------------------------------------------------------

    latest_date = live_df.index[-1]


    # ========================================================
    # MARKET OVERVIEW
    # ========================================================

    st.subheader("📊 Market Overview")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Latest Close",
            f"${current_close:.2f}"
        )


    with col2:

        st.metric(
            "Predicted Next Close",
            f"${predicted_close:.2f}"
        )


    with col3:

        st.metric(
            "Expected Change",
            f"{expected_change:+.2f}%"
        )


    with col4:

        st.metric(
            "Latest Available Date",
            latest_date.strftime("%Y-%m-%d")
        )


    # ========================================================
    # PRICE CHART
    # ========================================================

    st.subheader("📈 Historical Price")

    st.line_chart(
        live_df["Close"]
    )


    # ========================================================
    # OHLC CHART
    # ========================================================

    st.subheader("📊 OHLC Price Analysis")


    ohlc_data = live_df[
        ["Open", "High", "Low", "Close"]
    ]

    st.line_chart(
        ohlc_data
    )


    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    st.subheader("📐 Technical Indicators")


    # --------------------------------------------------------
    # SMA + EMA
    # --------------------------------------------------------

    st.write("### Moving Averages")


    moving_average_data = featured_df[
        [
            "Close",
            "SMA_20",
            "EMA_20"
        ]
    ]


    st.line_chart(
        moving_average_data
    )


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    st.write("### RSI (14)")


    st.line_chart(
        featured_df["RSI_14"]
    )


    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    st.write("### MACD")


    macd_data = featured_df[
        [
            "MACD",
            "MACD_Signal"
        ]
    ]


    st.line_chart(
        macd_data
    )


    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    st.write("### 20-Day Volatility")


    st.line_chart(
        featured_df["Volatility_20D"]
    )


    # ========================================================
    # LATEST TECHNICAL INDICATOR VALUES
    # ========================================================

    st.subheader("🔎 Latest Technical Indicators")


    latest_indicator = featured_df.iloc[-1]


    indicator_col1, indicator_col2, indicator_col3, indicator_col4 = (
        st.columns(4)
    )


    with indicator_col1:

        st.metric(
            "SMA 20",
            f"${latest_indicator['SMA_20']:.2f}"
        )


    with indicator_col2:

        st.metric(
            "EMA 20",
            f"${latest_indicator['EMA_20']:.2f}"
        )


    with indicator_col3:

        st.metric(
            "RSI 14",
            f"{latest_indicator['RSI_14']:.2f}"
        )


    with indicator_col4:

        st.metric(
            "Volatility 20D",
            f"{latest_indicator['Volatility_20D']:.4f}"
        )


    # ========================================================
    # HISTORICAL ACTUAL VS PREDICTED
    # ========================================================

    st.subheader("🎯 Actual vs Predicted")


    historical_file = "data/actual_vs_predicted.csv"


    if os.path.exists(historical_file):

        actual_vs_predicted = pd.read_csv(
            historical_file,
            index_col=0,
            parse_dates=True
        )


        st.line_chart(
            actual_vs_predicted[
                ["Actual", "Predicted"]
            ]
        )


    else:

        st.info(
            "Historical prediction file not found."
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader("📋 Model Performance")


    metrics_data = pd.DataFrame({

        "Model": [
            "Baseline",
            "Linear Regression",
            "Ridge Regression",
            "Random Forest",
            "Gradient Boosting",
            "Decision Tree"
        ],

        "MAE": [
            3.633125,
            8.103921,
            4.813406,
            4.340882,
            4.803666,
            7.452625
        ],

        "RMSE": [
            4.831931,
            9.355189,
            5.965248,
            5.698350,
            6.161704,
            10.007934
        ],

        "R2": [
            0.587361,
            -0.546801,
            0.371093,
            0.426111,
            0.328987,
            -0.770183
        ]
    })


    st.dataframe(
        metrics_data,
        use_container_width=True
    )


    # ========================================================
    # DATA PREVIEW
    # ========================================================

    st.subheader("🗃️ Latest Market Data")


    st.dataframe(
        live_df.tail(10),
        use_container_width=True
    )


    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    st.subheader("ℹ️ System Information")


    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        st.write("**Symbol**")
        st.write(symbol)


    with info_col2:

        st.write("**Available Records**")
        st.write(len(live_df))


    with info_col3:

        st.write("**Model**")
        st.write("Random Forest Regressor")


# ============================================================
# ERROR HANDLING
# ============================================================

except Exception as e:

    st.error(
        f"Unable to load market data: {e}"
    )
