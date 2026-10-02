import pandas as pd


# --------------------------------------------
# Feature List
# --------------------------------------------


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "Volatility_20D",
    "SMA_20",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_5",
    "Return_Lag_1",
    "Volume_Lag_1"
]


# --------------------------------------------
# Create Technical Indicators
# --------------------------------------------
def create_features(df):


    df = df.copy()

    # ----------------------------------------
    # 1. Daily Return
    # ----------------------------------------
    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    # ----------------------------------------
    # 2. 20-Day Volatility
    # ----------------------------------------
    df["Volatility_20D"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    # ----------------------------------------
    # 3. 20-Day Simple Moving Average
    # ----------------------------------------
    df["SMA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    # ----------------------------------------
    # 4. 20-Day Exponential Moving Average
    # ----------------------------------------
    df["EMA_20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # ----------------------------------------
    # 5. RSI
    # ----------------------------------------
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    rs = avg_gain / avg_loss

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    # ----------------------------------------
    # 6. MACD
    # ----------------------------------------
    ema_12 = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema_26 = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["MACD"] = ema_12 - ema_26

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # ----------------------------------------
    # 7. Lag Features
    # ----------------------------------------

    # Previous trading day's close
    df["Close_Lag_1"] = (
        df["Close"].shift(1)
    )

    # Two trading days before
    df["Close_Lag_2"] = (
        df["Close"].shift(2)
    )

    # Five trading days before
    df["Close_Lag_5"] = (
        df["Close"].shift(5)
    )

    # Previous day's return
    df["Return_Lag_1"] = (
        df["Daily_Return"].shift(1)
    )

    # Previous day's volume
    df["Volume_Lag_1"] = (
        df["Volume"].shift(1)
    )

    return df


# --------------------------------------------
# Get Model Features
# --------------------------------------------
def get_model_features(df):

    feature_df = df[FEATURES].copy()

    feature_df = feature_df.dropna()

    return feature_df