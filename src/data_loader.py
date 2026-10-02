import os
import requests
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

BASE_URL = "https://www.alphavantage.co/query"


# --------------------------------------------
# Fetch Market Data
# --------------------------------------------
def fetch_market_data(symbol):
    

    # API request parameters
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY
    }

    
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )

    
    response.raise_for_status()

    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(
            f"Market data unavailable: {data}"
        )

    time_series = data["Time Series (Daily)"]

    # Dictionary → DataFrame
    df = pd.DataFrame.from_dict(
        time_series,
        orient="index"
    )

    df.columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df.index = pd.to_datetime(df.index)

    df.index.name = "Date"

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df[numeric_columns] = df[numeric_columns].astype(float)

    # Old → New chronological order
    df = df.sort_index()

    return df