# ============================================
# Flask API for Financial Market Prediction
# ============================================

import os
import sys

from flask import Flask, jsonify, request

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(BASE_DIR)


# ------------------------------------------------
# Existing project modules import
# ------------------------------------------------
from src.data_loader import fetch_market_data
from src.features import create_features, get_model_features
from src.prediction import (
    load_model,
    load_feature_list,
    predict_next_close,
    calculate_expected_change
)


# ------------------------------------------------
# Flask application
# ------------------------------------------------
app = Flask(__name__)


# ------------------------------------------------
# Health Check API
# ------------------------------------------------
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "Financial Market API is running"
    })


# ------------------------------------------------
# Prediction API
# ------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict():

    try:

       
        symbol = request.args.get(
            "symbol",
            "AAPL"
        ).upper().strip()

        # ----------------------------------------
        # 1. Market data fetch
        # ----------------------------------------
        market_df = fetch_market_data(symbol)

        # ----------------------------------------
        # 2. Feature engineering
        # ----------------------------------------
        featured_df = create_features(
            market_df
        )

        # ----------------------------------------
        # 3. Model features
        # ----------------------------------------
        X_live = get_model_features(
            featured_df
        )

        # ----------------------------------------
        # 4. Load trained model
        # ----------------------------------------
        model = load_model()

        feature_list = load_feature_list()

        
        X_live = X_live[feature_list]

        # Latest available row
        latest_features = X_live.tail(1)

        # ----------------------------------------
        # 5. Current price
        # ----------------------------------------
        current_close = market_df[
            "Close"
        ].iloc[-1]

        # ----------------------------------------
        # 6. Prediction
        # ----------------------------------------
        predicted_close = predict_next_close(
            model,
            latest_features
        )

        # ----------------------------------------
        # 7. Expected percentage change
        # ----------------------------------------
        expected_change = calculate_expected_change(
            current_close,
            predicted_close
        )

        # ----------------------------------------
        # 8. JSON response
        # ----------------------------------------
        return jsonify({

            "status": "success",

            "symbol": symbol,

            "latest_date": str(
                market_df.index[-1].date()
            ),

            "current_close": round(
                float(current_close),
                2
            ),

            "predicted_next_close": round(
                float(predicted_close),
                2
            ),

            "expected_change_percent": round(
                float(expected_change),
                2
            )

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ------------------------------------------------
# Run Flask application
# ------------------------------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )