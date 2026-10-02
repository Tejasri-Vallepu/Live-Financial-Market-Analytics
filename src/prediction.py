import os
import joblib


# --------------------------------------------
# Model Paths
# --------------------------------------------

MODEL_PATH = "models/random_forest_model.pkl"

FEATURE_PATH = "models/feature_list.pkl"


# --------------------------------------------
# Load Saved Model
# --------------------------------------------

def load_model():
    

    # Model file undho check chestunnam
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Saved model load
    model = joblib.load(MODEL_PATH)

    return model


# --------------------------------------------
# Load Feature List
# --------------------------------------------

def load_feature_list():
    

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(
            f"Feature list not found: {FEATURE_PATH}"
        )

    feature_list = joblib.load(FEATURE_PATH)

    return feature_list


# --------------------------------------------
# Predict Next-Day Close
# --------------------------------------------

def predict_next_close(model, feature_data):
    

    # Model prediction
    prediction = model.predict(feature_data)

    # First prediction value
    predicted_close = prediction[0]

    return predicted_close


# --------------------------------------------
# Calculate Expected Change
# --------------------------------------------

def calculate_expected_change(
    current_close,
    predicted_close
):
    

    expected_change = (
        (predicted_close - current_close)
        / current_close
    ) * 100

    return expected_change