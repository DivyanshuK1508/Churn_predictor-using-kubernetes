# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request, jsonify
# import logging
# import pickle
# import os

# # --- Configuration ---
# MODEL_FILES = ["customer_churn_model.pkl", "random_forest_churn_model.pkl"]
# ENCODERS_FILE = "encoders.pkl"

# # --- Feature Definitions ---
# FEATURE_DEFS = {
#     'gender': {'type': 'select', 'options': ['Male', 'Female']},
#     'SeniorCitizen': {'type': 'select', 'options': [0, 1]},
#     'Partner': {'type': 'select', 'options': ['Yes', 'No']},
#     'Dependents': {'type': 'select', 'options': ['Yes', 'No']},
#     'tenure': {'type': 'number', 'min': 1, 'max': 72},
#     'PhoneService': {'type': 'select', 'options': ['Yes', 'No']},
#     'MultipleLines': {'type': 'select', 'options': ['No phone service', 'No', 'Yes']},
#     'InternetService': {'type': 'select', 'options': ['DSL', 'Fiber optic', 'No']},
#     'OnlineSecurity': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
#     'OnlineBackup': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
#     'DeviceProtection': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
#     'TechSupport': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
#     'StreamingTV': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
#     'StreamingMovies': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
#     'Contract': {'type': 'select', 'options': ['Month-to-month', 'One year', 'Two year']},
#     'PaperlessBilling': {'type': 'select', 'options': ['Yes', 'No']},
#     'PaymentMethod': {'type': 'select', 'options': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']},
#     'MonthlyCharges': {'type': 'number'},
#     'TotalCharges': {'type': 'number'}
# }

# # ---------- SIMPLE LOGGING ----------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("churn-app")

# # ---------- LOAD MODEL & ENCODERS ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# LOADED_MODEL = None
# FEATURE_NAMES = None
# ENCODERS = None

# # Try multiple model filenames
# for fname in MODEL_FILES:
#     path = os.path.join(BASE_DIR, fname)
#     if os.path.exists(path):
#         logger.info(f"Loading model from {path}")
#         model_data = pickle.load(open(path, "rb"))
#         LOADED_MODEL = model_data["model"]
#         FEATURE_NAMES = model_data["features_names"]
#         break

# if LOADED_MODEL is None:
#     logger.error("MODEL FILE NOT FOUND!")

# # Load encoders
# enc_path = os.path.join(BASE_DIR, ENCODERS_FILE)
# if os.path.exists(enc_path):
#     ENCODERS = pickle.load(open(enc_path, "rb"))
# else:
#     logger.error("ENCODERS FILE NOT FOUND!")

# app = Flask(__name__, template_folder="templates")


# def preprocess_input(input_data):
#     df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

#     # Apply encoders
#     for col, encoder in ENCODERS.items():
#         if col in df.columns:
#             df[col] = encoder.transform([df[col][0]])[0]

#     # Convert numeric types
#     for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     return df


# @app.route("/")
# def index():
#     return render_template("index.html", feature_defs=FEATURE_DEFS)


# @app.route("/predict", methods=['POST'])
# def predict():
#     data = request.form.to_dict()

#     processed_df = preprocess_input(data)
#     prediction = LOADED_MODEL.predict(processed_df)[0]
#     prob = LOADED_MODEL.predict_proba(processed_df)[0]

#     churn_prob = prob[1] * 100
#     no_churn_prob = prob[0] * 100

#     result = "HIGH RISK OF CHURN" if prediction == 1 else "LOW RISK OF CHURN"

#     logger.info(f"Prediction: {result}, Data: {data}")

#     return render_template(
#         "index.html",
#         feature_defs=FEATURE_DEFS,
#         result=result,
#         churn_prob=f"{churn_prob:.2f}%",
#         no_churn_prob=f"{no_churn_prob:.2f}%"
#     )


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8082)

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import logging
import pickle
import os
import requests
import json

# =========================================================
#                  SPLUNK HEC CONFIGURATION
# =========================================================

SPLUNK_HEC_URL = "http://localhost:8088/services/collector/event"   # HEC URL
SPLUNK_HEC_TOKEN = "a4592ea0-588f-48e7-9914-d279e7fca2e9"           # Your Token

def send_to_splunk(event_data):
    """Send JSON logs to Splunk HTTP Event Collector."""
    headers = {
        "Authorization": f"Splunk {SPLUNK_HEC_TOKEN}"
    }

    payload = {
        "event": event_data
    }

    try:
        response = requests.post(
            SPLUNK_HEC_URL,
            headers=headers,
            data=json.dumps(payload),
            verify=False   # disable SSL verification for localhost
        )

        if response.status_code != 200:
            print("❌ Splunk HEC Error:", response.text)

    except Exception as e:
        print("❌ Splunk Send Error:", e)


# =========================================================
#                     FLASK APP SETUP
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-app")

app = Flask(__name__, template_folder="templates")

# Model & Encoder Files
MODEL_FILES = ["customer_churn_model.pkl", "random_forest_churn_model.pkl"]
ENCODERS_FILE = "encoders.pkl"


# =========================================================
#                LOAD MODEL AND ENCODERS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOADED_MODEL = None
FEATURE_NAMES = None
ENCODERS = None

# Load model
for fname in MODEL_FILES:
    path = os.path.join(BASE_DIR, fname)
    if os.path.exists(path):
        model_dict = pickle.load(open(path, "rb"))
        LOADED_MODEL = model_dict["model"]
        FEATURE_NAMES = model_dict["features_names"]
        logger.info(f"Model loaded from {path}")
        break

if LOADED_MODEL is None:
    logger.error("❌ Model file NOT FOUND!")

# Load encoders
enc_path = os.path.join(BASE_DIR, ENCODERS_FILE)
if os.path.exists(enc_path):
    ENCODERS = pickle.load(open(enc_path, "rb"))
    logger.info("Encoders loaded successfully.")
else:
    logger.error("❌ Encoders file NOT FOUND!")


# =========================================================
#                  PREPROCESS INPUT
# =========================================================

def preprocess_input(data):
    df = pd.DataFrame([data], columns=FEATURE_NAMES)

    for col, encoder in ENCODERS.items():
        df[col] = encoder.transform([df[col][0]])[0]

    # numeric conversions
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
#                    FEATURE FORM
# =========================================================

FEATURE_DEFS = {
    'gender': {'type': 'select', 'options': ['Male', 'Female']},
    'SeniorCitizen': {'type': 'select', 'options': [0, 1]},
    'Partner': {'type': 'select', 'options': ['Yes', 'No']},
    'Dependents': {'type': 'select', 'options': ['Yes', 'No']},
    'tenure': {'type': 'number'},
    'PhoneService': {'type': 'select', 'options': ['Yes', 'No']},
    'MultipleLines': {'type': 'select', 'options': ['No phone service', 'No', 'Yes']},
    'InternetService': {'type': 'select', 'options': ['DSL', 'Fiber optic', 'No']},
    'OnlineSecurity': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
    'OnlineBackup': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
    'DeviceProtection': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
    'TechSupport': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
    'StreamingTV': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
    'StreamingMovies': {'type': 'select', 'options': ['No', 'Yes', 'No internet service']},
    'Contract': {'type': 'select', 'options': ['Month-to-month', 'One year', 'Two year']},
    'PaperlessBilling': {'type': 'select', 'options': ['Yes', 'No']},
    'PaymentMethod': {'type': 'select', 'options': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']},
    'MonthlyCharges': {'type': 'number'},
    'TotalCharges': {'type': 'number'}
}


# =========================================================
#                    FLASK ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html", feature_defs=FEATURE_DEFS, result=None)


@app.route("/predict", methods=['POST'])
def predict():
    form = request.form.to_dict()

    df = preprocess_input(form)

    prediction = LOADED_MODEL.predict(df)[0]
    prob = LOADED_MODEL.predict_proba(df)[0]

    churn_prob = prob[1] * 100
    no_churn_prob = prob[0] * 100

    result = "HIGH RISK OF CHURN" if prediction == 1 else "LOW RISK OF CHURN"

    # ===================== SPLUNK LOGGING ======================
    log_event = {
        "prediction": result,
        "input": form,
        "churn_probability": f"{churn_prob:.2f}%",
        "no_churn_probability": f"{no_churn_prob:.2f}%"
    }

    send_to_splunk(log_event)
    # ===========================================================

    return render_template(
        "index.html",
        feature_defs=FEATURE_DEFS,
        result=result,
        churn_prob=f"{churn_prob:.2f}%",
        no_churn_prob=f"{no_churn_prob:.2f}%"
    )


# =========================================================
#                 START FLASK APP
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
