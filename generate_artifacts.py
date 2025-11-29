import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# --- Configuration ---
DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE = "customer_churn_model.pkl"
ENCODERS_FILE = "encoders.pkl"

def generate_artifacts():
    """
    Loads, cleans, processes data, trains the model, and saves both the
    LabelEncoders dictionary and the trained model (RandomForestClassifier).
    """
    print("Starting artifact generation...")

    # 1. Data Loading
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}. Please ensure it is in the same directory.")
        return

    # Drop customerID (as per notebook)
    df = df.drop(columns=["customerID"])

    # 2. Data Cleaning and Type Conversion
    # Replace spaces in TotalCharges with '0.0' and convert to float
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 3. Target Encoding
    # Convert 'Churn' ('Yes', 'No') to (1, 0)
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

    # 4. Feature and Target Split
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 5. Label Encoding of Categorical Features (And saving encoders)
    object_columns = X.select_dtypes(include="object").columns
    encoders = {}

    for column in object_columns:
        label_encoder = LabelEncoder()
        X[column] = label_encoder.fit_transform(X[column])
        encoders[column] = label_encoder

    # SeniorCitizen is int64 (0/1), so it's already numerical.

    # 6. Train-Test Split (needed for SMOTE resampling)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. SMOTE Resampling (Handling Imbalance)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Training data shape after SMOTE: {X_train_smote.shape}")
    print(f"Target distribution: {y_train_smote.value_counts()}")

    # 8. Model Training (Using RandomForestClassifier as selected in notebook)
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train_smote, y_train_smote)
    print("Random Forest Model training complete.")

    # 9. Save Artifacts
    try:
        # Save Encoders
        with open(ENCODERS_FILE, "wb") as f:
            pickle.dump(encoders, f)
        print(f"Successfully saved encoders to {ENCODERS_FILE}")

        # Save Model
        model_data = {"model": rfc, "features_names": X.columns.tolist()}
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Successfully saved model to {MODEL_FILE}")

        print("\nAll artifacts generated successfully.")

    except Exception as e:
        print(f"An error occurred during file saving: {e}")

if __name__ == "__main__":
    generate_artifacts()