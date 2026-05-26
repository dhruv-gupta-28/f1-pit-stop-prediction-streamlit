import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# FAST SAFE LABEL ENCODER
# ============================================================

class SafeLabelEncoder:

    def __init__(self):
        self.mapping = {}

    def fit(self, values):

        values = values.astype(str).unique()

        self.mapping = {
            value: idx
            for idx, value in enumerate(values)
        }

        return self

    def transform(self, values):

        values = values.astype(str)

        return np.array([
            self.mapping.get(value, -1)
            for value in values
        ])


# ============================================================
# MAIN MODEL CLASS
# ============================================================

class F1PitStopPredictor:

    def __init__(self):

        self.model = None

        self.scaler = StandardScaler()

        self.encoders = {}

        self.feature_columns = []

    # ========================================================
    # RACE PHASE FUNCTION
    # ========================================================

    @staticmethod
    def get_race_phase(progress):

        if progress < 0.33:
            return "Early"

        elif progress < 0.66:
            return "Mid"

        else:
            return "Late"

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    def feature_engineering(self, df):

        df = df.copy()

        # ----------------------------------------------------
        # DROP ID COLUMN
        # ----------------------------------------------------

        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # ----------------------------------------------------
        # ENGINEERED FEATURES
        # ----------------------------------------------------

        df["TyreWearRate"] = (
            df["TyreLife"] /
            (df["LapNumber"] + 1)
        )

        df["DegradationPerLap"] = (
            df["Cumulative_Degradation"] /
            (df["LapNumber"] + 1)
        )

        df["PositionPressure"] = (
            df["Position"] *
            df["RaceProgress"]
        )

        df["RacePhase"] = (
            df["RaceProgress"]
            .apply(self.get_race_phase)
        )

        df["LapIntensity"] = np.abs(
            df["LapTime_Delta"]
        )

        return df

    # ========================================================
    # FIT LABEL ENCODERS
    # ========================================================

    def fit_encoders(self, df):

        categorical_cols = [
            "Driver",
            "Compound",
            "Race",
            "RacePhase"
        ]

        for col in categorical_cols:

            encoder = SafeLabelEncoder()

            encoder.fit(df[col])

            self.encoders[col] = encoder

    # ========================================================
    # ENCODE FEATURES
    # ========================================================

    def encode_features(self, df):

        df = df.copy()

        for col, encoder in self.encoders.items():

            df[col] = encoder.transform(df[col])

        return df

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train(self, train_path):

        print("\nLoading dataset...")

        train_df = pd.read_csv(train_path)

        print(f"Dataset Shape: {train_df.shape}")

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        print("\nApplying feature engineering...")

        train_df = self.feature_engineering(train_df)

        # ----------------------------------------------------
        # ENCODING
        # ----------------------------------------------------

        print("Encoding categorical features...")

        self.fit_encoders(train_df)

        train_df = self.encode_features(train_df)

        # ----------------------------------------------------
        # SPLIT FEATURES & TARGET
        # ----------------------------------------------------

        X = train_df.drop(columns=["PitNextLap"])

        y = train_df["PitNextLap"]

        self.feature_columns = X.columns.tolist()

        print(f"\nTotal Features: {len(self.feature_columns)}")

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        print("Scaling features...")

        X_scaled = self.scaler.fit_transform(X)

        # ----------------------------------------------------
        # TRAIN TEST SPLIT
        # ----------------------------------------------------

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"Training Samples: {len(X_train)}")
        print(f"Validation Samples: {len(X_valid)}")

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print("\nTraining Decision Tree Model...")

        self.model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = self.model.predict(X_valid)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        f1 = f1_score(
            y_valid,
            predictions
        )

        cm = confusion_matrix(
            y_valid,
            predictions
        )

        report = classification_report(
            y_valid,
            predictions
        )

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

        return metrics

    # ========================================================
    # SAVE ARTIFACTS
    # ========================================================

    def save_artifacts(self):

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        with open("artifacts/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open("artifacts/encoders.pkl", "wb") as f:
            pickle.dump(self.encoders, f)

        with open("artifacts/features.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)

        print("\nArtifacts saved successfully!")

    # ========================================================
    # PREDICT FUNCTION
    # ========================================================

    def predict(self, df):

        df = self.feature_engineering(df)

        df = self.encode_features(df)

        X = df[self.feature_columns]

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    predictor = F1PitStopPredictor()

    metrics = predictor.train("train.csv")

    predictor.save_artifacts()

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    print(f"\nAccuracy : {metrics['accuracy']:.4f}")

    print(f"F1 Score : {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])

    print("\nTraining completed successfully!")