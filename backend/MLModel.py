import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from constants import NOMINAL_COLUMNS, TARGET_COLUMN


def _make_onehot() -> OneHotEncoder:
    """
    Create an OneHotEncoder that works with both new (sparse_output) and
    older (sparse) sklearn versions.
    """
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown='ignore')


class MLModel:
    """
    Wrapper around preprocessing logic, model training, and MLflow artifact handling
    for the Car Evaluation dataset.
    """

    def __init__(self, client: Optional[MlflowClient] = None):
        self.client = client
        self.model: Optional[XGBClassifier] = None

        # Artifacts populated after training or when loading from MLflow
        self.fill_values_nominal: Optional[dict] = None
        self.min_max_scaler_dict: Optional[dict] = None
        self.onehot_encoders: Optional[dict] = None
        self.label_encoder: Optional[dict] = None

        if self.client is not None:
            self.load_staging_model()

    # ------------------------------------------------------------------
    # Loading model & artifacts from MLflow
    # ------------------------------------------------------------------
    def load_staging_model(self) -> None:
        """
        Load the latest registered model version in stage 'Staging'
        from the MLflow Model Registry, along with the artifacts saved
        in the run's artifact directory.
        """
        try:
            latest_staging_model = None
            for registered_model in self.client.search_registered_models():
                for latest_version in registered_model.latest_versions:
                    if latest_version.current_stage == "Staging":
                        latest_staging_model = latest_version
                        break
                if latest_staging_model:
                    break

            if not latest_staging_model:
                print("No staging model found.")
                return

            model_uri = latest_staging_model.source
            self.model = mlflow.sklearn.load_model(model_uri)
            artifact_uri = model_uri.rpartition('/')[0]
            self.load_artifacts(artifact_uri)
            print("Staging model and artifacts loaded successfully.")
        except Exception as exc:
            print(f"Error loading model or artifacts: {exc}")

    def load_artifacts(self, artifact_uri: str) -> None:
        """
        Download preprocessing artifacts saved alongside the model. Expected files:
          - fill_values_nominal.json
          - min_max_scaler_dict.pkl
          - onehot_encoders.pkl
          - label_encoder.json
        """
        try:
            nominal_path = download_artifacts(
                artifact_uri=f"{artifact_uri}/fill_values_nominal.json"
            )
            with open(nominal_path, 'r') as f:
                self.fill_values_nominal = json.load(f)

            scaler_path = download_artifacts(
                artifact_uri=f"{artifact_uri}/min_max_scaler_dict.pkl"
            )
            with open(scaler_path, 'rb') as f:
                self.min_max_scaler_dict = pickle.load(f)

            encoders_path = download_artifacts(
                artifact_uri=f"{artifact_uri}/onehot_encoders.pkl"
            )
            with open(encoders_path, 'rb') as f:
                self.onehot_encoders = pickle.load(f)

            label_path = download_artifacts(
                artifact_uri=f"{artifact_uri}/label_encoder.json"
            )
            with open(label_path, 'r') as f:
                self.label_encoder = json.load(f)
        except Exception as exc:
            print(f"Error loading artifacts: {exc}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_and_save_model(self, X_df: pd.DataFrame, y: np.ndarray) -> Tuple[float, float, XGBClassifier]:
        """
        Train an XGBoost classifier on already-preprocessed features `X_df`
        and integer labels `y`. Returns (train_accuracy, test_accuracy, model).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.10, random_state=42, stratify=y
        )

        model = XGBClassifier(max_depth=4, n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        self.model = model
        train_accuracy, test_accuracy = self.get_accuracy(X_train, X_test, y_train, y_test)
        return float(train_accuracy), float(test_accuracy), model

    def get_accuracy(self, X_train, X_test, y_train, y_test) -> Tuple[float, float]:
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)
        return train_accuracy, test_accuracy

    def get_accuracy_full(self, X: pd.DataFrame, y: np.ndarray) -> float:
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy: ", accuracy)
        return accuracy

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocessing_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Replace missing values, one-hot encode nominal columns, scale feature
        columns, encode the target labels, and log preprocessing artifacts to MLflow.
        Returns (X_df, y).
        """
        df = df.copy()
        df = df.replace('?', np.nan)

        missing = [col for col in NOMINAL_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in training data: {missing}")

        self.fill_values_nominal = {
            col: df[col].mode(dropna=True)[0] for col in NOMINAL_COLUMNS
        }
        for col in NOMINAL_COLUMNS:
            df[col] = df[col].fillna(self.fill_values_nominal[col]).astype(str)

        y_raw = df[TARGET_COLUMN].astype(str).values
        X_df = df[NOMINAL_COLUMNS].copy()

        self.onehot_encoders = {}
        for col in NOMINAL_COLUMNS:
            encoder = _make_onehot()
            transformed = encoder.fit_transform(X_df[col].to_numpy().reshape(-1, 1))
            encoded_df = pd.DataFrame(
                transformed,
                columns=encoder.get_feature_names_out([col]),
                index=X_df.index,
            )
            X_df = pd.concat([X_df.drop(columns=[col]), encoded_df], axis=1)
            self.onehot_encoders[col] = encoder

        self.min_max_scaler_dict = {}
        for col in X_df.columns:
            scaler = MinMaxScaler()
            X_df[col] = scaler.fit_transform(X_df[[col]])
            self.min_max_scaler_dict[col] = scaler

        classes = sorted(pd.unique(y_raw))
        class_to_int = {c: i for i, c in enumerate(classes)}
        int_to_class = {i: c for c, i in class_to_int.items()}
        y = pd.Series(y_raw).map(class_to_int).values
        self.label_encoder = {'class_to_int': class_to_int, 'int_to_class': int_to_class}

        # Persist preprocessing artifacts in the active MLflow run
        mlflow.log_dict(self.fill_values_nominal, "fill_values_nominal.json")

        with open("min_max_scaler_dict.pkl", "wb") as f:
            pickle.dump(self.min_max_scaler_dict, f)
        mlflow.log_artifact("min_max_scaler_dict.pkl")

        with open("onehot_encoders.pkl", "wb") as f:
            pickle.dump(self.onehot_encoders, f)
        mlflow.log_artifact("onehot_encoders.pkl")

        with open("label_encoder.json", "w") as f:
            json.dump(self.label_encoder, f)
        mlflow.log_artifact("label_encoder.json")

        return X_df, y

    def preprocessing_pipeline_inference(self, sample_row: List) -> pd.DataFrame:
        """
        Prepare a single-row sample for inference using the stored artifacts.
        Expects the raw categorical inputs in the `NOMINAL_COLUMNS` order.
        """
        if any(attr is None for attr in [self.fill_values_nominal, self.onehot_encoders, self.min_max_scaler_dict]):
            raise RuntimeError("Artifacts not loaded. Train the model or load artifacts first.")

        sample_row = [np.nan if item in ['?', 'null', None] else item for item in sample_row]
        sample = pd.DataFrame([sample_row], columns=NOMINAL_COLUMNS)

        for col in NOMINAL_COLUMNS:
            fill = self.fill_values_nominal[col]
            sample[col] = sample[col].fillna(fill).astype(str)

        encoded = pd.DataFrame(index=sample.index)
        for col, encoder in self.onehot_encoders.items():
            transformed = encoder.transform(sample[col].to_numpy().reshape(-1, 1))
            encoded_df = pd.DataFrame(
                transformed,
                columns=encoder.get_feature_names_out([col]),
                index=sample.index,
            )
            encoded = pd.concat([encoded, encoded_df], axis=1)

        for col, scaler in self.min_max_scaler_dict.items():
            if col not in encoded.columns:
                encoded[col] = 0.0
            encoded[col] = scaler.transform(encoded[[col]])

        encoded = encoded[list(self.min_max_scaler_dict.keys())]
        return encoded

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, inference_row: List) -> str:
        if self.model is None:
            raise RuntimeError("No model is loaded. Train the model first.")

        X = self.preprocessing_pipeline_inference(inference_row)
        predictions = self.model.predict(X)

        if isinstance(self.label_encoder, dict) and 'int_to_class' in self.label_encoder:
            return self.label_encoder['int_to_class'][int(predictions[0])]

        return str(predictions[0])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def create_new_folder(folder: str) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
