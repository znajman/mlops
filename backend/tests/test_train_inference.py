import numpy as np
import pandas as pd
import sys
from pathlib import Path

import mlflow

# Add the project root to sys.path so we can import MLModel
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

from MLModel import MLModel
from constants import NOMINAL_COLUMNS


def test_prediction_accuracy(tmp_path):
    data_path = project_root / "data" / "car_train.csv"
    df = pd.read_csv(data_path)

    mlflow.set_tracking_uri(f"file:{tmp_path.as_posix()}")

    obj_mlmodel = MLModel()

    with mlflow.start_run():
        X_df, y = obj_mlmodel.preprocessing_pipeline(df)
        obj_mlmodel.train_and_save_model(X_df, y)

    train_accuracy_full = np.round(
        obj_mlmodel.get_accuracy_full(X_df, y), 2
    )

    encoded_rows = []
    for _, row in df.iterrows():
        inference_row = [row[col] for col in NOMINAL_COLUMNS]
        encoded_rows.append(obj_mlmodel.preprocessing_pipeline_inference(inference_row))

    inference_df = pd.concat(encoded_rows, ignore_index=True)
    inference_accuracy_full = np.round(
        obj_mlmodel.get_accuracy_full(inference_df, y), 2
    )

    print(train_accuracy_full, inference_accuracy_full)
    assert train_accuracy_full == inference_accuracy_full, 'Inference prediction accuracy is not as expected'
