from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os
from datetime import datetime

import pandas as pd
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from MLModel import MLModel
from constants import NOMINAL_COLUMNS, TARGET_COLUMN

# Configure MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5102")
experiment_name = "default_experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation')

client = MlflowClient()

# Try loading the latest staging model (artifacts optional during local runs)
try:
    obj_mlmodel = MLModel(client=client)
    if obj_mlmodel.model is None:
        print("⚠️  Warning: No 'Staging' model found yet. Training is still possible.")
except Exception as exc:
    print(f"⚠️  Warning: Could not load 'Staging' model. Training is still possible. Error: {exc}")
    obj_mlmodel = MLModel()

predict_model = api.model(
    'PredictModel',
    {
        'inference_row': fields.List(
            fields.Raw,
            required=True,
            description=f"A row of data for inference in this exact order: {NOMINAL_COLUMNS}"
        )
    }
)

file_upload = api.parser()
file_upload.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help='CSV file for training (must include headers and a target column named "class")'
)

ns = api.namespace('model', description='Model operations')


@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        """Train, log to MLflow, and register a model for the Car Evaluation dataset."""
        args = file_upload.parse_args()
        uploaded_file = args['file']

        if os.path.splitext(uploaded_file.filename)[1].lower() != '.csv':
            return {'error': 'Invalid file type'}, 400

        data_path = 'temp_car_train.csv'
        uploaded_file.save(data_path)

        try:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = "Car_Evaluation_Model"

            with mlflow.start_run(run_name=run_name) as run:
                raw_df = pd.read_csv(data_path)

                missing = [col for col in NOMINAL_COLUMNS + [TARGET_COLUMN] if col not in raw_df.columns]
                if missing:
                    raise ValueError(f"Missing columns in training CSV: {missing}")

                X_df, y = obj_mlmodel.preprocessing_pipeline(raw_df)

                mlflow.log_artifact(data_path, artifact_path="datasets")

                train_accuracy, test_accuracy, model = obj_mlmodel.train_and_save_model(X_df, y)
                mlflow.log_metric("train_accuracy", float(train_accuracy))
                mlflow.log_metric("test_accuracy_split", float(test_accuracy))

                input_example = X_df.iloc[:1]
                signature = infer_signature(X_df, y)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature
                )

                if os.path.isdir("artifacts"):
                    mlflow.log_artifacts("artifacts", artifact_path="artifacts")

                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

                os.remove(data_path)

                return {
                    'message': 'Model Trained and Registered Successfully',
                    'train_accuracy': float(train_accuracy),
                    'test_accuracy': float(test_accuracy),
                    'mlflow_run_id': run.info.run_id,
                    'registered_model': model_name,
                    'registered_version': registered_model_version.version
                }, 200

        except MlflowException as mfe:
            return {'message': 'MLflow Error', 'error': str(mfe)}, 500
        except Exception as exc:
            if os.path.exists(data_path):
                os.remove(data_path)
            return {'message': 'Internal Server Error', 'error': str(exc)}, 500


@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        """Predict car acceptability using the latest loaded model."""
        try:
            data = request.get_json()
            if 'inference_row' not in data:
                return {'error': 'No inference_row found'}, 400

            if obj_mlmodel.model is None:
                return {'error': "No model is loaded yet. Train a model first."}, 400

            infer_array = data['inference_row']

            run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                prediction = obj_mlmodel.predict(infer_array)
                mlflow.log_param("inference_input_order", NOMINAL_COLUMNS)
                mlflow.log_param("inference_input", infer_array)
                mlflow.log_param("inference_output", prediction)

            return {'message': 'Inference Successful', 'prediction': prediction}, 200
        except Exception as exc:
            return {'message': 'Internal Server Error', 'error': str(exc)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
