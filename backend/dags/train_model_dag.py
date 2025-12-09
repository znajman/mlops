from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient
from pathlib import Path

# Define the MLflow client
mlflow.set_tracking_uri("http://127.0.0.1:5102")
client = MlflowClient()

# Path to the CSV file (kept relative to this DAG file for portability)
BASE_DIR = Path(__file__).resolve().parents[1]
csv_file_path = BASE_DIR / "data" / "car_train.csv"

def train_model():
    # Call the train endpoint with the CSV file
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('car_train.csv', f)}
        response = requests.post("http://127.0.0.1:8080/model/train", files=files)
    
    # Check if the request was successful
    if response.status_code != 200:
        data = response.json()
        raise Exception(f"Training failed: {data.get('error', 'Unknown error')}")
    
    data = response.json()
    new_accuracy = data["test_accuracy"]
    new_version = str(data["registered_version"])
    registered_model = data["registered_model"]

    latest_version_info = client.get_latest_versions(registered_model, stages=["Staging"])
    if latest_version_info:
        staging_version = latest_version_info[0]
        history = client.get_metric_history(
            staging_version.run_id,
            "test_accuracy_split"
        )
        staging_accuracy = history[-1].value if history else float('-inf')

        if new_accuracy >= staging_accuracy:
            client.transition_model_version_stage(
                name=registered_model,
                version=staging_version.version,
                stage="Archived"
            )
            client.transition_model_version_stage(
                name=registered_model,
                version=new_version,
                stage="Staging"
            )
        else:
            client.transition_model_version_stage(
                name=registered_model,
                version=new_version,
                stage="Archived"
            )
            return False
    else:
        client.transition_model_version_stage(
            name=registered_model,
            version=new_version,
            stage="Staging"
        )
    print("Model training and evaluation completed successfully.")
    return True

# Define the Airflow DAG
with DAG(
    dag_id="daily_model_training",
    start_date=datetime(2023, 11, 1),
    schedule_interval="0 2 * * *",  # Runs daily at 2 AM
    catchup=False,
) as dag:

    # Define a task that calls the train_model function
    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    # Define the email notification task
    notification_task = EmailOperator(
        task_id="send_notification",
        to="change-me@example.com",
        subject="Model Accuracy Notification",
        html_content="The new model's accuracy is equal to or lower than the old model's accuracy.",
    )

    # Set task dependencies
    train_and_compare_task >> notification_task

    # Run notification only if train_and_compare_task returns False
    notification_task.trigger_rule = 'all_done'
