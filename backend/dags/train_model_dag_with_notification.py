from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient
from pathlib import Path

# Define the MLflow client
mlflow.set_tracking_uri("http://127.0.0.1:5102")
client = MlflowClient()

# Path to the CSV file
BASE_DIR = Path(__file__).resolve().parents[1]
csv_file_path = BASE_DIR / "data" / "car_train.csv"

def train_model(**kwargs):
    # Call the train endpoint with the CSV file
    with open(csv_file_path, "rb") as f:
        files = {"file": ("car_train.csv", f)}
        response = requests.post("http://127.0.0.1:8080/model/train", files=files)
    
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
        history = client.get_metric_history(staging_version.run_id, "test_accuracy_split")
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
            return "skip_notification"
        client.transition_model_version_stage(
            name=registered_model,
            version=new_version,
            stage="Archived"
        )
        return "send_notification"

    client.transition_model_version_stage(
        name=registered_model,
        version=new_version,
        stage="Staging"
    )
    return "send_notification"

# Define the branching function
def branch_decision(**kwargs):
    return kwargs['ti'].xcom_pull(task_ids='train_and_compare_model') # ti: TaskInstance, xcom_pull: cross-communication - data sharing between tasks

# Define the Airflow DAG
with DAG(
    dag_id="daily_model_training_with_notification",
    start_date=datetime(2023, 11, 1),
    schedule_interval="0 2 * * *", # every day at 2am
    catchup=False,
) as dag:

    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=3,
        retry_delay=timedelta(minutes=5),
        provide_context=True,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=branch_decision,
        provide_context=True,
    )

    notification_task = EmailOperator(
        task_id="send_notification",
        to="change-me@example.com",
        subject="Model Accuracy Notification",
        html_content="The new model's accuracy is equal to or lower than the old model's accuracy.",
    )

    # Dummy task to mark the end if no notification is needed
    skip_notification = PythonOperator(
        task_id="skip_notification",
        python_callable=lambda: print("No notification sent."),
    )

    train_and_compare_task >> branch_task
    branch_task >> [notification_task, skip_notification]
