# Vizsgafeladat - End-to-End MLOps (Car Evaluation)

Self-contained exam project that combines MLflow tracking + registry, Flask REST API, Docker image, Airflow DAGs, Evidently reports, and Streamlit dashboards using the UCI Car Evaluation dataset.

## Folder layout
- `backend/` – notebooks, API + model code, data, Airflow DAGs, Dockerfile, tests.
- `monitoring/` – Streamlit apps, Evidently scripts/reports, train/test CSVs.

## Notebooks
Find them in `vizsgafeladat/backend/notebooks/`:
- `Car_Data_Preparation.ipynb` and `Car_Data_Understanding.ipynb` for the initial dataset exploration/preparation walkthroughs.

## Backend quickstart (MLflow + REST API)
Use Python 3.10/3.11 (or 3.9) for the smoothest install; Python 3.12 removes `distutils`, so if you must stay on 3.12 install `distutils` from your package manager first.
```bash
cd ~/cubix/vizsgafeladat/backend
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt mlflow==2.17.0
# If you’re using a different Python version, swap constraints-3.11 for your version (e.g., constraints-3.10.txt)
pip install "apache-airflow==2.6.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.6.3/constraints-3.11.txt"

# Terminal 1: MLflow tracking UI
mlflow server --host 0.0.0.0 --port 5102 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Terminal 2: REST API (Flask-RESTX)
python app.py    # http://127.0.0.1:8080 with Swagger UI
```

### Train and register a model
```bash
curl -X POST -F "file=@data/car_train.csv" \
  http://127.0.0.1:8080/model/train
```
- Logs metrics/artifacts to MLflow and registers `Car_Evaluation_Model`.
- Stage promotion: new model moves to `Staging` if its `test_accuracy` is >= current staging model; otherwise it is archived.

### Predict with the latest staging model
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"inference_row":["vhigh","vhigh","2","2","small","low"]}' \
  http://127.0.0.1:8080/model/predict
```
Input order must be: `buying, maint, doors, persons, lug_boot, safety`.

### Tests
```bash
pytest tests/test_train_inference.py
```

## Dockerized API + MLflow
From `vizsgafeladat/backend`:
```bash
docker build -t car-mlops .
docker run -p 8080:8080 -p 5102:5102 car-mlops
```
The container starts MLflow (file store at `/app/mlruns`) and the Flask API together.

## Airflow automation
- DAGs are in `backend/dags/`:
  - `train_model_dag_without_notification.py` (no email)
  - `train_model_dag_with_notification.py` (branching + email)
  - `train_model_dag.py` (simple + email)
- Update the `to="change-me@example.com"` address before using the email-enabled DAGs.
- Copy a DAG into your Airflow `dags/` folder (or point `dags_folder` to `vizsgafeladat/backend/dags`).
- Ensure the API (`:8080`) and MLflow (`:5102`) services are running, then start Airflow:
  ```bash
  airflow db init
  airflow webserver --port 8081
  airflow scheduler
  ```
- Note: default username and password is admin
- Trigger the DAG (`daily_model_training` or `daily_model_training_with_notification`); it posts `data/car_train.csv` to `/model/train`, compares accuracy to the staging model, and transitions registry stages accordingly.

## Monitoring, Evidently, Streamlit
```bash
cd vizsgafeladat/monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Rebuild Evidently artifacts
python generate_monitoring_reports.py   # writes report.html, tests.html, suite.html

# Launch Streamlit (multi-page picks up pages/test_data.py)
streamlit run monitor_with_streamlit_train_data.py   # http://localhost:8501
```
- Page 1: shows `train.csv`/`test.csv` samples and value counts.
- Page 2 (`pages/test_data.py`): upload or reuse `test.csv`, run Evidently drift, and render the HTML inline; includes a small CPU “live” chart demo.
