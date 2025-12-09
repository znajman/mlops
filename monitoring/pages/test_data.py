import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import * 

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *

st.markdown("# Inference dataset and data drift report")
st.sidebar.markdown("# Browse a test dataset for inference")

browse_test_file = st.container()
show_imported_dataset = st.container()
evidently = st.container()
live_streaming = st.container()

def get_raw_data():
    # load the project train/test files (user requested these)
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    experiment_name = '1'
    
    return df_train, df_test

df_train, df_test = get_raw_data()

with browse_test_file:
    st.write("Choose the current dataset to compare with the reference (train)")
    current_choice = st.radio("Current dataset", ["Use test.csv (project file)", "Upload a CSV file"])
    uploaded_file = None
    imported_test_df = None
    if current_choice == "Upload a CSV file":
        uploaded_file = st.file_uploader("Please browse a test dataset for inference")
        if uploaded_file is not None:
            # Can be used wherever a "file-like" object is accepted:
            imported_test_df = pd.read_csv(uploaded_file)


# show imported dataset head when present
if imported_test_df is not None:
    with show_imported_dataset:
        st.header("Head of the dataset you imported")
        st.write(imported_test_df.head(20))

with evidently:
    st.header("Data drift report")
    run_report = st.button("Run data drift report", type="primary")

    reference = df_train.copy()

    # choose current dataset
    if current_choice == "Use test.csv (project file)":
        current = df_test.copy()
    else:
        if uploaded_file is None:
            st.info("No uploaded file yet. Choose 'Use test.csv' or upload a CSV file and click 'Run report'.")
            current = None
        else:
            current = imported_test_df.copy()

    # Run report only when we have a current dataset
    if run_report and current is None:
        st.warning("Select or upload a current dataset before running the report.")

    if current is not None and (run_report or st.session_state.get("_last_report_success")):
        # Align columns: Evidently requires matching columns. Use intersection and warn about dropped columns.
        common_cols = [c for c in reference.columns if c in current.columns]
        dropped_in_current = [c for c in reference.columns if c not in current.columns]

        if len(common_cols) == 0:
            st.error("No common columns between reference (train.csv) and current dataset. Cannot compute drift.")
            st.session_state["_last_report_success"] = False
        else:
            if dropped_in_current:
                st.warning(f"The following columns are present in reference but missing in current and will be ignored: {dropped_in_current}")

            ref_sub = reference[common_cols].copy()
            cur_sub = current[common_cols].copy()

            report = Report(metrics=[DataDriftPreset()])
            try:
                report.run(reference_data=ref_sub, current_data=cur_sub)
            except Exception as e:
                st.error(f"Failed to run Evidently report: {e}")
                st.session_state["_last_report_success"] = False
            else:
                # persist and embed the report
                try:
                    report.save_html('report.html')
                    with open('report.html', 'r', encoding='utf-8') as HtmlFile:
                        source_code = HtmlFile.read()
                    components.html(source_code, height=1200)
                    st.session_state["_last_report_success"] = True
                except Exception as e:
                    st.error(f"Failed to save or render report.html: {e}")
                    st.session_state["_last_report_success"] = False
       
    # Live streaming (safe: button-driven updates stored in session_state)
    import psutil

    if 'cpu_load' not in st.session_state:
        st.session_state['cpu_load'] = []

    with live_streaming:
        st.header("Streaming plot 'CPU load'")
        if st.button('Add reading'):
            st.session_state['cpu_load'].append(psutil.cpu_percent())

        # keep only last 100 readings to avoid growth
        cpu_list = st.session_state['cpu_load'][-100:]
        if len(cpu_list) == 0:
            st.info('No CPU readings yet — click "Add reading" to sample CPU usage')
        else:
            chart = st.line_chart(pd.Series(cpu_list))
 