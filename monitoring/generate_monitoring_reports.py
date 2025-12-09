"""Utility script to rebuild Evidently monitoring artifacts based on train/test CSVs."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.test_preset import NoTargetPerformanceTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnsType,
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDriftedColumns,
    TestNumberOfDuplicatedColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfRowsWithMissingValues,
)

BASE_DIR = Path(__file__).parent


def load_datasets(base_dir: Path = BASE_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = base_dir / "train.csv"
    test_path = base_dir / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")
    reference = pd.read_csv(train_path)
    current = pd.read_csv(test_path)
    return reference, current


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def build_data_quality_suite(reference: pd.DataFrame, current: pd.DataFrame) -> TestSuite:
    tests = TestSuite(
        tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ]
    )
    tests.run(reference_data=reference, current_data=current)
    return tests


def build_performance_suite(reference: pd.DataFrame, current: pd.DataFrame) -> TestSuite:
    suite = TestSuite(tests=[NoTargetPerformanceTestPreset()])
    suite.run(reference_data=reference, current_data=current)
    return suite


def save_artifacts(report: Report, tests: TestSuite, suite: TestSuite, base_dir: Path = BASE_DIR) -> None:
    report_path = base_dir / "report.html"
    tests_path = base_dir / "tests.html"
    suite_path = base_dir / "suite.html"
    report.save_html(str(report_path))
    tests.save_html(str(tests_path))
    suite.save_html(str(suite_path))


def main() -> None:
    reference, current = load_datasets()
    drift_report = build_drift_report(reference, current)
    data_quality_suite = build_data_quality_suite(reference, current)
    performance_suite = build_performance_suite(reference, current)
    save_artifacts(drift_report, data_quality_suite, performance_suite)
    print("Saved Evidently artifacts: report.html, tests.html, suite.html")


if __name__ == "__main__":
    main()
