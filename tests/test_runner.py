#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import tempfile

import pytest

from nannyml._typing import ProblemType
from nannyml.chunk import DefaultChunker
from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_price_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.io import DatabaseWriter, RawFilesWriter
from nannyml.io.store import FilesystemStore
from nannyml.runner import run


@pytest.mark.slow
@pytest.mark.parametrize('timestamp_column_name', [None, 'timestamp'], ids=['without_timestamp', 'with_timestamp'])
def test_runner_executes_for_binary_classification_without_exceptions(timestamp_column_name):
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    analysis_with_targets = analysis.merge(analysis_targets, on='identifier')

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                reference_data=reference,
                analysis_data=analysis_with_targets,
                column_mapping={
                    'features': [
                        'distance_from_office',
                        'salary_range',
                        'gas_price_per_litre',
                        'public_transportation_cost',
                        'wfh_prev_workday',
                        'workday',
                        'tenure',
                    ],
                    'y_pred': 'y_pred',
                    'y_pred_proba': 'y_pred_proba',
                    'y_true': 'work_home_actual',
                    'timestamp': timestamp_column_name,
                },
                problem_type=ProblemType.CLASSIFICATION_BINARY,
                chunker=DefaultChunker(timestamp_column_name=timestamp_column_name),
                writer=RawFilesWriter(path=tmpdir, format='parquet'),
                store=FilesystemStore(root_path=f'{tmpdir}/cache'),
                run_in_console=False,
                ignore_errors=False,
            )
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize('timestamp_column_name', [None, 'timestamp'], ids=['without_timestamp', 'with_timestamp'])
def test_runner_executes_for_multiclass_classification_without_exceptions(timestamp_column_name):
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    analysis_with_targets = analysis.merge(analysis_targets, left_index=True, right_index=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                reference_data=reference,
                analysis_data=analysis_with_targets,
                column_mapping={
                    'features': [
                        'acq_channel',
                        'app_behavioral_score',
                        'requested_credit_limit',
                        'app_channel',
                        'credit_bureau_score',
                        'stated_income',
                        'is_customer',
                    ],
                    'y_pred': 'y_pred',
                    'y_pred_proba': {
                        'prepaid_card': 'y_pred_proba_prepaid_card',
                        'highstreet_card': 'y_pred_proba_highstreet_card',
                        'upmarket_card': 'y_pred_proba_upmarket_card',
                    },
                    'y_true': 'y_true',
                    'timestamp': timestamp_column_name,
                },
                problem_type=ProblemType.CLASSIFICATION_MULTICLASS,
                chunker=DefaultChunker(timestamp_column_name=timestamp_column_name),
                writer=RawFilesWriter(path=tmpdir, format='parquet'),
                store=FilesystemStore(root_path=f'{tmpdir}/cache'),
                run_in_console=False,
                ignore_errors=False,
            )
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize('timestamp_column_name', [None, 'timestamp'], ids=['without_timestamp', 'with_timestamp'])
def test_runner_executes_for_regression_without_exceptions(timestamp_column_name):
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    analysis_with_targets = analysis.join(analysis_targets)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                reference_data=reference,
                analysis_data=analysis_with_targets,
                column_mapping={
                    'features': [
                        'car_age',
                        'km_driven',
                        'price_new',
                        'accident_count',
                        'door_count',
                        'transmission',
                        'fuel',
                    ],
                    'y_pred': 'y_pred',
                    'y_true': 'y_true',
                    'timestamp': timestamp_column_name,
                },
                problem_type=ProblemType.REGRESSION,
                chunker=DefaultChunker(timestamp_column_name=timestamp_column_name),
                writer=RawFilesWriter(path=tmpdir, format='parquet'),
                store=FilesystemStore(root_path=f'{tmpdir}/cache'),
                run_in_console=False,
                ignore_errors=False,
            )
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.slow
def test_runner_executes_for_binary_classification_with_database_writer_without_exceptions():
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    analysis_with_targets = analysis.merge(analysis_targets, on='identifier')

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                reference_data=reference,
                analysis_data=analysis_with_targets,
                column_mapping={
                    'features': [
                        'distance_from_office',
                        'salary_range',
                        'gas_price_per_litre',
                        'public_transportation_cost',
                        'wfh_prev_workday',
                        'workday',
                        'tenure',
                    ],
                    'y_pred': 'y_pred',
                    'y_pred_proba': 'y_pred_proba',
                    'y_true': 'work_home_actual',
                    'timestamp': 'timestamp',
                },
                problem_type=ProblemType.CLASSIFICATION_BINARY,
                chunker=DefaultChunker(timestamp_column_name='timestamp'),
                writer=DatabaseWriter(connection_string='sqlite:///', model_name='test'),
                store=FilesystemStore(root_path=f'{tmpdir}/cache'),
                run_in_console=False,
                ignore_errors=False,
            )
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.slow
def test_runner_executes_for_multiclass_classification_with_database_writer_without_exceptions():
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    analysis_with_targets = analysis.merge(analysis_targets, left_index=True, right_index=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                reference_data=reference,
                analysis_data=analysis_with_targets,
                column_mapping={
                    'features': [
                        'acq_channel',
                        'app_behavioral_score',
                        'requested_credit_limit',
                        'app_channel',
                        'credit_bureau_score',
                        'stated_income',
                        'is_customer',
                    ],
                    'y_pred': 'y_pred',
                    'y_pred_proba': {
                        'prepaid_card': 'y_pred_proba_prepaid_card',
                        'highstreet_card': 'y_pred_proba_highstreet_card',
                        'upmarket_card': 'y_pred_proba_upmarket_card',
                    },
                    'y_true': 'y_true',
                    'timestamp': 'timestamp',
                },
                problem_type=ProblemType.CLASSIFICATION_MULTICLASS,
                chunker=DefaultChunker(timestamp_column_name='timestamp'),
                writer=DatabaseWriter(connection_string='sqlite:///', model_name='test'),
                store=FilesystemStore(root_path=f'{tmpdir}/cache'),
                run_in_console=False,
                ignore_errors=False,
            )
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.slow
def test_runner_executes_for_regression_with_database_writer_without_exceptions():
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    analysis_with_targets = analysis.join(analysis_targets)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            run(
                reference_data=reference,
                analysis_data=analysis_with_targets,
                column_mapping={
                    'features': [
                        'car_age',
                        'km_driven',
                        'price_new',
                        'accident_count',
                        'door_count',
                        'transmission',
                        'fuel',
                    ],
                    'y_pred': 'y_pred',
                    'y_true': 'y_true',
                    'timestamp': 'timestamp',
                },
                problem_type=ProblemType.REGRESSION,
                chunker=DefaultChunker(timestamp_column_name='timestamp'),
                writer=DatabaseWriter(connection_string='sqlite:///', model_name='test'),
                store=FilesystemStore(root_path=f'{tmpdir}/cache'),
                run_in_console=False,
                ignore_errors=False,
            )
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
