#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import tempfile

import pytest
from pytest_lazyfixture import lazy_fixture

from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_price_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.drift.multivariate.data_reconstruction import DataReconstructionDriftCalculator
from nannyml.drift.univariate import UnivariateDriftCalculator
from nannyml.io import DatabaseWriter, PickleFileWriter, RawFilesWriter
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.direct_loss_estimation import DLE


@pytest.fixture(scope='module')
def univariate_drift_for_binary_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_binary_classification_dataset()
    calc = UnivariateDriftCalculator(
        column_names=[col for col in reference_df if col not in ['timestamp', 'work_home_actual']],
        timestamp_column_name='timestamp',
    ).fit(reference_df)
    result = calc.calculate(analysis_df)
    return result


@pytest.fixture(scope='module')
def univariate_drift_for_multiclass_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_multiclass_classification_dataset()
    calc = UnivariateDriftCalculator(
        column_names=[col for col in reference_df if col not in ['timestamp', 'y_true']],
        timestamp_column_name='timestamp',
    ).fit(reference_df)
    result = calc.calculate(analysis_df)
    return result


@pytest.fixture(scope='module')
def univariate_drift_for_regression_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_car_price_dataset()
    calc = UnivariateDriftCalculator(
        column_names=[col for col in reference_df if col not in ['timestamp', 'y_true']],
        timestamp_column_name='timestamp',
    ).fit(reference_df)
    result = calc.calculate(analysis_df)
    return result


@pytest.fixture(scope='module')
def data_reconstruction_drift_for_binary_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_binary_classification_dataset()
    calc = DataReconstructionDriftCalculator(
        column_names=[
            col for col in reference_df if col not in ['timestamp', 'y_pred', 'y_pred_proba', 'work_home_actual']
        ],
        timestamp_column_name='timestamp',
    ).fit(reference_df)
    result = calc.calculate(analysis_df)
    return result


@pytest.fixture(scope='module')
def data_reconstruction_drift_for_multiclass_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_multiclass_classification_dataset()
    calc = DataReconstructionDriftCalculator(
        column_names=[
            col
            for col in reference_df
            if col
            not in [
                'timestamp',
                'y_pred',
                'y_pred_proba_upmarket_card',
                'y_pred_proba_highstreet_card',
                'y_pred_proba_prepaid_card',
                'y_true',
            ]
        ],
        timestamp_column_name='timestamp',
    ).fit(reference_df)
    result = calc.calculate(analysis_df)
    return result


@pytest.fixture(scope='module')
def data_reconstruction_drift_for_regression_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_car_price_dataset()
    calc = DataReconstructionDriftCalculator(
        column_names=[col for col in reference_df if col not in ['timestamp', 'y_pred', 'y_true']],
        timestamp_column_name='timestamp',
    ).fit(reference_df)
    result = calc.calculate(analysis_df)
    return result


@pytest.fixture(scope='module')
def realized_performance_for_binary_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_binary_classification_dataset()
    calc = PerformanceCalculator(
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        problem_type='classification_binary',
        timestamp_column_name='timestamp',
        metrics=['roc_auc', 'f1'],
    ).fit(reference_df)
    result = calc.calculate(analysis_df.merge(analysis_targets_df, on='identifier'))
    return result


@pytest.fixture(scope='module')
def realized_performance_for_multiclass_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_multiclass_classification_dataset()
    calc = PerformanceCalculator(
        y_pred='y_pred',
        y_pred_proba={
            'upmarket_card': 'y_pred_proba_upmarket_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'prepaid_card': 'y_pred_proba_prepaid_card',
        },
        y_true='y_true',
        timestamp_column_name='timestamp',
        problem_type='classification_multiclass',
        metrics=['roc_auc', 'f1'],
    ).fit(reference_df)
    result = calc.calculate(analysis_df.merge(analysis_targets_df, left_index=True, right_index=True))
    return result


@pytest.fixture(scope='module')
def realized_performance_for_regression_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_car_price_dataset()
    calc = PerformanceCalculator(
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        problem_type='regression',
        metrics=['mae', 'mape'],
    ).fit(reference_df)
    result = calc.calculate(analysis_df.join(analysis_targets_df))
    return result


@pytest.fixture(scope='module')
def cbpe_estimated_performance_for_binary_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_binary_classification_dataset()
    calc = CBPE(
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        problem_type='classification_binary',
        timestamp_column_name='timestamp',
        metrics=['roc_auc', 'f1'],
    ).fit(reference_df)
    result = calc.estimate(analysis_df.merge(analysis_targets_df, on='identifier'))
    return result


@pytest.fixture(scope='module')
def cbpe_estimated_performance_for_multiclass_classification_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_multiclass_classification_dataset()
    calc = CBPE(
        y_pred='y_pred',
        y_pred_proba={
            'upmarket_card': 'y_pred_proba_upmarket_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'prepaid_card': 'y_pred_proba_prepaid_card',
        },
        y_true='y_true',
        timestamp_column_name='timestamp',
        problem_type='classification_multiclass',
        metrics=['roc_auc', 'f1'],
    ).fit(reference_df)
    result = calc.estimate(analysis_df.merge(analysis_targets_df, left_index=True, right_index=True))
    return result


@pytest.fixture(scope='module')
def dle_estimated_performance_for_regression_result():
    reference_df, analysis_df, analysis_targets_df = load_synthetic_car_price_dataset()
    calc = DLE(
        feature_column_names=[col for col in reference_df if col not in ['timestamp', 'y_pred', 'y_true']],
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        metrics=['mae', 'mape'],
    ).fit(reference_df)
    result = calc.estimate(analysis_df.join(analysis_targets_df))
    return result


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('univariate_drift_for_binary_classification_result'),
        lazy_fixture('univariate_drift_for_multiclass_classification_result'),
        lazy_fixture('univariate_drift_for_regression_result'),
        lazy_fixture('data_reconstruction_drift_for_binary_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_multiclass_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_regression_result'),
        lazy_fixture('realized_performance_for_binary_classification_result'),
        lazy_fixture('realized_performance_for_multiclass_classification_result'),
        lazy_fixture('realized_performance_for_regression_result'),
        lazy_fixture('cbpe_estimated_performance_for_binary_classification_result'),
        lazy_fixture('cbpe_estimated_performance_for_multiclass_classification_result'),
        lazy_fixture('dle_estimated_performance_for_regression_result'),
    ],
)
def test_raw_files_writer_raises_no_exceptions_when_writing_to_parquet(result):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RawFilesWriter(path=tmpdir)
            writer.write(result, filename='export.pq')
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('univariate_drift_for_binary_classification_result'),
        lazy_fixture('univariate_drift_for_multiclass_classification_result'),
        lazy_fixture('univariate_drift_for_regression_result'),
        lazy_fixture('data_reconstruction_drift_for_binary_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_multiclass_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_regression_result'),
        lazy_fixture('realized_performance_for_binary_classification_result'),
        lazy_fixture('realized_performance_for_multiclass_classification_result'),
        lazy_fixture('realized_performance_for_regression_result'),
        lazy_fixture('cbpe_estimated_performance_for_binary_classification_result'),
        lazy_fixture('cbpe_estimated_performance_for_multiclass_classification_result'),
        lazy_fixture('dle_estimated_performance_for_regression_result'),
    ],
)
def test_raw_files_writer_raises_no_exceptions_when_writing_to_csv(result):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RawFilesWriter(path=tmpdir)
            writer.write(result, filename='export.csv', format='csv')
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('univariate_drift_for_binary_classification_result'),
        lazy_fixture('univariate_drift_for_multiclass_classification_result'),
        lazy_fixture('univariate_drift_for_regression_result'),
        lazy_fixture('data_reconstruction_drift_for_binary_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_multiclass_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_regression_result'),
        lazy_fixture('realized_performance_for_binary_classification_result'),
        lazy_fixture('realized_performance_for_multiclass_classification_result'),
        lazy_fixture('realized_performance_for_regression_result'),
        lazy_fixture('cbpe_estimated_performance_for_binary_classification_result'),
        lazy_fixture('cbpe_estimated_performance_for_multiclass_classification_result'),
        lazy_fixture('dle_estimated_performance_for_regression_result'),
    ],
)
def test_database_writer_raises_no_exceptions_when_writing(result):
    try:
        writer = DatabaseWriter(connection_string='sqlite:///', model_name='test')
        writer.write(result)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'result',
    [
        lazy_fixture('univariate_drift_for_binary_classification_result'),
        lazy_fixture('univariate_drift_for_multiclass_classification_result'),
        lazy_fixture('univariate_drift_for_regression_result'),
        lazy_fixture('data_reconstruction_drift_for_binary_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_multiclass_classification_result'),
        lazy_fixture('data_reconstruction_drift_for_regression_result'),
        lazy_fixture('realized_performance_for_binary_classification_result'),
        lazy_fixture('realized_performance_for_multiclass_classification_result'),
        lazy_fixture('realized_performance_for_regression_result'),
        lazy_fixture('cbpe_estimated_performance_for_binary_classification_result'),
        lazy_fixture('cbpe_estimated_performance_for_multiclass_classification_result'),
        lazy_fixture('dle_estimated_performance_for_regression_result'),
    ],
)
def test_pickle_file_writer_raises_no_exceptions_when_writing(result):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PickleFileWriter(path=tmpdir)
            writer.write(result, filename='export.pkl')
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
