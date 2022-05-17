#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit testing for CBPE."""
import typing
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from nannyml.calibration import Calibrator, IsotonicCalibrator
from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata import BinaryClassificationMetadata, RegressionMetadata, extract_metadata
from nannyml.performance_estimation import CBPE
from nannyml.performance_estimation.base import PerformanceEstimatorResult


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].apply(lambda p: int(p >= 0.8))
    return ref_df, ana_df


@pytest.fixture
def metadata(data) -> BinaryClassificationMetadata:  # noqa: D103
    md = extract_metadata(data[0], exclude_columns=['identifier'], model_type='classification_binary')
    md.target_column_name = 'work_home_actual'
    return md


@pytest.fixture
def estimates(metadata, data) -> PerformanceEstimatorResult:  # noqa: D103
    reference, analysis = data
    estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])  # type: ignore
    estimator.fit(reference)
    return estimator.estimate(pd.concat([reference, analysis]))


def test_cbpe_will_calibrate_scores_when_needed(metadata, data):  # noqa: D103
    ref_df = data[0]

    sut = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    sut.fit(ref_df)

    assert sut.needs_calibration


def test_cbpe_will_not_calibrate_scores_when_not_needed(metadata, data):  # noqa: D103
    ref_df = data[0]
    # If predictions equal targets no calibration will be required
    ref_df[metadata.predicted_probability_column_name] = ref_df[metadata.target_column_name]

    sut = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    sut.fit(ref_df)

    assert sut.needs_calibration is False


def test_cbpe_will_not_fail_on_work_from_home_sample(metadata, data):  # noqa: D103
    reference, analysis = data
    try:
        estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])
        estimator.fit(reference)
        _ = estimator.estimate(analysis)
    except Exception as exc:
        pytest.fail(f'unexpected exception was raised: {exc}')


def test_cbpe_raises_invalid_arguments_exception_when_giving_invalid_metric_value(metadata):  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        _ = CBPE(model_metadata=metadata, metrics=['f1', 'foo'])


def test_cbpe_fitting_raises_invalid_arguments_exception_when_giving_invalid_metric_value(metadata, data):  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        estimator = CBPE(model_metadata=metadata, metrics=['f1'])
        estimator.metrics = ['foo']
        estimator.fit(data[0])


def test_cbpe_estimating_raises_invalid_arguments_exception_when_giving_invalid_metric_value(  # noqa: D103
    metadata, data
):
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        estimator = CBPE(model_metadata=metadata, metrics=['f1'])
        estimator.fit(data[0])
        estimator.metrics = ['foo']
        estimator.estimate(data[1])


def test_cbpe_raises_invalid_arguments_exception_when_given_empty_metrics_list(metadata):  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = CBPE(model_metadata=metadata, metrics=[])


def test_cbpe_raises_invalid_arguments_exception_when_given_none_metrics_list(metadata):  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = CBPE(model_metadata=metadata, metrics=None)


def test_cbpe_raises_missing_metadata_exception_when_predictions_are_required_but_not_given(  # noqa: D103
    metadata, data
):
    metadata.prediction_column_name = None

    reference, _ = data
    estimator = CBPE(model_metadata=metadata, metrics=['f1'])  # requires predictions!
    with pytest.raises(
        MissingMetadataException,
        match="missing value for 'prediction_column_name'. "
        "Please ensure predicted label values are specified and present "
        "in the sample.",
    ):
        estimator.fit(reference)


def test_cbpe_defaults_to_isotonic_calibrator_when_none_given(metadata):  # noqa: D103
    estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    assert isinstance(estimator.calibrator, IsotonicCalibrator)


def test_cbpe_uses_custom_calibrator_when_provided(metadata):  # noqa: D103
    class TestCalibrator(Calibrator):
        def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
            pass

        def calibrate(self, y_pred_proba: np.ndarray):
            pass

    estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'], calibrator=TestCalibrator())
    assert isinstance(estimator.calibrator, TestCalibrator)


def test_cbpe_uses_calibrator_to_calibrate_predicted_probabilities_when_needed(  # noqa: D103
    metadata, data, mocker: MockerFixture
):
    reference, analysis = data

    calibrator = IsotonicCalibrator()
    estimator = CBPE(  # type: ignore
        model_metadata=metadata, chunk_size=5000, metrics=['roc_auc'], calibrator=calibrator
    ).fit(reference)
    assert typing.cast(CBPE, estimator).needs_calibration

    spy = mocker.spy(calibrator, 'calibrate')

    estimator.estimate(analysis)
    spy.assert_called_once()


def test_cbpe_doesnt_use_calibrator_to_calibrate_predicted_probabilities_when_not_needed(  # noqa: D103
    metadata, data, mocker: MockerFixture
):
    reference, analysis = data

    calibrator = IsotonicCalibrator()
    estimator = CBPE(  # type: ignore
        model_metadata=metadata, chunk_size=5000, metrics=['roc_auc'], calibrator=calibrator
    ).fit(reference)

    typing.cast(CBPE, estimator).needs_calibration = False  # Override this to disable calibration

    spy = mocker.spy(calibrator, 'calibrate')
    estimator.estimate(analysis)
    spy.assert_not_called()


def test_cbpe_raises_missing_metadata_exception_when_predicted_probabilities_are_required_but_not_given(  # noqa: D103
    metadata, data
):
    reference, _ = data

    metadata.predicted_probability_column_name = None

    estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    with pytest.raises(
        MissingMetadataException,
        match="missing value for 'predicted_probability_column_name'. "
        "Please ensure predicted label values are specified and present "
        "in the sample.",
    ):
        estimator.fit(reference)


@pytest.mark.parametrize('metric', ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_cbpe_runs_for_all_metrics(metadata, data, metric):  # noqa: D103
    reference, analysis = data
    try:
        estimator = CBPE(model_metadata=metadata, chunk_size=5000, metrics=[metric]).fit(reference)
        _ = estimator.estimate(pd.concat([reference, analysis]))
    except Exception as e:
        pytest.fail(f'an unexpected exception occurred: {e}')


def test_cbpe_results_plot_raises_invalid_arguments_exception_given_invalid_plot_kind(estimates):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        estimates.plot(kind="foo")


def test_cbpe_results_plot_raises_invalid_arguments_exception_given_no_plot_kind(estimates):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        estimates.plot()


def test_cbpe_results_plot_raises_invalid_arguments_exception_given_no_metric_for_performance_plot(  # noqa: D103
    estimates,
):
    with pytest.raises(InvalidArgumentsException):
        estimates.plot(kind="performance")


def test_cbpe_results_plot_raises_invalid_arguments_exception_given_invalid_metric_for_performance_plot(  # noqa: D103
    estimates,
):
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        estimates.plot(kind="performance", metric="foo")


def test_cbpe_for_regression_metadata_raises_not_implemented_error():  # noqa: D103
    with pytest.raises(NotImplementedError):
        _ = CBPE(model_metadata=RegressionMetadata(), metrics=['f1'])
