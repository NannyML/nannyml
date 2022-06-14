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
from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata import (
    BinaryClassificationMetadata,
    MulticlassClassificationMetadata,
    RegressionMetadata,
    extract_metadata,
)
from nannyml.performance_estimation import CBPE
from nannyml.performance_estimation.base import PerformanceEstimatorResult


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].apply(lambda p: int(p >= 0.8))
    return ref_df, ana_df


@pytest.fixture
def binary_classification_metadata(binary_classification_data) -> BinaryClassificationMetadata:  # noqa: D103
    md = extract_metadata(
        binary_classification_data[0], exclude_columns=['identifier'], model_type='classification_binary'
    )
    md.target_column_name = 'work_home_actual'
    return md


@pytest.fixture
def multiclass_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_multiclass_classification_dataset()
    return ref_df, ana_df


@pytest.fixture
def multiclass_classification_metadata(
    multiclass_classification_data,
) -> MulticlassClassificationMetadata:  # noqa: D103
    md = extract_metadata(
        multiclass_classification_data[0], exclude_columns=['identifier'], model_type='classification_multiclass'
    )
    return md


@pytest.fixture
def estimates(binary_classification_metadata, binary_classification_data) -> PerformanceEstimatorResult:  # noqa: D103
    reference, analysis = binary_classification_data
    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])  # type: ignore
    estimator.fit(reference)
    return estimator.estimate(pd.concat([reference, analysis]))


def test_cbpe_will_calibrate_scores_when_needed(
    binary_classification_metadata, binary_classification_data
):  # noqa: D103
    ref_df = binary_classification_data[0]

    sut = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
    sut.fit(ref_df)

    assert sut.needs_calibration


def test_cbpe_will_not_calibrate_scores_when_not_needed(
    binary_classification_metadata, binary_classification_data
):  # noqa: D103
    ref_df = binary_classification_data[0]
    # If predictions equal targets no calibration will be required
    ref_df[binary_classification_metadata.predicted_probability_column_name] = ref_df[
        binary_classification_metadata.target_column_name
    ]

    sut = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
    sut.fit(ref_df)

    assert sut.needs_calibration is False


def test_cbpe_will_not_fail_on_work_from_home_sample(
    binary_classification_metadata, binary_classification_data
):  # noqa: D103
    reference, analysis = binary_classification_data
    try:
        estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
        estimator.fit(reference)
        _ = estimator.estimate(analysis)
    except Exception as exc:
        pytest.fail(f'unexpected exception was raised: {exc}')


def test_cbpe_raises_invalid_arguments_exception_when_giving_invalid_metric_value(
    binary_classification_metadata,
):  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        _ = CBPE(model_metadata=binary_classification_metadata, metrics=['f1', 'foo'])


def test_cbpe_fitting_raises_invalid_arguments_exception_when_giving_invalid_metric_value(
    binary_classification_metadata, binary_classification_data
):  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['f1'])
        estimator.metrics = ['foo']
        estimator.fit(binary_classification_data[0])


def test_cbpe_estimating_raises_invalid_arguments_exception_when_giving_invalid_metric_value(  # noqa: D103
    binary_classification_metadata, binary_classification_data
):
    with pytest.raises(InvalidArgumentsException, match="unknown 'metric' value: 'foo'."):
        estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['f1'])
        estimator.fit(binary_classification_data[0])
        estimator.metrics = ['foo']
        estimator.estimate(binary_classification_data[1])


def test_cbpe_raises_invalid_arguments_exception_when_given_empty_metrics_list(
    binary_classification_metadata,
):  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = CBPE(model_metadata=binary_classification_metadata, metrics=[])


def test_cbpe_raises_invalid_arguments_exception_when_given_none_metrics_list(
    binary_classification_metadata,
):  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = CBPE(model_metadata=binary_classification_metadata, metrics=None)


def test_cbpe_raises_missing_metadata_exception_when_predictions_are_required_but_not_given(  # noqa: D103
    binary_classification_metadata, binary_classification_data
):
    binary_classification_metadata.prediction_column_name = None

    reference, _ = binary_classification_data
    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['f1'])  # requires predictions!
    with pytest.raises(MissingMetadataException, match='prediction_column_name'):
        estimator.fit(reference)


def test_cbpe_defaults_to_isotonic_calibrator_when_none_given(binary_classification_metadata):  # noqa: D103
    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
    assert isinstance(estimator.calibrator, IsotonicCalibrator)


def test_cbpe_uses_custom_calibrator_when_provided(binary_classification_metadata):  # noqa: D103
    class TestCalibrator(Calibrator):
        def fit(self, y_pred_proba: np.ndarray, y_true: np.ndarray):
            pass

        def calibrate(self, y_pred_proba: np.ndarray):
            pass

    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'], calibrator=TestCalibrator())
    assert isinstance(estimator.calibrator, TestCalibrator)


def test_cbpe_uses_calibrator_to_calibrate_predicted_probabilities_when_needed(  # noqa: D103
    binary_classification_metadata, binary_classification_data, mocker: MockerFixture
):
    reference, analysis = binary_classification_data

    calibrator = IsotonicCalibrator()
    estimator = CBPE(  # type: ignore
        model_metadata=binary_classification_metadata, chunk_size=5000, metrics=['roc_auc'], calibrator=calibrator
    ).fit(reference)
    assert typing.cast(CBPE, estimator).needs_calibration

    spy = mocker.spy(calibrator, 'calibrate')

    estimator.estimate(analysis)
    spy.assert_called_once()


def test_cbpe_doesnt_use_calibrator_to_calibrate_predicted_probabilities_when_not_needed(  # noqa: D103
    binary_classification_metadata, binary_classification_data, mocker: MockerFixture
):
    reference, analysis = binary_classification_data

    calibrator = IsotonicCalibrator()
    estimator = CBPE(  # type: ignore
        model_metadata=binary_classification_metadata, chunk_size=5000, metrics=['roc_auc'], calibrator=calibrator
    ).fit(reference)

    typing.cast(CBPE, estimator).needs_calibration = False  # Override this to disable calibration

    spy = mocker.spy(calibrator, 'calibrate')
    estimator.estimate(analysis)
    spy.assert_not_called()


def test_cbpe_raises_missing_metadata_exception_when_predicted_probabilities_are_required_but_not_given(  # noqa: D103
    binary_classification_metadata, binary_classification_data
):
    reference, _ = binary_classification_data

    binary_classification_metadata.predicted_probability_column_name = None

    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
    with pytest.raises(MissingMetadataException):
        estimator.fit(reference)


@pytest.mark.parametrize('metric', ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_cbpe_runs_for_all_metrics(binary_classification_metadata, binary_classification_data, metric):  # noqa: D103
    reference, analysis = binary_classification_data
    try:
        estimator = CBPE(model_metadata=binary_classification_metadata, chunk_size=5000, metrics=[metric]).fit(
            reference
        )
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


def test_cbpe_for_binary_classification_does_not_fail_when_fitting_with_subset_of_reference_data(  # noqa: D103
    binary_classification_metadata, binary_classification_data
):
    reference = binary_classification_data[0].loc[40000:, :]
    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['f1'])
    try:
        estimator.fit(reference_data=reference)
    except KeyError:
        pytest.fail(
            'fitting on subset resulted in KeyError => misaligned indices between data and stratified shuffle'
            'split results.'
        )


def reduce_confidence_bounds(monkeypatch, ref_df, ana_df, metadata):
    estimator = CBPE(model_metadata=metadata, metrics=['roc_auc'])
    estimator.fit(reference_data=ref_df)
    results = estimator.estimate(pd.concat([ref_df, ana_df]))
    min_confidence = results.data['lower_confidence_roc_auc'].min()
    max_confidence = results.data['upper_confidence_roc_auc'].max()

    new_lower_bound = min_confidence + 0.001
    new_upper_bound = max_confidence - 0.001
    monkeypatch.setattr(estimator, 'confidence_lower_bound', new_lower_bound)
    monkeypatch.setattr(estimator, 'confidence_upper_bound', new_upper_bound)

    return estimator, new_lower_bound, new_upper_bound


def test_cbpe_for_binary_classification_does_not_output_confidence_bounds_outside_appropriate_interval(
    monkeypatch, binary_classification_metadata, binary_classification_data
):
    reference, analysis = binary_classification_data
    metadata = binary_classification_metadata
    estimator, new_lower_bound, new_upper_bound = reduce_confidence_bounds(monkeypatch, reference, analysis, metadata)
    results = estimator.estimate(analysis)
    assert all(results.data['lower_confidence_roc_auc'] >= new_lower_bound)
    assert all(results.data['upper_confidence_roc_auc'] <= new_upper_bound)


def test_cbpe_for_multiclass_classification_does_not_output_confidence_bounds_outside_appropriate_interval(
    monkeypatch, multiclass_classification_metadata, multiclass_classification_data
):
    reference, analysis = multiclass_classification_data
    metadata = multiclass_classification_metadata
    estimator, new_lower_bound, new_upper_bound = reduce_confidence_bounds(monkeypatch, reference, analysis, metadata)
    results = estimator.estimate(analysis)
    assert all(results.data['lower_confidence_roc_auc'] >= new_lower_bound)
    assert all(results.data['upper_confidence_roc_auc'] <= new_upper_bound)


@pytest.mark.parametrize(
    'properties',
    [
        ['target_column_name'],
        ['predicted_probability_column_name'],
        ['target_column_name', 'predicted_probability_column_name'],
    ],
)
def test_cbpe_for_binary_raises_missing_metadata_exception_when_fitting_without_required_metadata(  # noqa: D103
    binary_classification_data, binary_classification_metadata, properties
):
    reference, analysis = binary_classification_data

    for prop in properties:
        setattr(binary_classification_metadata, prop, None)

    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
    with pytest.raises(MissingMetadataException):
        estimator.fit(reference_data=reference)


@pytest.mark.parametrize('properties', [['predicted_probability_column_name']])
def test_cbpe_for_binary_raises_missing_metadata_exception_when_estimating_without_required_metadata(  # noqa: D103
    binary_classification_data, binary_classification_metadata, properties
):
    reference, analysis = binary_classification_data

    for prop in properties:
        setattr(binary_classification_metadata, prop, None)

    estimator = CBPE(model_metadata=binary_classification_metadata, metrics=['roc_auc'])
    with pytest.raises(MissingMetadataException):
        estimator.fit(reference_data=reference)
        estimator.estimate(data=analysis)


@pytest.mark.parametrize(
    'properties',
    [
        ['target_column_name'],
        ['predicted_probabilities_column_names'],
        ['target_column_name', 'predicted_probabilities_column_names'],
    ],
)
def test_cbpe_for_multiclass_raises_missing_metadata_exception_when_fitting_without_required_metadata(  # noqa: D103
    multiclass_classification_data, multiclass_classification_metadata, properties
):
    reference, analysis = multiclass_classification_data

    for prop in properties:
        setattr(multiclass_classification_metadata, prop, None)

    estimator = CBPE(model_metadata=multiclass_classification_metadata, metrics=['roc_auc'])
    with pytest.raises(MissingMetadataException):
        estimator.fit(reference_data=reference)


@pytest.mark.parametrize('properties', [['predicted_probabilities_column_names']])
def test_cbpe_for_multiclass_raises_missing_metadata_exception_when_estimating_without_required_metadata(  # noqa: D103
    multiclass_classification_data, multiclass_classification_metadata, properties
):
    reference, analysis = multiclass_classification_data

    for prop in properties:
        setattr(multiclass_classification_metadata, prop, None)

    estimator = CBPE(model_metadata=multiclass_classification_metadata, metrics=['roc_auc'])
    with pytest.raises(MissingMetadataException):
        estimator.fit(reference_data=reference)
        estimator.estimate(data=analysis)
