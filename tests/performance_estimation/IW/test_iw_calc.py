#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit testing for IW."""
import re
import typing
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from nannyml.datasets import (
    load_synthetic_car_loan_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.importance_weighting.iw import IW, DEFAULT_THRESHOLDS
from nannyml.performance_estimation.importance_weighting.results import Result
from nannyml.thresholds import ConstantThreshold


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_car_loan_dataset()
    return ref_df.head(15_000), ana_df.tail(5_000)


@pytest.fixture
def multiclass_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_multiclass_classification_dataset()
    return ref_df.head(15_000), ana_df.tail(5_000)


@pytest.fixture
def estimates(binary_classification_data) -> Result:  # noqa: D103
    reference, analysis = binary_classification_data
    estimator = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=['f1'],
        problem_type='classification_binary',
    )
    estimator.fit(reference)
    return estimator.estimate(analysis)  # type: ignore


@pytest.mark.parametrize('metrics, expected', [('roc_auc', ['roc_auc']), (['roc_auc', 'f1'], ['roc_auc', 'f1'])])
def test_iw_create_with_single_or_list_of_metrics(metrics, expected):
    sut = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=metrics,
        problem_type='classification_binary',
    )
    assert [metric.name for metric in sut.metrics] == expected


def test_iw_will_not_fail_on_car_loan_sample(binary_classification_data):  # noqa: D103
    reference, analysis = binary_classification_data
    try:
        estimator = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=['accuracy'],
            problem_type='classification_binary',
            chunk_size=5_000
        )
        estimator.fit(reference)
        _ = estimator.estimate(analysis)
    except Exception as exc:
        pytest.fail(f'unexpected exception was raised: {exc}')


def test_iw_raises_invalid_arguments_exception_when_giving_invalid_metric_value():  # noqa: D103
    with pytest.raises(InvalidArgumentsException, match="unknown metric key 'foo' given."):
        _ = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=['foo'],
            problem_type='classification_binary',
            chunk_size=5_000
        )


def test_iw_raises_invalid_arguments_exception_when_given_empty_metrics_list():  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=[],
            problem_type='classification_binary',
            chunk_size=5_000
        )


def test_iw_raises_invalid_arguments_exception_when_given_none_metrics_list():  # noqa: D103
    with pytest.raises(
        InvalidArgumentsException, match="no metrics provided. Please provide a non-empty list of metrics."
    ):
        _ = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=None,
            problem_type='classification_binary',
            chunk_size=5_000
        )


def test_iw_raises_value_error_when_business_value_matrix_wrong_shape():  # noqa: D103
    with pytest.raises(
        ValueError, match=re.escape("business_value_matrix must have shape (2,2), but got matrix of shape (4,)")
    ):
        _ = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=['business_value'],
            problem_type='classification_binary',
            business_value_matrix=[1, 2, 3, 4],
            chunk_size=5_000
        )


def test_cbpe_raises_value_error_when_business_value_matrix_wrong_type():  # noqa: D103
    with pytest.raises(
        ValueError, match="business_value_matrix must be a numpy array or a list, but got <class 'str'>"
    ):
        _ = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=['business_value'],
            problem_type='classification_binary',
            business_value_matrix='[1, 2, 3, 4]',
            chunk_size=5_000
        )


def test_cbpe_raises_value_error_when_business_value_matrix_not_given():  # noqa: D103
    with pytest.raises(ValueError, match="business_value_matrix must be provided for 'business_value' metric"):
        _ = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=['business_value'],
            problem_type='classification_binary',
            chunk_size=5_000
        )


def test_iw_raises_missing_metadata_exception_when_predictions_are_required_but_not_given(  # noqa: D103
    binary_classification_data,
):
    reference, _ = binary_classification_data
    estimator = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='predictions',
            y_pred_proba='y_pred_proba',
            metrics=['f1'],
            problem_type='classification_binary',
            chunk_size=5_000
        )  # requires predictions!
    with pytest.raises(InvalidArgumentsException, match='predictions'):
        estimator.fit(reference)


def test_iw_raises_missing_metadata_exception_when_predicted_probabilities_are_required_but_not_given(  # noqa: D103
    binary_classification_data,
):
    reference, _ = binary_classification_data

    estimator = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='probabilities',
            metrics=['f1'],
            problem_type='classification_binary',
            chunk_size=5_000
        )
    with pytest.raises(InvalidArgumentsException, match='probabilities'):
        estimator.fit(reference)


@pytest.mark.parametrize('metric', ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'roc_auc'])
def test_iw_runs_for_all_metrics(binary_classification_data, metric):  # noqa: D103
    reference, analysis = binary_classification_data
    try:
        estimator = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "car_value",
                "debt_to_income_ratio",
                "loan_length",
                "driver_tenure",
                "salary_range",
                "repaid_loan_on_prev_car",
                "size_of_downpayment"
            ],
            y_true='repaid',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            metrics=[metric],
            problem_type='classification_binary',
            chunk_size=5_000
        ).fit(reference)
        _ = estimator.estimate(pd.concat([reference, analysis]))
    except Exception as e:
        pytest.fail(f'an unexpected exception occurred: {e}')


def test_iw_results_plot_raises_invalid_arguments_exception_given_invalid_plot_kind(estimates):  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        estimates.plot(kind="foo", metric='roc_auc')


@pytest.mark.parametrize('metric', ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'])
def test_iw_for_binary_classification_does_not_fail_when_fitting_with_subset_of_reference_data(  # noqa: D103
    binary_classification_data, metric
):
    reference = binary_classification_data[0].loc[10000:, :]
    estimator = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
        problem_type='classification_binary',
        chunk_size=5_000
    )
    try:
        estimator.fit(reference_data=reference)
    except KeyError:
        pytest.fail(
            'fitting on subset resulted in KeyError => misaligned indices between data and stratified shuffle'
            'split results.'
        )


def reduce_confidence_bounds(monkeypatch, estimator, results):
    min_confidence = results.data[('roc_auc', 'lower_confidence_boundary')].min()
    max_confidence = results.data[('roc_auc', 'upper_confidence_boundary')].max()

    new_lower_bound = min_confidence + 0.001
    new_upper_bound = max_confidence - 0.001

    for metric in estimator.metrics:
        monkeypatch.setattr(metric, 'lower_threshold_value_limit', new_lower_bound)
        monkeypatch.setattr(metric, 'upper_threshold_value_limit', new_upper_bound)

    return estimator, new_lower_bound, new_upper_bound


def test_iw_for_binary_classification_does_not_output_confidence_bounds_outside_appropriate_interval(
    monkeypatch, binary_classification_data
):
    reference, analysis = binary_classification_data
    estimator = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=['roc_auc'],
        problem_type='classification_binary',
        chunk_size=5_000
    ).fit(reference)
    results = estimator.estimate(pd.concat([reference, analysis]))
    estimator, new_lower_bound, new_upper_bound = reduce_confidence_bounds(monkeypatch, estimator, results)
    # manually remove previous 'analysis' results
    results.data = results.data[results.data[('chunk', 'period')] == 'reference']
    results = estimator.estimate(analysis)
    sut = results.filter(period='analysis').to_df()
    assert all(sut.loc[:, ('roc_auc', 'lower_confidence_boundary')] >= new_lower_bound)
    assert all(sut.loc[:, ('roc_auc', 'upper_confidence_boundary')] <= new_upper_bound)


def test_iw_returns_distinct_but_consistent_results_when_reused(binary_classification_data):
    reference, analysis = binary_classification_data
    sut = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=['roc_auc'],
        problem_type='classification_binary',
        chunk_size=5_000
    )
    sut.fit(reference)
    result1 = sut.estimate(analysis)
    result2 = sut.estimate(analysis)
    # Checks two distinct results are returned. Previously there was a bug causing the previous result instance to be
    # modified on subsequent estimates.
    assert result1 is not result2
    pd.testing.assert_frame_equal(result1.to_df(), result2.to_df())


@pytest.mark.parametrize(
    'custom_thresholds',
    [
        {'roc_auc': ConstantThreshold(lower=1, upper=2)},
        {'roc_auc': ConstantThreshold(lower=1, upper=2), 'f1': ConstantThreshold(lower=1, upper=2)},
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
            'specificity': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
            'specificity': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
            'specificity': ConstantThreshold(lower=1, upper=2),
            'accuracy': ConstantThreshold(lower=1, upper=2),
        },
    ],
)
def test_iw_with_custom_thresholds(custom_thresholds):
    est = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=['roc_auc'],
        problem_type='classification_binary',
        thresholds=custom_thresholds,
        chunk_size=5_000
    )
    sut = est.thresholds
    expected_thresholds = DEFAULT_THRESHOLDS
    expected_thresholds.update(**custom_thresholds)
    assert sut == expected_thresholds


def test_iw_with_default_thresholds():
    est = IW(  # type: ignore
        timestamp_column_name='timestamp',
        feature_column_names=[
            "car_value",
            "debt_to_income_ratio",
            "loan_length",
            "driver_tenure",
            "salary_range",
            "repaid_loan_on_prev_car",
            "size_of_downpayment"
        ],
        y_true='repaid',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        metrics=['roc_auc'],
        problem_type='classification_binary',
        chunk_size=5_000
    )
    sut = est.thresholds
    assert sut == DEFAULT_THRESHOLDS


def test_iw_will_not_fail_on_mc_synthetic_sample(multiclass_classification_data):  # noqa: D103
    reference, analysis = multiclass_classification_data
    try:
        estimator = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "app_behavioral_score",
                "requested_credit_limit",
                "credit_bureau_score",
                "stated_income",
                "acq_channel",
                "app_channel",
                "is_customer"
            ],
            y_true='y_true',
            y_pred='y_pred',
            y_pred_proba={
                'prepaid_card': 'y_pred_proba_prepaid_card',
                'highstreet_card': 'y_pred_proba_highstreet_card',
                'upmarket_card': 'y_pred_proba_upmarket_card'},
            metrics=['accuracy'],
            problem_type='classification_multiclass',
            chunk_size=5_000
        )
        estimator.fit(reference)
        _ = estimator.estimate(analysis)
    except Exception as exc:
        pytest.fail(f'unexpected exception was raised: {exc}')


@pytest.mark.parametrize('metric', ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'specificity', 'confusion_matrix'])
def test_iw_runs_for_all_mc_metrics(multiclass_classification_data, metric):  # noqa: D103
    reference, analysis = multiclass_classification_data
    try:
        estimator = IW(  # type: ignore
            timestamp_column_name='timestamp',
            feature_column_names=[
                "app_behavioral_score",
                "requested_credit_limit",
                "credit_bureau_score",
                "stated_income",
                "acq_channel",
                "app_channel",
                "is_customer"
            ],
            y_true='y_true',
            y_pred='y_pred',
            y_pred_proba={
                'prepaid_card': 'y_pred_proba_prepaid_card',
                'highstreet_card': 'y_pred_proba_highstreet_card',
                'upmarket_card': 'y_pred_proba_upmarket_card'},
            metrics=[metric],
            problem_type='classification_multiclass',
            chunk_size=5_000,
            normalize_confusion_matrix='all'
        ).fit(reference)
        _ = estimator.estimate(pd.concat([reference, analysis]))
    except Exception as e:
        pytest.fail(f'an unexpected exception occurred: {e}')
