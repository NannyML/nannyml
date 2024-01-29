#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml._typing import ProblemType
from nannyml.datasets import (
    load_synthetic_car_loan_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.importance_weighting import IW, Result


@pytest.fixture(scope='module')
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_car_loan_dataset()
    return ref_df.head(15_000), ana_df.tail(5_000)


@pytest.fixture(scope='module')
def multiclass_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_multiclass_classification_dataset()
    return ref_df.head(15_000), ana_df.tail(5_000)


@pytest.fixture(scope='module')
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
        metrics=['roc_auc', 'f1', 'precision', 'recall', 'accuracy', 'confusion_matrix', 'business_value'],
        problem_type='classification_binary',
        business_value_matrix=np.array([[0, 1], [1, 0]]),
    )
    estimator.fit(reference)
    return estimator.estimate(analysis)  # type: ignore


@pytest.mark.parametrize(
    'metric, component_column_names',
    [
        ('roc_auc', ['roc_auc']),
        ('f1', ['f1']),
        ('precision', ['precision']),
        ('recall', ['recall']),
        ('accuracy', ['accuracy']),
        ('confusion_matrix', ['true_positive', 'false_positive', 'true_negative', 'false_negative']),
        ('business_value', ['business_value']),
    ],
)
def test_filter_on_metric_name_returns_only_matching_components(estimates: Result, metric, component_column_names):
    sut = estimates.filter(metrics=metric).to_df().columns

    for col in component_column_names:
        assert col in sut

    for col in sut.get_level_values(level=0):
        assert col in component_column_names + ['chunk']


@pytest.mark.parametrize(
    'metric, component_column_names',
    [
        ('roc_auc', ['roc_auc']),
        ('f1', ['f1']),
        ('precision', ['precision']),
        ('recall', ['recall']),
        ('accuracy', ['accuracy']),
        ('true_positive', ['true_positive']),
        ('business_value', ['business_value']),
    ],
)
def test_filter_on_metric_component_returns_only_matching_components(estimates: Result, metric, component_column_names):
    sut = estimates.filter(metrics=metric).to_df().columns

    for col in component_column_names:
        assert col in sut


@pytest.mark.parametrize(
    'metric, component_column_names',
    [
        ('true_positive', ['true_positive']),
        (['true_positive', 'false_positive'], ['true_positive', 'false_positive']),
        (['true_positive', 'false_positive', 'true_negative'], ['true_positive', 'false_positive', 'true_negative']),
        (['business_value', 'true_positive'], ['business_value', 'true_positive']),
    ],
)
def test_filter_on_multiple_metric_component_returns_matching_components(
    estimates: Result, metric, component_column_names
):
    sut = estimates.filter(metrics=metric).to_df().columns

    for col in component_column_names:
        assert col in sut


def test_filter_on_both_metric_and_component_names_returns_all_matching_components(estimates):
    sut = estimates.filter(metrics=['confusion_matrix', 'true_positive', 'business_value']).to_df().columns

    assert 'false_positive' in sut
    assert 'true_negative' in sut
    assert 'false_negative' in sut
    assert 'true_positive' in sut

    assert 'business_value' in sut


def test_filter_on_non_existing_metric_raises_invalid_arguments_exception(estimates):
    with pytest.raises(InvalidArgumentsException, match="invalid metric 'foo'"):
        _ = estimates.filter(metrics=['foo'])


def test_filter_on_non_calculated_metric_raises_invalid_arguments_exception(binary_classification_data):
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
    )
    estimator.fit(reference)
    estimates = estimator.estimate(analysis)  # type: ignore
    with pytest.raises(InvalidArgumentsException, match="no 'f1' in result, did you calculate it?"):
        _ = estimates.filter(metrics='f1')


@pytest.mark.parametrize(
    'estimator_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
    ],
    ids=[
        'performance_with_timestamp_without_reference',
        'performance_without_timestamp_without_reference',
        'performance_with_timestamp_with_reference',
        'performance_without_timestamp_with_reference',
    ],
)
def test_multiclass_classification_result_plots_raise_no_exceptions(estimator_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    est = IW(  # type: ignore
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
            metrics=['roc_auc', 'f1'],
            problem_type='classification_multiclass',
            chunk_size=5_000,
            normalize_confusion_matrix='all'
        ).fit(reference)
    sut = est.estimate(analysis)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'estimator_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
    ],
    ids=[
        'performance_with_timestamp_without_reference',
        'performance_without_timestamp_without_reference',
        'performance_with_timestamp_with_reference',
        'performance_without_timestamp_with_reference',
    ],
)
def test_binary_classification_result_plots_raise_no_exceptions(estimator_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_car_loan_dataset()
    est = IW(  # type: ignore
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
        **estimator_args
    ).fit(reference)
    sut = est.estimate(analysis)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
