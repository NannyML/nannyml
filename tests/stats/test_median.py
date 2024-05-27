#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""
import numpy as np
import pytest
import pandas as pd

from typing import Tuple
from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsMedianCalculator


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    reference, monitored, _ = load_synthetic_car_loan_dataset()
    return reference.head(15_000), monitored.tail(5_000)


@pytest.fixture
def calculator_results(binary_classification_data):
    reference, monitored = binary_classification_data
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results = calc.calculate(data=monitored)
    return results


def test_stats_median_calculator_with_default_params_should_not_fail(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsMedianCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_median_calculator_should_not_fail_given_nan_values(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    reference.loc[1000:11000, 'car_value'] = np.NaN
    try:
        calc = SummaryStatsMedianCalculator(
            column_names=['car_value'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


@pytest.mark.parametrize(
    'column, expected_dir',
    [
        ('value', {
            'car_value': [21985.5, 21970.5, 21932.0, 44438.0],
            'debt_to_income_ratio': [0.6578, 0.6574, 0.6611, 0.661],
            'driver_tenure': [5.6424, 5.6498, 5.5153, 5.5879],
        }),
        ('sampling_error', {
            'car_value': [281.5532, 281.5532, 281.5532, 281.5532],
            'debt_to_income_ratio': [0.0023, 0.0023, 0.0023, 0.0023],
            'driver_tenure': [0.0388, 0.0388, 0.0388, 0.0388],
        }),
        ('upper_confidence_boundary', {
            'car_value': [22830.1596, 22815.1596, 22776.6596, 45282.6596],
            'debt_to_income_ratio': [0.6647, 0.6644, 0.668, 0.668],
            'driver_tenure': [5.7587, 5.7661, 5.6317, 5.7042],
        }),
        ('lower_confidence_boundary', {
            'car_value': [21140.8404, 21125.8404, 21087.3404, 43593.3404],
            'debt_to_income_ratio': [0.6509, 0.6505, 0.6541, 0.6541],
            'driver_tenure': [5.526, 5.5334, 5.399, 5.4716],
        }),
        ('upper_threshold', {
            'car_value': [22030.2647, 22030.2647, 22030.2647, 22030.2647],
            'debt_to_income_ratio': [0.6637, 0.6637, 0.6637, 0.6637],
            'driver_tenure': [5.7875, 5.7875, 5.7875, 5.7875],
        }),
        ('lower_threshold', {
            'car_value': [21895.0686, 21895.0686, 21895.0686, 21895.0686],
            'debt_to_income_ratio': [0.6539, 0.6539, 0.6539, 0.6539],
            'driver_tenure': [5.4174, 5.4174, 5.4174, 5.4174],
        }),
        ('alert', {
            'car_value': [False, False, False, True],
            'debt_to_income_ratio': [False, False, False, False],
            'driver_tenure': [False, False, False, False],
        }),
    ],
)
def test_stats_median_calculator_results(calculator_results, column, expected_dir):  # noqa: D103
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    eval_cols = [(col, column) for col in column_names]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame(expected_dir)
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(calculator_results.to_df()[eval_cols].round(4), expected)


def test_stats_median_calculator_returns_distinct_but_consistent_results_when_reused(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_median_calculator_returns_distinct_but_consistent_results_when_data_reused(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    reference2 = reference.copy(deep=True)
    monitored2 = monitored.copy(deep=True)
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference2)
    results = calc.calculate(data=monitored2)  # noqa: F841
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)
