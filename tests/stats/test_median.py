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
    return reference.head(15_000), monitored.head(5_000)


def test_stats_median_calculator_with_default_params_should_not_fail(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsMedianCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_median_calculator_should_not_fail_given_nan_values(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    reference.loc[1000:11000, 'car_value'] = np.NaN
    try:
        calc = SummaryStatsMedianCalculator(
            column_names=['car_value'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_median_calculator_results(binary_classification_data):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results = calc.calculate(data=monitored)
    eval_cols = [('car_value', 'value'), ('debt_to_income_ratio', 'value'), ('driver_tenure', 'value'),]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [21985.5, 21970.5, 21932.0, 22009.5],
        'debt_to_income_ratio': [0.6578, 0.6574, 0.6611, 0.6632],
        'driver_tenure': [5.6424, 5.6498, 5.5153, 5.4783],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)


def test_stats_median_calculator_returns_distinct_but_consistent_results_when_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_median_calculator_returns_distinct_but_consistent_results_when_data_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)

    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsMedianCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results2 = calc.calculate(data=monitored)

    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())
