#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pytest
import pandas as pd

from typing import Tuple
from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsAvgCalculator


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    reference, monitored, _ = load_synthetic_car_loan_dataset()
    return reference.head(15_000), monitored.head(5_000)


def test_stats_avg_calculator_with_default_params_should_not_fail(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsAvgCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_avg_calculator_results(binary_classification_data):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsAvgCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results = calc.calculate(data=monitored)
    eval_cols = [('car_value', 'value'), ('debt_to_income_ratio', 'value'), ('driver_tenure', 'value'),]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [29660.4932, 29617.694, 29577.5972, 29961.1702],
        'debt_to_income_ratio': [0.5851, 0.5827, 0.5863, 0.5895],
        'driver_tenure': [4.6161, 4.6169, 4.5716, 4.5122],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)


def test_stats_avg_calculator_returns_distinct_but_consistent_results_when_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsAvgCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_avg_calculator_returns_distinct_but_consistent_results_when_data_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsAvgCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)

    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsAvgCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results2 = calc.calculate(data=monitored)

    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())
