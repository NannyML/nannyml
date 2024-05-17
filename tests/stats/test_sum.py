#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pytest
import pandas as pd

from typing import Tuple
from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsSumCalculator

@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    reference, monitored, _ = load_synthetic_car_loan_dataset()
    return reference.head(15_000), monitored.head(5_000)


def test_stats_sum_calculator_with_default_params_should_not_fail(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsSumCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_sum_calculator_results(binary_classification_data):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results = calc.calculate(data=monitored)
    eval_cols = [('car_value', 'value'), ('debt_to_income_ratio', 'value'), ('driver_tenure', 'value'),]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [148302466.0, 148088470.0, 147887986.0, 149805851.0],
        'debt_to_income_ratio': [2925.3933, 2913.6406, 2931.7188, 2947.695],
        'driver_tenure': [23080.7173, 23084.6483, 22858.1008, 22560.8987],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)


def test_stats_sum_calculator_returns_distinct_but_consistent_results_when_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_sum_calculator_returns_distinct_but_consistent_results_when_data_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)

    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results2 = calc.calculate(data=monitored)

    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())
