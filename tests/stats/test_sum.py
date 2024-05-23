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
    return reference.head(15_000), monitored.tail(5_000)


@pytest.fixture
def calculator_results(binary_classification_data):
    reference, monitored = binary_classification_data
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results = calc.calculate(data=monitored)
    return results


def test_stats_sum_calculator_with_default_params_should_not_fail(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsSumCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


@pytest.mark.parametrize(
    'column, expected_dir',
    [
        ('value', {
            'car_value': [148302466.0, 148088470.0, 147887986.0, 243531686.0],
            'debt_to_income_ratio': [2925.3933, 2913.6406, 2931.7188, 2925.0408],
            'driver_tenure': [23080.7173, 23084.6483, 22858.1008, 23014.0695],
        }),
        ('sampling_error', {
            'car_value': [1438811.819, 1438811.819, 1438811.819, 1438811.819],
            'debt_to_income_ratio': [11.0172, 11.0172, 11.0172, 11.0172],
            'driver_tenure': [162.4546, 162.4546, 162.4546, 162.4546],
        }),
        ('upper_confidence_boundary', {
            'car_value': [152618901.4569, 152404905.4569, 152204421.4569, 247848121.4569],
            'debt_to_income_ratio': [2958.4449, 2946.6922, 2964.7704, 2958.0924],
            'driver_tenure': [23568.081, 23572.012, 23345.4645, 23501.4332],
        }),
        ('lower_confidence_boundary', {
            'car_value': [143986030.5431, 143772034.5431, 143571550.5431, 239215250.5431],
            'debt_to_income_ratio': [2892.3417, 2880.589, 2898.6672, 2891.9892],
            'driver_tenure': [22593.3536, 22597.2846, 22370.7371, 22526.7058],
        }),
        ('upper_threshold', {
            'car_value': [148600696.1609, 148600696.1609, 148600696.1609, 148600696.1609],
            'debt_to_income_ratio': [2946.0554, 2946.0554, 2946.0554, 2946.0554],
            'driver_tenure': [23325.4655, 23325.4655, 23325.4655, 23325.4655],
        }),
        ('lower_threshold', {
            'car_value': [147585251.8391, 147585251.8391, 147585251.8391, 147585251.8391],
            'debt_to_income_ratio': [2901.113, 2901.113, 2901.113, 2901.113],
            'driver_tenure': [22690.1788, 22690.1788, 22690.1788, 22690.1788],
        }),
        ('alert', {
            'car_value': [False, False, False, True],
            'debt_to_income_ratio': [False, False, False, False],
            'driver_tenure': [False, False, False, False],
        }),
    ],
)
def test_stats_sum_calculator_results(calculator_results, column, expected_dir):  # noqa: D103
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    eval_cols = [(col, column) for col in column_names]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame(expected_dir)
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(calculator_results.to_df()[eval_cols].round(4), expected)


def test_stats_sum_calculator_returns_distinct_but_consistent_results_when_reused(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_sum_calculator_returns_distinct_but_consistent_results_when_data_reused(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    reference2 = reference.copy(deep=True)
    monitored2 = monitored.copy(deep=True)
    column_names = ['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsSumCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference2)
    results = calc.calculate(data=monitored2)  # noqa: F841
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)
