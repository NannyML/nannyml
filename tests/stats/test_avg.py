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
    return reference.head(15_000), monitored.tail(5_000)


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
        'car_value': [29660.4932, 29617.694, 29577.5972, 48706.3372],
        'debt_to_income_ratio': [0.5851, 0.5827, 0.5863, 0.585],
        'driver_tenure': [4.6161, 4.6169, 4.5716, 4.6028],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [
        ('car_value', 'sampling_error'),
        ('debt_to_income_ratio', 'sampling_error'),
        ('driver_tenure', 'sampling_error'),
    ]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [287.7624, 287.7624, 287.7624, 287.7624],
        'debt_to_income_ratio': [0.0022, 0.0022, 0.0022, 0.0022],
        'driver_tenure': [0.0325, 0.0325, 0.0325, 0.0325],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [
        ('car_value', 'upper_confidence_boundary'),
        ('debt_to_income_ratio', 'upper_confidence_boundary'),
        ('driver_tenure', 'upper_confidence_boundary'),
    ]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [30523.7803, 30480.9811, 30440.8843, 49569.6243],
        'debt_to_income_ratio': [0.5917, 0.5893, 0.593, 0.5916],
        'driver_tenure': [4.7136, 4.7144, 4.6691, 4.7003],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [
        ('car_value', 'lower_confidence_boundary'),
        ('debt_to_income_ratio', 'lower_confidence_boundary'),
        ('driver_tenure', 'lower_confidence_boundary'),
    ]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [28797.2061, 28754.4069, 28714.3101, 47843.0501],
        'debt_to_income_ratio': [0.5785, 0.5761, 0.5797, 0.5784],
        'driver_tenure': [4.5187, 4.5195, 4.4741, 4.5053],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [
        ('car_value', 'upper_threshold'),
        ('debt_to_income_ratio', 'upper_threshold'),
        ('driver_tenure', 'upper_threshold'),
    ]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [29720.1392, 29720.1392, 29720.1392, 29720.1392],
        'debt_to_income_ratio': [0.5892, 0.5892, 0.5892, 0.5892],
        'driver_tenure': [4.6651, 4.6651, 4.6651, 4.6651],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [
        ('car_value', 'lower_threshold'),
        ('debt_to_income_ratio', 'lower_threshold'),
        ('driver_tenure', 'lower_threshold'),
    ]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [29517.0504, 29517.0504, 29517.0504, 29517.0504],
        'debt_to_income_ratio': [0.5802, 0.5802, 0.5802, 0.5802],
        'driver_tenure': [4.538, 4.538, 4.538, 4.538],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [
        ('car_value', 'alert'),
        ('debt_to_income_ratio', 'alert'),
        ('driver_tenure', 'alert'),
    ]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [False, False, False, True],
        'debt_to_income_ratio': [False, False, False, False],
        'driver_tenure': [False, False, False, False],
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
    reference2 = reference.copy(deep=True)
    monitored2 = monitored.copy(deep=True)
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsAvgCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference2)
    results = calc.calculate(data=monitored2)
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)
