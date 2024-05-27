#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pytest
import pandas as pd

from typing import Tuple
from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsRowCountCalculator


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    reference, monitored, _ = load_synthetic_car_loan_dataset()
    return reference.head(15_000), monitored.tail(5_000)


@pytest.fixture
def calculator_results(binary_classification_data):
    reference, monitored = binary_classification_data
    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference)
    results = calc.calculate(data=monitored)
    return results


def test_stats_count_calculator_with_default_params_should_not_fail(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsRowCountCalculator().fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


@pytest.mark.parametrize(
    'column, expected_dir',
    [
        ('value', {
            'count': [5120, 4625, 5119, 136, 294, 4706],
        }),
        ('upper_threshold', {
            'count': [10038.8619, 10038.8619, 10038.8619, 10038.8619, 10038.8619, 10038.8619],
        }),
        ('lower_threshold', {
            'count': [None, None, None, None, None, None],
        }),
        ('alert', {
            'count': [False, False, False, False, False, False],
        }),
    ],
)
def test_stats_count_calculator_results(calculator_results, column, expected_dir):  # noqa: D103
    eval_cols = [('rows_count', column)]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame(expected_dir)
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(calculator_results.to_df()[eval_cols].round(4), expected)


def test_stats_count_calculator_returns_distinct_but_consistent_results_when_reused(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_count_calculator_returns_distinct_but_consistent_results_when_data_reused(  # noqa: D103
    binary_classification_data
):
    reference, monitored = binary_classification_data
    reference2 = reference.copy(deep=True)
    monitored2 = monitored.copy(deep=True)
    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference2)
    results = calc.calculate(data=monitored2)  # noqa: F841
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)
