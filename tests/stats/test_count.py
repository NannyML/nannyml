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


def test_stats_count_calculator_with_default_params_should_not_fail(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsRowCountCalculator().fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_count_calculator_results(binary_classification_data):  # noqa: D103
    reference, monitored = binary_classification_data
    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference)
    results = calc.calculate(data=monitored)
    eval_cols = [('rows_count', 'value')]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'count': [5120, 4625, 5119, 136, 294, 4706],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [('rows_count', 'upper_threshold')]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'count': [10038.8619, 10038.8619, 10038.8619, 10038.8619, 10038.8619, 10038.8619],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [('rows_count', 'lower_threshold')]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'count': [None, None, None, None, None, None,],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)

    eval_cols = [('rows_count', 'alert')]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'count': [False, False, False, False, False, False],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)


def test_stats_count_calculator_returns_distinct_but_consistent_results_when_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_count_calculator_returns_distinct_but_consistent_results_when_data_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference)
    results1 = calc.calculate(data=monitored)

    calc = SummaryStatsRowCountCalculator(
        chunk_period='M',
        timestamp_column_name='timestamp'
    ).fit(reference)
    results2 = calc.calculate(data=monitored)

    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())
