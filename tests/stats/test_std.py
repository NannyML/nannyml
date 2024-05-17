#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import pytest

from typing import Tuple
from nannyml.chunk import SizeBasedChunker
from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsStdCalculator

@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    reference, monitored, _ = load_synthetic_car_loan_dataset()
    return reference.head(15_000), monitored.head(5_000)


def test_stats_std_calculator_with_default_params_should_not_fail(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    try:
        calc = SummaryStatsStdCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()


def test_stats_std_calculator_with_default_params_chunk_size_one():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_dataset()

    chunker = SizeBasedChunker(chunk_size=5_000, incomplete='keep')
    calc = SummaryStatsStdCalculator(column_names=['car_value'], chunker=chunker).fit(reference)
    result = calc.calculate(data=analysis.head(5_001))
    expected = pd.DataFrame(
        {
            ('chunk', 'key'): ['[0:4999]', '[5000:5000]'],
            ('chunk', 'chunk_index'): [0, 1],
            ('chunk', 'start_index'): [0, 5000],
            ('chunk', 'end_index'): [4999, 5000],
            ('chunk', 'start_date'): [None, None],
            ('chunk', 'end_date'): [None, None],
            ('chunk', 'period'): ['analysis', 'analysis'],
            ('car_value', 'value'): [20614.8926, np.nan],
            ('car_value', 'sampling_error'): [271.9917, np.nan],
            ('car_value', 'upper_confidence_boundary'): [21430.8679, np.nan],
            ('car_value', 'lower_confidence_boundary'): [19798.9174, np.nan],
            ('car_value', 'upper_threshold'): [20978.5658, 20978.5658],
            ('car_value', 'lower_threshold'): [19816.9091, 19816.9091],
            ('car_value', 'alert'): [False, True],
        }
    )
    pd.testing.assert_frame_equal(expected, result.filter(period='analysis').to_df().round(4))


def test_stats_std_calculator_results(binary_classification_data):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsStdCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results = calc.calculate(data=monitored)
    eval_cols = [('car_value', 'value'), ('debt_to_income_ratio', 'value'), ('driver_tenure', 'value'),]
    exp_cols = pd.MultiIndex.from_tuples(eval_cols)
    expected = pd.DataFrame({
        'car_value': [20403.1164, 20527.4427, 20114.8408, 20614.8926],
        'debt_to_income_ratio': [0.1541, 0.1576, 0.1558, 0.1524],
        'driver_tenure': [2.2973, 2.2971, 2.2981, 2.3396],
    })
    expected.columns = exp_cols
    pd.testing.assert_frame_equal(results.to_df()[eval_cols].round(4), expected)


def test_stats_std_calculator_returns_distinct_but_consistent_results_when_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsStdCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)
    results2 = calc.calculate(data=monitored)
    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())


def test_stats_std_calculator_returns_distinct_but_consistent_results_when_data_reused(
    binary_classification_data
):  # noqa: D103
    reference, monitored = binary_classification_data
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsStdCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results1 = calc.calculate(data=monitored)

    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsStdCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference)
    results2 = calc.calculate(data=monitored)

    assert results1 is not results2
    pd.testing.assert_frame_equal(results1.to_df(), results2.to_df())
