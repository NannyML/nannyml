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
    return reference.head(15_000), monitored.tail(5_000)


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
        'car_value': [20403.1164, 20527.4427, 20114.8408, 22013.6951],
        'debt_to_income_ratio': [0.1541, 0.1576, 0.1558, 0.1559],
        'driver_tenure': [2.2973, 2.2971, 2.2981, 2.3083],
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
        'car_value': [270.6971, 270.6971, 270.6971, 270.6971],
        'debt_to_income_ratio': [0.0012, 0.0012, 0.0012, 0.0012],
        'driver_tenure': [0.0173, 0.0173, 0.0173, 0.0173],
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
        'car_value': [21215.2075, 21339.5339, 20926.9319, 22825.7862],
        'debt_to_income_ratio': [0.1578, 0.1613, 0.1595, 0.1596],
        'driver_tenure': [2.3492, 2.3491, 2.3501, 2.3602],
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
        'car_value': [19591.0252, 19715.3515, 19302.7496, 21201.6039],
        'debt_to_income_ratio': [0.1504, 0.1538, 0.152, 0.1521],
        'driver_tenure': [2.2453, 2.2452, 2.2462, 2.2564],
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
        'car_value': [20866.9261, 20866.9261, 20866.9261, 20866.9261],
        'debt_to_income_ratio': [0.1601, 0.1601, 0.1601, 0.1601],
        'driver_tenure': [2.2988, 2.2988, 2.2988, 2.2988],
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
        'car_value': [19830.0071, 19830.0071, 19830.0071, 19830.0071],
        'debt_to_income_ratio': [0.1515, 0.1515, 0.1515, 0.1515],
        'driver_tenure': [2.2962, 2.2962, 2.2962, 2.2962],
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
        'driver_tenure': [False, False, False, True],
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
    reference2 = reference.copy(deep=True)
    monitored2 = monitored.copy(deep=True)
    column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure']
    calc = SummaryStatsStdCalculator(
        column_names=column_names,
        chunk_size=5_000
    ).fit(reference2)
    results = calc.calculate(data=monitored2)
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)
