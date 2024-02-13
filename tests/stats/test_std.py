#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pytest
import pandas as pd
import numpy as np


from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsStdCalculator
from nannyml.chunk import SizeBasedChunker

# @pytest.fixture(scope="module")
# def status_sum_result() -> Result:
#     reference, analysis, _ = load_synthetic_car_loan_dataset()

#     calc = SummaryStatsStdCalculator(
#         column_names=[
#             'car_value',
#             'debt_to_income_ratio',
#             'driver_tenure'
#         ],
#     ).fit(reference)
#     return calc.calculate(data=analysis)


def test_stats_std_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_dataset()
    try:
        calc = SummaryStatsStdCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_stats_std_calculator_with_default_params_chunk_size_one():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_dataset()

    chunker = SizeBasedChunker(chunk_size=5_000, incomplete='keep')
    calc = SummaryStatsStdCalculator(
        column_names=['car_value'],
        chunker=chunker
    ).fit(reference)
    result = calc.calculate(data=analysis.head(5_001))
    expected = pd.DataFrame(
        {
            ('chunk', 'key'): ['[0:4999]', '[5000:5000]'],
            ('chunk', 'chunk_index'): [0,1],
            ('chunk', 'start_index'): [0,5000],
            ('chunk', 'end_index'): [4999,5000],
            ('chunk', 'start_date'): [None,None],
            ('chunk', 'end_date'): [None,None],
            ('chunk', 'period'): ['analysis','analysis'],
            ('car_value', 'value'): [20614.8926,np.nan],
            ('car_value', 'sampling_error'): [271.9917,np.nan],
            ('car_value', 'upper_confidence_boundary'): [21430.8679,np.nan],
            ('car_value', 'lower_confidence_boundary'): [19798.9174,np.nan],
            ('car_value', 'upper_threshold'): [20978.5658, 20978.5658],
            ('car_value', 'lower_threshold'): [19816.9091, 19816.9091],
            ('car_value', 'alert'): [False, False],
        }
    )
    pd.testing.assert_frame_equal(
        expected,
        result.filter(period='analysis').to_df().round(4)
    )
