#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pytest

from nannyml.datasets import load_synthetic_car_loan_dataset
from nannyml.stats import SummaryStatsAvgCalculator

# @pytest.fixture(scope="module")
# def status_sum_result() -> Result:
#     reference, monitored, _ = load_synthetic_car_loan_dataset()

#     calc = SummaryStatsSumCalculator(
#         column_names=[
#             'car_value',
#             'debt_to_income_ratio',
#             'driver_tenure'
#         ],
#     ).fit(reference)
#     return calc.calculate(data=monitored)


def test_stats_avg_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, monitored, _ = load_synthetic_car_loan_dataset()
    try:
        calc = SummaryStatsAvgCalculator(
            column_names=['car_value', 'debt_to_income_ratio', 'driver_tenure'],
        ).fit(reference)
        _ = calc.calculate(data=monitored)
    except Exception:
        pytest.fail()
