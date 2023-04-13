#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import pytest

from nannyml._typing import Result
from nannyml.stats import SimpleStatsSumCalculator

from nannyml.datasets import load_synthetic_car_loan_dataset

from nannyml.exceptions import InvalidArgumentsException


# @pytest.fixture(scope="module")
# def status_sum_result() -> Result:
#     reference, analysis, _ = load_synthetic_car_loan_dataset()

#     calc = SimpleStatsSumCalculator(
#         column_names=[
#             'car_value',
#             'debt_to_income_ratio',
#             'driver_tenure'
#         ],
#     ).fit(reference)
#     return calc.calculate(data=analysis)


def test_stats_sum_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_dataset()
    try:
        calc = SimpleStatsSumCalculator(
            column_names=[
                'car_value',
                'debt_to_income_ratio',
                'driver_tenure'
            ],
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()