#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import pytest

from nannyml._typing import Result
from nannyml.chunk import PeriodBasedChunker, SizeBasedChunker
from nannyml.data_quality.calculator import MissingValueCalculator

from nannyml.datasets import load_synthetic_car_loan_data_quality_dataset

from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture(scope="module")
def missing_value_result() -> Result:
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()

    calc = MissingValueCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure'
        ],
    ).fit(reference)
    return calc.calculate(data=analysis)


def test_missing_value_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = MissingValueCalculator(
            column_names=[
                'car_value',
                'salary_range',
                'debt_to_income_ratio',
                'loan_length',
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
                'driver_tenure'
            ],
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_missing_value_calculator_with_custom_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = MissingValueCalculator(
            column_names=[
                'car_value',
                'salary_range',
                'debt_to_income_ratio',
                'loan_length',
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
                'driver_tenure'
            ],
            timestamp_column_name='timestamp',
            normalize=False
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_missing_value_calculator_fit_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    calc = MissingValueCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure'
        ],
        timestamp_column_name='timestamp',
        normalize=False
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(pd.DataFrame())


def test_missing_value_calculator_calculate_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValueCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure'
        ],
        timestamp_column_name='timestamp',
        normalize=False
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(pd.DataFrame())


def test_missing_value_calculator_fit_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103, F821
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValueCalculator(
        column_names=[
            'car_value',
            'missing_column',
        ],
        timestamp_column_name='timestamp',
        normalize=False
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(reference)


def test_missing_value_calculator_calculate_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103, F821
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValueCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure'
        ],
        timestamp_column_name='timestamp',
        normalize=False
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(analysis.drop('driver_tenure', axis=1))


def test_whether_data_quality_metric_property_on_results_mv_rate(missing_value_result):
    assert missing_value_result.data_quality_metric == 'missing_value_rate'


def test_whether_data_quality_metric_property_on_results_mv_count():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValueCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure'
        ],
        normalize=False
    ).fit(reference)
    assert calc.calculate(data=analysis).data_quality_metric == 'missing_value_count'


def test_results_filtering_column_str(missing_value_result):
    try:
        missing_value_result.filter(column_names='car_value')
    except Exception:
        pytest.fail()


def test_results_filtering_columns_list(missing_value_result):
    try:
        missing_value_result.filter(column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
        ])
    except Exception:
        pytest.fail()


# def test_missing_value_calculator_should_contain_chunk_details_and_results_properties(  # noqa: D103
#     sample_drift_data_with_nans,
# ):
#     reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()

#     calc = MissingValueCalculator(
#         column_names=[
#                 'car_value',
#                 'salary_range',
#                 'debt_to_income_ratio',
#                 'loan_length',
#                 'repaid_loan_on_prev_car',
#                 'size_of_downpayment',
#                 'driver_tenure'
#             ],
#         timestamp_column_name='timestamp',
#     )
#     calc.fit(data_ref)
#     results = calc.calculate(data=data_anl)

#     assert results.data_quality_metric == 'missing_value_rate'
#     # print("debug1")
#     # print(results.data)
#     sut = results.data.columns
#     assert len(sut) == 7 + 4*7
#     assert ('chunk', 'key') in sut
#     assert ('chunk', 'chunk_index') in sut
#     assert ('chunk', 'start_index') in sut
#     assert ('chunk', 'start_date') in sut
#     assert ('chunk', 'end_index') in sut
#     assert ('chunk', 'end_date') in sut
#     assert ('chunk', 'period') in sut
#     assert ('f1', 'value') in sut
#     assert ('f1', 'upper_threshold') in sut
#     assert ('f1', 'lower_threshold') in sut
#     assert ('f1', 'alert') in sut
#     assert ('f1', 'sampling_error') in sut
#     assert ('f1', 'upper_confidence_boundary') in sut
#     assert ('f1', 'lower_confidence_boundary') in sut
#     assert ('f2', 'value') in sut
#     assert ('f2', 'upper_threshold') in sut
#     assert ('f2', 'lower_threshold') in sut
#     assert ('f2', 'alert') in sut
#     assert ('f2', 'sampling_error') in sut
#     assert ('f2', 'upper_confidence_boundary') in sut
#     assert ('f2', 'lower_confidence_boundary') in sut
#     assert ('f3', 'value') in sut
#     assert ('f3', 'upper_threshold') in sut
#     assert ('f3', 'lower_threshold') in sut
#     assert ('f3', 'alert') in sut
#     assert ('f3', 'sampling_error') in sut
#     assert ('f3', 'upper_confidence_boundary') in sut
#     assert ('f3', 'lower_confidence_boundary') in sut
#     assert ('f4', 'value') in sut
#     assert ('f4', 'upper_threshold') in sut
#     assert ('f4', 'lower_threshold') in sut
#     assert ('f4', 'alert') in sut
#     assert ('f4', 'sampling_error') in sut
#     assert ('f4', 'upper_confidence_boundary') in sut
#     assert ('f4', 'lower_confidence_boundary') in sut

#     assert results.data.loc[:, ('f1', 'value')].tolist() == [
#         0.1259920634920635,
#         0.11210317460317461,
#         0.09325396825396826,
#         0.10317460317460317,
#         0.10515873015873016,
#         0.10317460317460317,
#         0.1111111111111111,
#         0.10912698412698413,
#         0.10317460317460317,
#         0.11011904761904762,
#         0.125,
#         0.1111111111111111,
#         0.10515873015873016,
#         0.12202380952380952,
#         0.11011904761904762,
#         0.2361111111111111,
#         0.21329365079365079,
#         0.2371031746031746,
#         0.2390873015873016,
#         0.22420634920634921
#     ]

#     assert results.data.loc[:, ('f1', 'alert')].tolist() == [False]*15 + [True]*5

#     assert results.data.loc[:, ('f4', 'value')].tolist() == [
#         0.12568328786892544,
#         0.11179439898003656,
#         0.0929451926308302,
#         0.10286582755146512,
#         0.10484995453559211,
#         0.10286582755146512,
#         0.11080233548797305,
#         0.10881820850384608,
#         0.10286582755146512,
#         0.10981027199590956,
#         0.12469122437686195,
#         0.11080233548797305,
#         0.10484995453559211,
#         0.12171503390067147,
#         0.10981027199590956,
#         0.23580233548797305,
#         0.21298487517051273,
#         0.23679439898003654,
#         0.23877852596416355,
#         0.22389757358321116
#     ]

#     assert results.data.loc[:, ('f4', 'alert')].tolist() == [False]*15 + [True]*5
