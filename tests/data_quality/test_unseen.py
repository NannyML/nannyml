#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import numpy as np
import pandas as pd
import pytest

from nannyml._typing import Result
from nannyml.data_quality.calculator import UnseenValuesCalculator

from nannyml.datasets import load_synthetic_car_loan_data_quality_dataset

from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture(scope="module")
def unseen_value_result() -> Result:
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()

    calc = UnseenValuesCalculator(
        column_names=[
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
        ],
    ).fit(reference)
    return calc.calculate(data=analysis)


def test_unseen_value_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = UnseenValuesCalculator(
            column_names=[
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()

def test_unseen_value_calculator_raises_invalidargumentsexception_on_numeric_columns():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    with pytest.raises(InvalidArgumentsException):
        calc = UnseenValuesCalculator(
            column_names=[
                'car_value',
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
        ).fit(reference)

def test_unseen_value_calculator_with_custom_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = UnseenValuesCalculator(
            column_names=[
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
            timestamp_column_name='timestamp',
            normalize=False
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_unseen_value_calculator_validates_column_names_list_elements():
    with pytest.raises(InvalidArgumentsException):
        calc = UnseenValuesCalculator(
            column_names=[
                'car_value',
                {'ab':1},
            ],
            timestamp_column_name='timestamp',
            normalize=False
        )


def test_unseen_value_calculator_fit_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    calc = UnseenValuesCalculator(
        column_names=[
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
        timestamp_column_name='timestamp',
        normalize=False
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(pd.DataFrame())


def test_unseen_value_calculator_calculate_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
        timestamp_column_name='timestamp',
        normalize=False
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(pd.DataFrame())


def test_unseen_value_calculator_fit_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103, F821
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
            'car_value',
            'missing_column',
        ],
        timestamp_column_name='timestamp',
        normalize=False
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(reference)


def test_unseen_value_calculator_calculate_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103, F821
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
        timestamp_column_name='timestamp',
        normalize=False
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(analysis.drop('size_of_downpayment', axis=1))


def test_whether_data_quality_metric_property_on_results_mv_rate(unseen_value_result):
    assert unseen_value_result.data_quality_metric == 'unseen_values_rate'


def test_whether_result_data_dataframe_has_proper_columns(unseen_value_result):
    cols = unseen_value_result.data.columns
    assert len(cols) == 7 + 2*7
    assert ('chunk', 'key') in cols
    assert ('chunk', 'chunk_index') in cols
    assert ('chunk', 'start_index') in cols
    assert ('chunk', 'start_date') in cols
    assert ('chunk', 'end_index') in cols
    assert ('chunk', 'end_date') in cols
    assert ('chunk', 'period') in cols
    assert ('repaid_loan_on_prev_car', 'value') in cols
    assert ('repaid_loan_on_prev_car', 'upper_threshold') in cols
    assert ('repaid_loan_on_prev_car', 'lower_threshold') in cols
    assert ('repaid_loan_on_prev_car', 'alert') in cols
    assert ('repaid_loan_on_prev_car', 'sampling_error') in cols
    assert ('repaid_loan_on_prev_car', 'upper_confidence_boundary') in cols
    assert ('repaid_loan_on_prev_car', 'lower_confidence_boundary') in cols
    assert ('size_of_downpayment', 'value') in cols
    assert ('size_of_downpayment', 'upper_threshold') in cols
    assert ('size_of_downpayment', 'lower_threshold') in cols
    assert ('size_of_downpayment', 'alert') in cols
    assert ('size_of_downpayment', 'sampling_error') in cols
    assert ('size_of_downpayment', 'upper_confidence_boundary') in cols
    assert ('size_of_downpayment', 'lower_confidence_boundary') in cols


def test_whether_data_quality_metric_property_on_results_mv_count():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
            ],
        normalize=False
    ).fit(reference)
    assert calc.calculate(data=analysis).data_quality_metric == 'unseen_values_count'


# def test_results_filtering_column_repaid_loan_on_prev_car_str(unseen_value_result):
#     try:
#         missing_value_result.filter(column_names='repaid_loan_on_prev_car')
#     except Exception:
#         pytest.fail()


# def test_results_filtering_columns_size_of_downpayment_list(unseen_value_result):
#     try:
#         missing_value_result.filter(column_names=[
#             'size_of_downpayment',
#         ])
#     except Exception:
#         pytest.fail()


# def test_results_car_value_values(missing_value_result):
 
#     res = missing_value_result.filter(column_names='car_value').to_df()
#     assert list(res[('car_value', 'value')]) == [0]*20


# def test_results_car_value_alerts(missing_value_result):
    
#     res = missing_value_result.filter(column_names='car_value').to_df()
#     assert list(res[('car_value', 'alert')]) == [False]*20


# def test_results_salary_range_values(missing_value_result):
    
#     res = missing_value_result.filter(column_names='salary_range').to_df()
#     assert list(res[('salary_range', 'value')]) == [
#         0.0998,
#         0.1004,
#         0.0912,
#         0.1018,
#         0.1014,
#         0.0966,
#         0.1028,
#         0.1108,
#         0.1018,
#         0.0934,
#         0.1004,
#         0.1064,
#         0.0982,
#         0.0966,
#         0.0984,
#         0.2188,
#         0.2214,
#         0.2178,
#         0.2264,
#         0.2156
#     ]


# def test_results_salary_range_alerts(missing_value_result):
#     res = missing_value_result.filter(column_names='salary_range').to_df()
#     assert list(res[('salary_range', 'alert')]) == [False]*15 + [True]*5


# def test_results_driver_tenure_values(missing_value_result): 
#     res = missing_value_result.filter(column_names='driver_tenure').to_df()
#     assert list(res[('driver_tenure', 'value')]) == [
#         0.1018,
#         0.1080,
#         0.0942,
#         0.099,
#         0.094,
#         0.1066,
#         0.0978,
#         0.1062,
#         0.0958,
#         0.0966,
#         0.1074,
#         0.1042,
#         0.094,
#         0.1,
#         0.0944,
#         0.2196,
#         0.2138,
#         0.2236,
#         0.2296,
#         0.2134
#     ]


# def test_results_driver_tenure_alerts(missing_value_result):
    
#     res = missing_value_result.filter(column_names='driver_tenure').to_df()
#     assert list(res[('driver_tenure', 'alert')]) == [False]*15 + [True]*5
