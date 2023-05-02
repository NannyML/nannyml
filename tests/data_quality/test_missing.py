#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pandas as pd
import pytest

from nannyml._typing import Result
from nannyml.data_quality import MissingValuesCalculator
from nannyml.datasets import load_synthetic_car_loan_data_quality_dataset
from nannyml.exceptions import InvalidArgumentsException


@pytest.fixture(scope="module")
def missing_value_result() -> Result:
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()

    calc = MissingValuesCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure',
        ],
    ).fit(reference)
    return calc.calculate(data=analysis)


def test_missing_value_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = MissingValuesCalculator(
            column_names=[
                'car_value',
                'salary_range',
                'debt_to_income_ratio',
                'loan_length',
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
                'driver_tenure',
            ],
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_missing_value_calculator_with_custom_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = MissingValuesCalculator(
            column_names=[
                'car_value',
                'salary_range',
                'debt_to_income_ratio',
                'loan_length',
                'repaid_loan_on_prev_car',
                'size_of_downpayment',
                'driver_tenure',
            ],
            timestamp_column_name='timestamp',
            normalize=False,
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_missing_value_calculator_validates_column_names_list_elements():
    with pytest.raises(InvalidArgumentsException):
        _ = MissingValuesCalculator(
            column_names=[
                'car_value',
                {'ab': 1},
            ],
            timestamp_column_name='timestamp',
            normalize=False,
        )


def test_missing_value_calculator_fit_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    calc = MissingValuesCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(pd.DataFrame())


def test_missing_value_calculator_calculate_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValuesCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(pd.DataFrame())


def test_missing_value_calculator_fit_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValuesCalculator(
        column_names=[
            'car_value',
            'missing_column',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(reference)


def test_missing_value_calculator_calculate_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValuesCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(analysis.drop('driver_tenure', axis=1))


def test_whether_data_quality_metric_property_on_results_mv_rate(missing_value_result):
    assert missing_value_result.data_quality_metric == 'missing_values_rate'


def test_whether_result_data_dataframe_has_proper_columns(missing_value_result):
    cols = missing_value_result.data.columns
    assert len(cols) == 7 + 7 * 7
    assert ('chunk', 'key') in cols
    assert ('chunk', 'chunk_index') in cols
    assert ('chunk', 'start_index') in cols
    assert ('chunk', 'start_date') in cols
    assert ('chunk', 'end_index') in cols
    assert ('chunk', 'end_date') in cols
    assert ('chunk', 'period') in cols
    assert ('car_value', 'value') in cols
    assert ('car_value', 'upper_threshold') in cols
    assert ('car_value', 'lower_threshold') in cols
    assert ('car_value', 'alert') in cols
    assert ('car_value', 'sampling_error') in cols
    assert ('car_value', 'upper_confidence_boundary') in cols
    assert ('car_value', 'lower_confidence_boundary') in cols
    assert ('salary_range', 'value') in cols
    assert ('salary_range', 'upper_threshold') in cols
    assert ('salary_range', 'lower_threshold') in cols
    assert ('salary_range', 'alert') in cols
    assert ('salary_range', 'sampling_error') in cols
    assert ('salary_range', 'upper_confidence_boundary') in cols
    assert ('salary_range', 'lower_confidence_boundary') in cols
    assert ('debt_to_income_ratio', 'value') in cols
    assert ('debt_to_income_ratio', 'upper_threshold') in cols
    assert ('debt_to_income_ratio', 'lower_threshold') in cols
    assert ('debt_to_income_ratio', 'alert') in cols
    assert ('debt_to_income_ratio', 'sampling_error') in cols
    assert ('debt_to_income_ratio', 'upper_confidence_boundary') in cols
    assert ('debt_to_income_ratio', 'lower_confidence_boundary') in cols
    assert ('loan_length', 'value') in cols
    assert ('loan_length', 'upper_threshold') in cols
    assert ('loan_length', 'lower_threshold') in cols
    assert ('loan_length', 'alert') in cols
    assert ('loan_length', 'sampling_error') in cols
    assert ('loan_length', 'upper_confidence_boundary') in cols
    assert ('loan_length', 'lower_confidence_boundary') in cols
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
    assert ('driver_tenure', 'value') in cols
    assert ('driver_tenure', 'upper_threshold') in cols
    assert ('driver_tenure', 'lower_threshold') in cols
    assert ('driver_tenure', 'alert') in cols
    assert ('driver_tenure', 'sampling_error') in cols
    assert ('driver_tenure', 'upper_confidence_boundary') in cols
    assert ('driver_tenure', 'lower_confidence_boundary') in cols


def test_whether_data_quality_metric_property_on_results_mv_count():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = MissingValuesCalculator(
        column_names=[
            'car_value',
            'salary_range',
            'debt_to_income_ratio',
            'loan_length',
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
            'driver_tenure',
        ],
        normalize=False,
    ).fit(reference)
    assert calc.calculate(data=analysis).data_quality_metric == 'missing_values_count'


def test_results_filtering_column_str(missing_value_result):
    try:
        missing_value_result.filter(column_names='car_value')
    except Exception:
        pytest.fail()


def test_results_filtering_columns_list(missing_value_result):
    try:
        missing_value_result.filter(
            column_names=[
                'car_value',
                'salary_range',
                'debt_to_income_ratio',
            ]
        )
    except Exception:
        pytest.fail()


def test_results_car_value_values(missing_value_result):

    res = missing_value_result.filter(column_names='car_value').to_df()
    assert list(res[('car_value', 'value')]) == [0] * 20


def test_results_car_value_alerts(missing_value_result):

    res = missing_value_result.filter(column_names='car_value').to_df()
    assert list(res[('car_value', 'alert')]) == [False] * 20


def test_results_salary_range_values(missing_value_result):

    res = missing_value_result.filter(column_names='salary_range').to_df()
    assert list(res[('salary_range', 'value')]) == [
        0.0998,
        0.1004,
        0.0912,
        0.1018,
        0.1014,
        0.0966,
        0.1028,
        0.1108,
        0.1018,
        0.0934,
        0.1004,
        0.1064,
        0.0982,
        0.0966,
        0.0984,
        0.2188,
        0.2214,
        0.2178,
        0.2264,
        0.2156,
    ]


def test_results_salary_range_alerts(missing_value_result):
    res = missing_value_result.filter(column_names='salary_range').to_df()
    assert list(res[('salary_range', 'alert')]) == [False] * 15 + [True] * 5


def test_results_driver_tenure_values(missing_value_result):
    res = missing_value_result.filter(column_names='driver_tenure').to_df()
    assert list(res[('driver_tenure', 'value')]) == [
        0.1018,
        0.1080,
        0.0942,
        0.099,
        0.094,
        0.1066,
        0.0978,
        0.1062,
        0.0958,
        0.0966,
        0.1074,
        0.1042,
        0.094,
        0.1,
        0.0944,
        0.2196,
        0.2138,
        0.2236,
        0.2296,
        0.2134,
    ]


def test_results_driver_tenure_alerts(missing_value_result):

    res = missing_value_result.filter(column_names='driver_tenure').to_df()
    assert list(res[('driver_tenure', 'alert')]) == [False] * 15 + [True] * 5
