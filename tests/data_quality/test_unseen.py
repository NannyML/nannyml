#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Drift package."""

import pandas as pd
import pytest

from nannyml._typing import Result
from nannyml.data_quality import UnseenValuesCalculator
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
        _ = UnseenValuesCalculator(
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
            normalize=False,
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_unseen_value_calculator_validates_column_names_list_elements():
    with pytest.raises(InvalidArgumentsException):
        _ = UnseenValuesCalculator(
            column_names=[
                'car_value',
                {'ab': 1},
            ],
            timestamp_column_name='timestamp',
            normalize=False,
        )


def test_unseen_value_calculator_fit_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    calc = UnseenValuesCalculator(
        column_names=[
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(pd.DataFrame())


def test_unseen_value_calculator_calculate_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(pd.DataFrame())


def test_unseen_value_calculator_fit_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
            'car_value',
            'missing_column',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(reference)


def test_unseen_value_calculator_calculate_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
        ],
        timestamp_column_name='timestamp',
        normalize=False,
    ).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(analysis.drop('size_of_downpayment', axis=1))


def test_whether_data_quality_metric_property_on_results_mv_rate(unseen_value_result):
    assert unseen_value_result.data_quality_metric == 'unseen_values_rate'


def test_whether_result_data_dataframe_has_proper_columns(unseen_value_result):
    cols = unseen_value_result.data.columns
    assert len(cols) == 7 + 2 * 4
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
    assert ('size_of_downpayment', 'value') in cols
    assert ('size_of_downpayment', 'upper_threshold') in cols
    assert ('size_of_downpayment', 'lower_threshold') in cols
    assert ('size_of_downpayment', 'alert') in cols


def test_whether_data_quality_metric_property_on_results_mv_count():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = UnseenValuesCalculator(
        column_names=[
            'repaid_loan_on_prev_car',
            'size_of_downpayment',
        ],
        normalize=False,
    ).fit(reference)
    assert calc.calculate(data=analysis).data_quality_metric == 'unseen_values_count'


def test_results_filtering_column_repaid_loan_on_prev_car_str(unseen_value_result):
    try:
        unseen_value_result.filter(column_names='repaid_loan_on_prev_car')
    except Exception:
        pytest.fail()


def test_results_filtering_columns_size_of_downpayment_list(unseen_value_result):
    try:
        unseen_value_result.filter(
            column_names=[
                'size_of_downpayment',
            ]
        )
    except Exception:
        pytest.fail()


def test_results_repaid_loan_on_prev_car_values(unseen_value_result):
    res = unseen_value_result.filter(column_names='repaid_loan_on_prev_car').to_df()
    assert list(res[('repaid_loan_on_prev_car', 'value')]) == [0] * 20


def test_results_repaid_loan_on_prev_car_alerts(unseen_value_result):
    res = unseen_value_result.filter(column_names='repaid_loan_on_prev_car').to_df()
    assert list(res[('repaid_loan_on_prev_car', 'alert')]) == [False] * 20


def test_results_size_of_downpayment_values(unseen_value_result):

    res = unseen_value_result.filter(column_names='size_of_downpayment').to_df()
    assert list(res[('size_of_downpayment', 'value')]) == [0] * 10 + [
        0.0094,
        0.0062,
        0.0076,
        0.0072,
        0.0096,
        0.0188,
        0.0178,
        0.0202,
        0.0222,
        0.021,
    ]


def test_results_size_of_downpayment_alerts(unseen_value_result):
    res = unseen_value_result.filter(column_names='size_of_downpayment').to_df()
    assert list(res[('size_of_downpayment', 'alert')]) == [False] * 10 + [True] * 10
