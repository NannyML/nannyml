#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Numerical Range Data Quality package."""

import pandas as pd
import pytest

from nannyml.data_quality.range import NumericalRangeCalculator, Result
from nannyml.datasets import load_synthetic_car_loan_data_quality_dataset
from nannyml.exceptions import InvalidArgumentsException

continuous_column_names = ['car_value', 'debt_to_income_ratio', 'loan_length']


@pytest.fixture(scope="module")
def numerical_range_result() -> Result:  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()

    calc = NumericalRangeCalculator(column_names=continuous_column_names).fit(reference)
    return calc.calculate(data=analysis)


def test_numerical_range_calculator_with_default_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = NumericalRangeCalculator(column_names=continuous_column_names).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_numerical_range_calculator_raises_invalid_arguments_exception_on_non_continuous_columns():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    with pytest.raises(InvalidArgumentsException, match=r".*['salary_range'].*"):
        _ = NumericalRangeCalculator(column_names=continuous_column_names + ['salary_range']).fit(reference)


def test_numerical_range_calculator_with_custom_params_should_not_fail():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    try:
        calc = NumericalRangeCalculator(
            column_names=continuous_column_names,
            chunk_period='M',
            timestamp_column_name='timestamp',
            normalize=False,
        ).fit(reference)
        _ = calc.calculate(data=analysis)
    except Exception:
        pytest.fail()


def test_numerical_range_calculator_validates_column_names_list_elements():  # noqa: D103
    with pytest.raises(InvalidArgumentsException):
        _ = NumericalRangeCalculator(
            column_names=[
                'car_value',
                {'ab': 1},
            ],
            timestamp_column_name='timestamp',
            normalize=False,
        )


def test_numerical_range_calculator_fit_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103, F821
    calc = NumericalRangeCalculator(
        column_names=continuous_column_names,
        timestamp_column_name='timestamp',
        normalize=False,
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(pd.DataFrame())


def test_numerical_range_calculator_calculate_should_raise_invalid_args_exception_when_no_data_present():  # noqa: D103
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = NumericalRangeCalculator(column_names=continuous_column_names).fit(reference_data=reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(pd.DataFrame())


def test_numerical_range_calculator_fit_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103
    reference, _, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = NumericalRangeCalculator(column_names=continuous_column_names)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(reference.drop('car_value', axis=1))


def test_numerical_range_calculator_calculate_should_raise_invalid_args_exception_when_column_missing():  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = NumericalRangeCalculator(column_names=continuous_column_names).fit(reference)
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(analysis.drop('car_value', axis=1))


@pytest.mark.parametrize(
    'normalize, expected_metric', [(True, 'out_of_range_values_rate'), (False, 'out_of_range_values_count')]
)
def test_metric_is_set_properly(normalize, expected_metric):  # noqa: D103
    reference, analysis, _ = load_synthetic_car_loan_data_quality_dataset()
    calc = NumericalRangeCalculator(column_names=continuous_column_names, normalize=normalize).fit(reference)
    res = calc.calculate(analysis)
    assert calc.data_quality_metric == expected_metric
    assert res.data_quality_metric == expected_metric


def test_whether_result_data_dataframe_has_proper_columns(numerical_range_result):  # noqa: D103
    cols = numerical_range_result.data.columns
    assert len(cols) == 7 + 3 * 4
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


def test_results_filtering_column_as_str(numerical_range_result):  # noqa: D103
    try:
        numerical_range_result.filter(column_names='car_value')
    except Exception:
        pytest.fail()


def test_results_filtering_column_as_list(numerical_range_result):  # noqa: D103
    try:
        numerical_range_result.filter(
            column_names=[
                'car_value',
            ]
        )
    except Exception:
        pytest.fail()


@pytest.mark.parametrize(
    'column_name, expected_values',
    [
        ('car_value', [0] * 20),
        (
            'debt_to_income_ratio',
            [0.0] * 16 + [0.0004, 0.0, 0.0, 0.0],
        ),
        ('loan_length', [0] * 20),
    ],
)
def test_results_repaid_loan_on_prev_car_values(numerical_range_result, column_name, expected_values):  # noqa: D103
    res = numerical_range_result.filter(column_names=column_name).to_df()
    assert list(res[(column_name, 'value')]) == expected_values


def test_results_alerts(numerical_range_result):  # noqa: D103
    res = numerical_range_result.filter(column_names='debt_to_income_ratio').to_df()
    assert list(res[('debt_to_income_ratio', 'alert')]) == [False] * 16 + [True, False, False, False]
