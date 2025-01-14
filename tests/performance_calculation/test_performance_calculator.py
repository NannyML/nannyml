#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the PerformanceCalculator."""
from copy import copy
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml._typing import ProblemType, Result
from nannyml.datasets import (
    load_synthetic_binary_classification_dataset,
    load_synthetic_car_price_dataset,
    load_synthetic_multiclass_classification_dataset,
)
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.calculator import DEFAULT_THRESHOLDS, PerformanceCalculator
from nannyml.performance_calculation.metrics.binary_classification import (
    BinaryClassificationAUROC,
    BinaryClassificationF1,
)
from nannyml.thresholds import ConstantThreshold, StandardDeviationThreshold


@pytest.fixture(scope='module')
def data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture()
def performance_calculator() -> PerformanceCalculator:  # noqa: D103
    return PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    )


@pytest.fixture(scope='module')
def performance_result(data) -> Result:  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    ).fit(reference_data=data[0])

    ref_with_tgt = data[1].merge(data[2], on='id')
    return calc.calculate(ref_with_tgt)


def test_calculator_init_with_empty_metrics_should_not_fail():  # noqa: D103, F821
    try:
        _ = PerformanceCalculator(
            timestamp_column_name='timestamp',
            y_pred='y_pred',
            y_pred_proba='y_pred_proba',
            y_true='y_true',
            metrics=[],
            problem_type='classification_binary',
        )
    except Exception as exc:
        pytest.fail(f'unexpected exception: {exc}')


def test_calculator_init_should_set_metrics(performance_calculator):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    )
    sut = calc.metrics
    assert len(sut) == 2
    assert sut[0] == BinaryClassificationAUROC(
        y_true=calc.y_true, y_pred=calc.y_pred, y_pred_proba=calc.y_pred_proba, threshold=StandardDeviationThreshold()
    )
    assert sut[1] == BinaryClassificationF1(
        y_true=calc.y_true, y_pred=calc.y_pred, y_pred_proba=calc.y_pred_proba, threshold=StandardDeviationThreshold()
    )


@pytest.mark.parametrize('metrics, expected', [('roc_auc', ['roc_auc']), (['roc_auc', 'f1'], ['roc_auc', 'f1'])])
def test_performance_calculator_create_with_single_or_list_of_metrics(metrics, expected):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='y_true',
        metrics=metrics,
        problem_type='classification_binary',
    )
    assert [metric.column_name for metric in calc.metrics] == expected


@pytest.mark.parametrize(
    'problem',
    [
        "classification_multiclass",
        "regression",
    ],
)
def test_performance_calculator_create_raises_exception_when_y_pred_not_given_and_problem_type_not_binary_clf(  # noqa: D103, E501
    problem,
):
    with pytest.raises(InvalidArgumentsException, match=f"'y_pred' can not be 'None' for problem type {problem}"):
        _ = PerformanceCalculator(
            timestamp_column_name='timestamp',
            y_pred_proba='y_pred_proba',
            y_true='y_true',
            metrics=['roc_auc', 'f1'],
            problem_type=problem,
        )


@pytest.mark.parametrize(
    'metric, expected',
    [
        (['roc_auc', 'f1'], "['f1']"),
        (['roc_auc', 'f1', 'average_precision', 'precision'], "['f1', 'precision']"),
    ],
)
def test_performance_calculator_create_without_y_pred_raises_exception_when_metrics_require_it(  # noqa: D103
    metric, expected
):
    with pytest.raises(InvalidArgumentsException, match=expected):
        _ = PerformanceCalculator(
            timestamp_column_name='timestamp',
            y_pred_proba='y_pred_proba',
            y_true='y_true',
            metrics=metric,
            problem_type='classification_binary',
        )


@pytest.mark.parametrize('metric', ['roc_auc', 'average_precision'])
def test_performance_calculator_create_without_y_pred_works_when_metrics_dont_require_it(metric):  # noqa: D103
    try:
        _ = PerformanceCalculator(
            timestamp_column_name='timestamp',
            y_pred_proba='y_pred_proba',
            y_true='y_true',
            metrics=metric,
            problem_type='classification_binary',
        )
    except Exception as exc:
        pytest.fail(f'unexpected exception: {exc}')


def test_calculator_fit_should_raise_invalid_args_exception_when_no_target_data_present(data):  # noqa: D103, F821
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    )
    with pytest.raises(InvalidArgumentsException):
        _ = calc.fit(reference_data=data[0])


def test_calculator_calculate_should_raise_invalid_args_exception_when_no_target_data_present(data):  # noqa: D103, F821
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    ).fit(reference_data=data[0])
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(data[1])


def test_calculator_calculate_should_include_chunk_information_columns(data):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    ).fit(reference_data=data[0])

    ref_with_tgt = data[1].merge(data[2], on='id')
    sut = calc.calculate(ref_with_tgt)

    assert ('chunk', 'key') in sut.data.columns
    assert ('chunk', 'start_index') in sut.data.columns
    assert ('chunk', 'end_index') in sut.data.columns
    assert ('chunk', 'start_date') in sut.data.columns
    assert ('chunk', 'end_date') in sut.data.columns
    assert ('chunk', 'period') in sut.data.columns


def test_calculator_calculate_should_include_target_completeness_rate(data):  # noqa: D103
    # Let's artificially modify the target completeness of different chunks.
    ref_data = data[0]
    data = data[1].merge(data[2], on='id')

    # Drop 10% of the target values in the first chunk
    data.loc[0:499, 'work_home_actual'] = np.nan

    # Drop 90% of the target values in the second chunk
    data.loc[5000:9499, 'work_home_actual'] = np.nan

    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    ).fit(reference_data=ref_data)
    result = calc.calculate(data)
    sut = result.filter(period='analysis').to_df()

    assert ('chunk', 'targets_missing_rate') in sut.columns
    assert sut.loc[0, ('chunk', 'targets_missing_rate')] == 0.1
    assert sut.loc[1, ('chunk', 'targets_missing_rate')] == 0.9


def test_calculator_calculate_should_support_partial_bool_targets(data, performance_calculator):
    """Test that the calculator supports partial bool targets.

    Pandas converts bool columns to object dtype when they contain NaN values. This previously resulted in problems
    when calculating the performance metrics. This test ensures that the calculator supports partial bool targets.
    """
    ref_data = data[0]
    analysis_data = data[1].merge(data[2], on='id')

    # Convert target column to bool dtype
    analysis_data = analysis_data.astype({'work_home_actual': 'bool'})

    # Drop 10% of the target values in the first chunk
    analysis_data.loc[0:499, 'work_home_actual'] = np.nan

    performance_calculator.fit(reference_data=ref_data)
    performance_calculator.calculate(analysis_data)

    # No further checks needed, if the above code runs without errors, the test passes.


@pytest.mark.parametrize(
    'custom_thresholds',
    [
        {'roc_auc': ConstantThreshold(lower=1, upper=2)},
        {'roc_auc': ConstantThreshold(lower=1, upper=2), 'f1': ConstantThreshold(lower=1, upper=2)},
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
            'specificity': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
            'specificity': ConstantThreshold(lower=1, upper=2),
        },
        {
            'roc_auc': ConstantThreshold(lower=1, upper=2),
            'f1': ConstantThreshold(lower=1, upper=2),
            'precision': ConstantThreshold(lower=1, upper=2),
            'recall': ConstantThreshold(lower=1, upper=2),
            'specificity': ConstantThreshold(lower=1, upper=2),
            'accuracy': ConstantThreshold(lower=1, upper=2),
        },
        {
            'mae': ConstantThreshold(lower=1, upper=2),
            'mape': ConstantThreshold(lower=1, upper=2),
            'mse': ConstantThreshold(lower=1, upper=2),
            'msle': ConstantThreshold(lower=1, upper=2),
            'rmse': ConstantThreshold(lower=1, upper=2),
            'rmsle': ConstantThreshold(lower=1, upper=2),
        },
    ],
)
def test_performance_calculator_with_custom_thresholds(custom_thresholds):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
        thresholds=custom_thresholds,
    )
    sut = calc.thresholds
    expected_thresholds = copy(DEFAULT_THRESHOLDS)
    expected_thresholds.update(**custom_thresholds)
    assert sut == expected_thresholds


def test_performance_calculator_with_default_thresholds():  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    )
    sut = calc.thresholds

    assert sut == DEFAULT_THRESHOLDS


# See https://github.com/NannyML/nannyml/issues/192
def test_calculator_returns_distinct_but_consistent_results_when_reused(data, performance_calculator):  # noqa: D103
    reference, analysis, target = data

    data = analysis.merge(target, on='id')
    performance_calculator.fit(reference)
    result1 = performance_calculator.calculate(data)
    result2 = performance_calculator.calculate(data)

    # Checks two distinct results are returned. Previously there was a bug causing the previous result instance to be
    # modified on subsequent estimates.
    assert result1 is not result2
    pd.testing.assert_frame_equal(result1.to_df(), result2.to_df())


# See https://github.com/NannyML/nannyml/issues/197
def test_performance_calculator_result_filter_should_preserve_data_with_default_args(performance_result):  # noqa: D103
    filtered_result = performance_result.filter()
    assert filtered_result.data.equals(performance_result.data)


# See https://github.com/NannyML/nannyml/issues/197
def test_performance_calculator_result_filter_metrics(performance_result):  # noqa: D103
    filtered_result = performance_result.filter(metrics=['roc_auc'])
    columns = tuple(set(metric for (metric, _) in filtered_result.data.columns if metric != 'chunk'))
    assert columns == ('roc_auc',)
    assert filtered_result.data.shape[0] == performance_result.data.shape[0]


# See https://github.com/NannyML/nannyml/issues/197
def test_performance_calculator_result_filter_period(performance_result):  # noqa: D103
    ref_period = performance_result.data.loc[performance_result.data.loc[:, ('chunk', 'period')] == 'reference', :]
    filtered_result = performance_result.filter(period='reference')
    assert filtered_result.data.equals(ref_period)


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': False, 'metric': 'mae'}),
        ({}, {'kind': 'performance', 'plot_reference': False, 'metric': 'mae'}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': True, 'metric': 'mae'}),
        ({}, {'kind': 'performance', 'plot_reference': True, 'metric': 'mae'}),
    ],
    ids=[
        'performance_with_timestamp_without_reference',
        'performance_without_timestamp_without_reference',
        'performance_with_timestamp_with_reference',
        'performance_without_timestamp_with_reference',
    ],
)
def test_regression_result_plots_raise_no_exceptions(calc_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    calc = PerformanceCalculator(
        y_true='y_true', y_pred='y_pred', problem_type=ProblemType.REGRESSION, metrics=['mae', 'mape'], **calc_args
    ).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, on='id'))

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
    ],
    ids=[
        'performance_with_timestamp_without_reference',
        'performance_without_timestamp_without_reference',
        'performance_with_timestamp_with_reference',
        'performance_without_timestamp_with_reference',
    ],
)
def test_multiclass_classification_result_plots_raise_no_exceptions(calc_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_multiclass_classification_dataset()
    calc = PerformanceCalculator(
        y_true='y_true',
        y_pred='y_pred',
        y_pred_proba={
            'upmarket_card': 'y_pred_proba_upmarket_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'prepaid_card': 'y_pred_proba_prepaid_card',
        },
        problem_type=ProblemType.CLASSIFICATION_MULTICLASS,
        metrics=['roc_auc', 'f1'],
        **calc_args,
    ).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, left_index=True, right_index=True))

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


@pytest.mark.parametrize(
    'calc_args, plot_args',
    [
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': False, 'metric': 'f1'}),
        ({'timestamp_column_name': 'timestamp'}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
        ({}, {'kind': 'performance', 'plot_reference': True, 'metric': 'f1'}),
    ],
    ids=[
        'performance_with_timestamp_without_reference',
        'performance_without_timestamp_without_reference',
        'performance_with_timestamp_with_reference',
        'performance_without_timestamp_with_reference',
    ],
)
def test_binary_classification_result_plots_raise_no_exceptions(calc_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    calc = PerformanceCalculator(
        y_true='work_home_actual',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        problem_type=ProblemType.CLASSIFICATION_BINARY,
        metrics=['roc_auc', 'f1'],
        **calc_args,
    ).fit(reference)
    sut = calc.calculate(analysis.merge(analysis_targets, on='id'))

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_binary_classification_calculate_without_prediction_column():  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_binary_classification_dataset()
    try:
        calc = PerformanceCalculator(
            y_true='work_home_actual',
            y_pred_proba='y_pred_proba',
            problem_type=ProblemType.CLASSIFICATION_BINARY,
            metrics=['roc_auc', 'average_precision'],
            timestamp_column_name='timestamp',
            chunk_period='M',
        ).fit(reference)
        _ = calc.calculate(analysis.merge(analysis_targets, on='id'))
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_input_dataframes_are_not_altered_by_bin_class_calculator(data):  # noqa: D103
    reference, monitored, targets = data
    reference2 = reference.copy(deep=True)
    monitored = monitored.merge(targets, left_index=True, right_index=True)
    monitored2 = monitored.copy(deep=True)
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba='y_pred_proba',
        y_true='work_home_actual',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    )
    calc.fit(reference2)
    results = calc.calculate(monitored2)  # noqa: F841
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)


def test_input_dataframes_are_not_altered_by_multiclass_calculator(data):  # noqa: D103
    reference, monitored, targets = load_synthetic_multiclass_classification_dataset()
    reference2 = reference.copy(deep=True)
    monitored = monitored.merge(targets, left_index=True, right_index=True)
    monitored2 = monitored.copy(deep=True)
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_pred_proba={
            'prepaid_card': 'y_pred_proba_prepaid_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'upmarket_card': 'y_pred_proba_upmarket_card'
        },
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_multiclass',
    )
    calc.fit(reference2)
    results = calc.calculate(monitored2)  # noqa: F841
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)


def test_input_dataframes_are_not_altered_by_regression_calculator(data):  # noqa: D103
    reference, monitored, targets = load_synthetic_car_price_dataset()
    reference2 = reference.copy(deep=True)
    monitored = monitored.merge(targets, left_index=True, right_index=True)
    monitored2 = monitored.copy(deep=True)
    calc = PerformanceCalculator(
        y_pred='y_pred',
        y_true='y_true',
        timestamp_column_name='timestamp',
        problem_type='regression',
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    calc.fit(reference2)
    results = calc.calculate(monitored2)  # noqa: F841
    pd.testing.assert_frame_equal(monitored, monitored2)
    pd.testing.assert_frame_equal(reference, reference2)
