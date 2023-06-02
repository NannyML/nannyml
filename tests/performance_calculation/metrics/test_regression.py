#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml import DefaultChunker, PerformanceCalculator
from nannyml._typing import ProblemType
from nannyml.datasets import load_synthetic_car_price_dataset
from nannyml.performance_calculation.metrics.base import MetricFactory
from nannyml.performance_calculation.metrics.regression import MAE, MAPE, MSE, MSLE, RMSE, RMSLE
from nannyml.thresholds import ConstantThreshold, StandardDeviationThreshold


@pytest.fixture(scope='module')
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_car_price_dataset()

    return ref_df, ana_df, tgt_df


@pytest.fixture(scope='module')
def performance_calculator() -> PerformanceCalculator:
    return PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
        problem_type='regression',
    )


@pytest.fixture(scope='module')
def realized_performance_metrics(performance_calculator, regression_data) -> pd.DataFrame:
    # Get rid of negative values for log based metrics
    reference = regression_data[0][~(regression_data[0]['y_pred'] < 0)]
    analysis = regression_data[1][~(regression_data[1]['y_pred'] < 0)]

    performance_calculator.fit(reference)
    results = performance_calculator.calculate(analysis.join(regression_data[2])).filter(period='analysis')
    return results.data


@pytest.fixture(scope='module')
def no_timestamp_metrics(regression_data) -> pd.DataFrame:
    # Get rid of negative values for log based metrics
    reference = regression_data[0][~(regression_data[0]['y_pred'] < 0)]
    analysis = regression_data[1][~(regression_data[1]['y_pred'] < 0)]

    performance_calculator = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
        problem_type='regression',
    ).fit(reference)
    results = performance_calculator.calculate(analysis.join(regression_data[2])).filter(period='analysis')
    return results.to_df()


@pytest.mark.parametrize(
    'key,problem_type,metric',
    [
        ('mae', ProblemType.REGRESSION, MAE),
        ('mape', ProblemType.REGRESSION, MAPE),
        ('mse', ProblemType.REGRESSION, MSE),
        ('msle', ProblemType.REGRESSION, MSLE),
        ('rmse', ProblemType.REGRESSION, RMSE),
        ('rmsle', ProblemType.REGRESSION, RMSLE),
    ],
)
def test_metric_factory_returns_correct_metric_given_key_and_problem_type(key, problem_type, metric):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba=None,
        y_pred='y_pred',
        y_true='y_true',
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
        problem_type='regression',
    )
    sut = MetricFactory.create(
        key,
        problem_type,
        y_true=calc.y_true,
        y_pred=calc.y_pred,
        y_pred_proba=calc.y_pred_proba,
        threshold=StandardDeviationThreshold(),
    )
    assert sut == metric(
        y_true=calc.y_true, y_pred=calc.y_pred, y_pred_proba=calc.y_pred_proba, threshold=StandardDeviationThreshold()
    )


@pytest.mark.parametrize(
    'metric, expected',
    [
        (
            'mae',
            [853.39967, 853.13667, 846.304, 855.4945, 849.3295, 702.51767, 700.73583, 684.70167, 705.814, 698.34383],
        ),
        ('mape', [0.22871, 0.23082, 0.22904, 0.23362, 0.23389, 0.26286, 0.26346, 0.26095, 0.26537, 0.26576]),
        (
            'mse',
            [
                1143129.298,
                1139867.667,
                1128720.807,
                1158285.6715,
                1124285.66517,
                829589.49233,
                829693.3775,
                792286.80933,
                835916.964,
                825935.67917,
            ],
        ),
        ('msle', [0.07049, 0.06999, 0.06969, 0.07193, 0.07249, 0.10495, 0.10481, 0.10435, 0.10471, 0.10588]),
        (
            'rmse',
            [
                1069.17225,
                1067.64585,
                1062.41273,
                1076.23681,
                1060.32338,
                910.81803,
                910.87506,
                890.10494,
                914.28495,
                908.81003,
            ],
        ),
        ('rmsle', [0.2655, 0.26456, 0.26399, 0.2682, 0.26924, 0.32396, 0.32375, 0.32303, 0.3236, 0.32539]),
    ],
)
def test_metric_values_are_calculated_correctly(realized_performance_metrics, metric, expected):
    metric_values = realized_performance_metrics.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()


@pytest.mark.parametrize(
    'metric, expected',
    [
        (
            'mae',
            [853.39967, 853.13667, 846.304, 855.4945, 849.3295, 702.51767, 700.73583, 684.70167, 705.814, 698.34383],
        ),
        ('mape', [0.22871, 0.23082, 0.22904, 0.23362, 0.23389, 0.26286, 0.26346, 0.26095, 0.26537, 0.26576]),
        (
            'mse',
            [
                1143129.298,
                1139867.667,
                1128720.807,
                1158285.6715,
                1124285.66517,
                829589.49233,
                829693.3775,
                792286.80933,
                835916.964,
                825935.67917,
            ],
        ),
        ('msle', [0.07049, 0.06999, 0.06969, 0.07193, 0.07249, 0.10495, 0.10481, 0.10435, 0.10471, 0.10588]),
        (
            'rmse',
            [
                1069.17225,
                1067.64585,
                1062.41273,
                1076.23681,
                1060.32338,
                910.81803,
                910.87506,
                890.10494,
                914.28495,
                908.81003,
            ],
        ),
        ('rmsle', [0.2655, 0.26456, 0.26399, 0.2682, 0.26924, 0.32396, 0.32375, 0.32303, 0.3236, 0.32539]),
    ],
)
def test_metric_values_without_timestamps_are_calculated_correctly(no_timestamp_metrics, metric, expected):
    metric_values = no_timestamp_metrics.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()


@pytest.mark.parametrize('metric_cls', [MAE, MAPE, MSE, MSLE, RMSE, RMSLE])
def test_metric_logs_warning_when_lower_threshold_is_overridden_by_metric_limits(caplog, metric_cls, regression_data):
    reference = regression_data[0]
    metric = metric_cls(y_pred='y_pred', y_true='y_true', threshold=ConstantThreshold(lower=-1))
    metric.fit(reference, chunker=DefaultChunker())

    assert (
        f'{metric.display_name} lower threshold value -1 overridden by '
        f'lower threshold value limit {metric.lower_threshold_value_limit}' in caplog.messages
    )
