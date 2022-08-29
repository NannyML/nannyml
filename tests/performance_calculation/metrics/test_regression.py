#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml import PerformanceCalculator
from nannyml._typing import ProblemType
from nannyml.datasets import load_synthetic_regression_dataset
from nannyml.performance_calculation.metrics.base import MetricFactory
from nannyml.performance_calculation.metrics.regression import MAE, MAPE, MSE, MSLE, RMSE, RMSLE


@pytest.fixture(scope='module')
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_regression_dataset()

    return ref_df, ana_df, tgt_df


@pytest.fixture(scope='module')
def performance_calculator() -> PerformanceCalculator:
    return PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba=None,
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
    results = performance_calculator.calculate(analysis.join(regression_data[2]))
    return results.data


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
    sut = MetricFactory.create(key, problem_type, {'calculator': calc})
    assert sut == metric(calculator=calc)


@pytest.mark.parametrize(
    'metric, expected',
    [
        (
            'mae',
            [
                913.56065,
                906.18711,
                897.32938,
                917.75395,
                908.1,
                807.02749,
                809.91271,
                813.6012,
                800.02577,
                799.59124,
                294.0,
            ],
        ),
        ('mape', [0.23289, 0.23985, 0.23447, 0.23937, 0.23582, 0.25144, 0.25695, 0.25657, 0.25292, 0.25268, 0.15638]),
        (
            'mse',
            [
                1350900.19433,
                1313639.44381,
                1298236.53179,
                1352916.15292,
                1324969.00481,
                1029658.31787,
                1010832.30653,
                1039277.80808,
                992979.93333,
                1001155.24725,
                86436.0,
            ],
        ),
        ('msle', [0.09745, 0.09485, 0.08649, 0.08964, 0.08742, 0.25874, 0.27357, 0.29243, 0.28181, 0.27527, 0.02889]),
        (
            'rmse',
            [
                1162.28232,
                1146.14111,
                1139.40183,
                1163.14924,
                1151.07298,
                1014.72081,
                1005.40156,
                1019.44976,
                996.48378,
                1000.57746,
                294.0,
            ],
        ),
        ('rmsle', [0.31216, 0.30797, 0.2941, 0.2994, 0.29567, 0.50867, 0.52304, 0.54076, 0.53086, 0.52467, 0.16996]),
    ],
)
def test_metric_values_are_calculated_correctly(realized_performance_metrics, metric, expected):
    metric_values = realized_performance_metrics[metric]
    assert (round(metric_values, 5) == expected).all()
