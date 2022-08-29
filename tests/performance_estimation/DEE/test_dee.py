#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import List, Tuple

import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_regression_dataset
from nannyml.performance_estimation.direct_error_estimation import DEE


@pytest.fixture(scope='module')
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference, analysis, _ = load_synthetic_regression_dataset()

    return reference, analysis


@pytest.fixture(scope='module')
def regression_feature_columns(regression_data) -> List[str]:
    return [col for col in regression_data[0].columns if col not in ['y_pred', 'y_true', 'timestamp']]


@pytest.fixture(scope='module')
def direct_error_estimator(regression_feature_columns) -> DEE:
    return DEE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )


@pytest.fixture(scope='module')
def estimates(regression_data, direct_error_estimator):
    reference, analysis = regression_data

    # Get rid of negative values for log based metrics
    reference = reference[~(reference['y_pred'] < 0)]
    analysis = analysis[~(analysis['y_pred'] < 0)]

    direct_error_estimator.fit(reference)
    return direct_error_estimator.estimate(analysis)


def test_direct_error_estimator_does_not_tune_hyperparameters_by_default(regression_feature_columns):
    sut = DEE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    assert not sut.tune_hyperparameters


def test_direct_error_estimator_has_default_hyperparameter_tuning_config(regression_feature_columns):
    sut = DEE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    assert sut.hyperparameter_tuning_config


def test_direct_error_estimator_sets_custom_hyperparameter_tuning_config_when_given(regression_feature_columns):
    sut = DEE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
        hyperparameter_tuning_config={'foo': True, 'bar': 3},
    )
    assert sut.hyperparameter_tuning_config['foo'] is True
    assert sut.hyperparameter_tuning_config['bar'] == 3


@pytest.mark.parametrize(
    'metric, expected',
    [
        (
            'mae',
            [
                922.24492,
                913.32718,
                912.36343,
                912.68182,
                914.27627,
                907.69949,
                809.35055,
                819.30675,
                815.97475,
                814.09953,
                811.62199,
                817.23316,
            ],
        ),
        (
            'mape',
            [
                0.23742,
                0.23989,
                0.24011,
                0.23916,
                0.23854,
                0.24065,
                0.25886,
                0.25487,
                0.25825,
                0.25707,
                0.25437,
                0.25737,
            ],
        ),
        (
            'mse',
            [
                1362237.07081,
                1324592.82477,
                1323114.60686,
                1323382.20771,
                1329641.2642,
                1303324.34242,
                1025880.96975,
                1047471.07265,
                1043703.77853,
                1031991.05493,
                1025920.19823,
                1042267.59821,
            ],
        ),
        (
            'msle',
            [0.09005, 0.09708, 0.09716, 0.08698, 0.09267, 0.10068, 0.20863, 0.189, 0.19365, 0.19378, 0.18325, 0.20827],
        ),
        (
            'rmse',
            [
                1167.14912,
                1150.90956,
                1150.26719,
                1150.3835,
                1153.10072,
                1141.63231,
                1012.85782,
                1023.46034,
                1021.61822,
                1015.86961,
                1012.87719,
                1020.91508,
            ],
        ),
        (
            'rmsle',
            [0.30008, 0.31157, 0.3117, 0.29493, 0.30442, 0.31731, 0.45677, 0.43475, 0.44006, 0.4402, 0.42807, 0.45636],
        ),
    ],
)
def test_direct_error_estimation_yields_correct_results_for_metric(estimates, metric, expected):
    all(estimates.data[f'estimated_{metric}'] == expected)
