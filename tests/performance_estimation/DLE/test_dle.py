#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import List, Optional, Tuple

import pandas as pd
import plotly.graph_objects
import pytest

from nannyml._typing import Key, ProblemType, Result, Self
from nannyml.base import Abstract1DResult, AbstractEstimator
from nannyml.datasets import load_synthetic_car_price_dataset
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation.dle import DEFAULT_THRESHOLDS, DLE
from nannyml.performance_estimation.direct_loss_estimation.metrics import MetricFactory
from nannyml.thresholds import ConstantThreshold, StandardDeviationThreshold


class FakeEstimatorResult(Abstract1DResult):
    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> Self:
        return self

    def keys(self) -> List[Key]:
        return []

    def plot(self, *args, **kwargs) -> plotly.graph_objects.Figure:
        return plotly.graph_objects.Figure()


class FakeEstimator(AbstractEstimator):
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        return FakeEstimatorResult(pd.DataFrame())


@pytest.fixture(scope='module')
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference, analysis, _ = load_synthetic_car_price_dataset()

    return reference, analysis


@pytest.fixture(scope='module')
def regression_feature_columns(regression_data) -> List[str]:
    return [col for col in regression_data[0].columns if col not in ['y_pred', 'y_true', 'timestamp']]


@pytest.fixture(scope='module')
def direct_error_estimator(regression_feature_columns) -> DLE:
    return DLE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=10000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )


@pytest.fixture(scope='module')
def estimates(regression_data, direct_error_estimator):
    reference, analysis = regression_data

    # Get rid of negative values for log based metrics
    reference = reference[~(reference['y_pred'] < 0)]
    analysis = analysis[~(analysis['y_pred'] < 0)]

    direct_error_estimator.fit(reference)
    estimates = direct_error_estimator.estimate(analysis)
    return estimates


@pytest.fixture(scope='module')
def hypertuned_estimates(regression_data, direct_error_estimator: DLE):
    direct_error_estimator.tune_hyperparameters = True
    direct_error_estimator.hyperparameter_tuning_config = {
        "time_budget": 3,  # total running time in seconds
        "metric": "mse",
        "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
        "eval_method": "cv",  # resampling strategy
        "hpo_method": "cfo",  # hyperparameter optimization method, cfo is default.
        "n_splits": 5,  # Default Value is 5
        "task": 'regression',  # task type
        "seed": 1,  # random seed
        "verbose": 0,
    }

    reference, analysis = regression_data

    # Get rid of negative values for log based metrics
    reference = reference[~(reference['y_pred'] < 0)]
    analysis = analysis[~(analysis['y_pred'] < 0)]

    direct_error_estimator.fit(reference)
    return direct_error_estimator.estimate(analysis)


@pytest.fixture(scope='module')
def custom_hyperparameter_estimates(regression_data, direct_error_estimator: DLE):
    direct_error_estimator.hyperparameters = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 0.9222473950589268,
        'importance_type': 'split',
        'learning_rate': 1.0,
        'max_depth': -1,
        'min_child_samples': 22,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 9,
        'n_jobs': -1,
        'num_leaves': 5,
        'objective': None,
        'random_state': None,
        'reg_alpha': 0.003805728782845039,
        'reg_lambda': 0.1090149084007093,
        'silent': 'warn',
        'subsample': 1.0,
        'subsample_for_bin': 200000,
        'subsample_freq': 0,
        'max_bin': 1023,
        'verbose': -1,
    }

    reference, analysis = regression_data

    # Get rid of negative values for log based metrics
    reference = reference[~(reference['y_pred'] < 0)]
    analysis = analysis[~(analysis['y_pred'] < 0)]

    direct_error_estimator.fit(reference)
    return direct_error_estimator.estimate(analysis)


@pytest.mark.parametrize(
    'metrics, expected',
    [('mae', ['mae']), (['mae', 'mape'], ['mae', 'mape']), (None, ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle'])],
)
def test_dle_create_with_single_or_list_of_metrics(regression_feature_columns, metrics, expected):
    sut = DLE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=metrics,
    )
    assert [metric.column_name for metric in sut.metrics] == expected


def test_direct_error_estimator_does_not_tune_hyperparameters_by_default(regression_feature_columns):
    sut = DLE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    assert not sut.tune_hyperparameters


def test_direct_error_estimator_has_default_hyperparameter_tuning_config(regression_feature_columns):
    sut = DLE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=5000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    assert sut.hyperparameter_tuning_config


def test_direct_error_estimator_sets_custom_hyperparameter_tuning_config_when_given(regression_feature_columns):
    sut = DLE(
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
        ('mae', [917.78605, 912.52263, 910.98788, 814.32865, 815.03714, 813.81214]),
        ('mape', [0.23866, 0.23964, 0.23959, 0.25686, 0.25766, 0.25554]),
        ('mse', [1343414.94779, 1323248.40729, 1316482.80331, 1036676.0212, 1037847.41673, 1032300.88685]),
        ('msle', [0.09356, 0.09207, 0.09668, 0.19882, 0.19372, 0.19301]),
        ('rmse', [1159.05778, 1150.32535, 1147.38084, 1018.17288, 1018.74797, 1016.02209]),
        ('rmsle', [0.30588, 0.30343, 0.31093, 0.44589, 0.44013, 0.43933]),
    ],
)
def test_direct_error_estimation_yields_correct_results_for_metric(estimates, metric, expected):
    sut = estimates.filter(period='analysis').to_df()
    all(round(sut.loc[:, (metric, 'value')], 5) == expected)


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('mae', [917.63663, 912.81705, 911.64143, 818.89734, 819.74329, 818.98411]),
        ('mape', [0.23839, 0.23956, 0.23966, 0.25599, 0.2568, 0.25406]),
        ('mse', [1337792.58904, 1322011.64262, 1322062.44625, 1078916.36436, 1079277.22531, 1079555.08966]),
        ('msle', [0.0986, 0.09122, 0.09389, 0.18337, 0.19963, 0.17961]),
        ('rmse', [1156.62984, 1149.78765, 1149.80974, 1038.70899, 1038.88268, 1039.0164]),
        ('rmsle', [0.31401, 0.30202, 0.30642, 0.42822, 0.4468, 0.42381]),
    ],
)
def test_direct_error_estimation_yields_correct_results_for_metric_with_hypertuning(
    hypertuned_estimates, metric, expected
):
    sut = hypertuned_estimates.filter(period='analysis').to_df()
    all(round(sut.loc[:, (metric, 'value')], 5) == expected)


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('mae', [917.63663, 912.81705, 911.64143, 818.89734, 819.74329, 818.98411]),
        ('mape', [0.23839, 0.23956, 0.23966, 0.25599, 0.2568, 0.25406]),
        ('mse', [1337792.58904, 1322011.64262, 1322062.44625, 1078916.36436, 1079277.22531, 1079555.08966]),
        ('msle', [0.0986, 0.09122, 0.09389, 0.18337, 0.19963, 0.17961]),
        ('rmse', [1156.62984, 1149.78765, 1149.80974, 1038.70899, 1038.88268, 1039.0164]),
        ('rmsle', [0.31401, 0.30202, 0.30642, 0.42822, 0.4468, 0.42381]),
    ],
)
def test_direct_error_estimation_yields_correct_results_for_metric_with_custom_hyperparameters(
    custom_hyperparameter_estimates, metric, expected
):
    sut = custom_hyperparameter_estimates.filter(period='analysis').to_df()
    all(round(sut.loc[:, (metric, 'value')], 5) == expected)


def test_result_plot_raises_invalid_args_exception_when_given_incorrect_kind(estimates):
    with pytest.raises(InvalidArgumentsException):
        _ = estimates.plot(kind='foo')


# See https://github.com/NannyML/nannyml/issues/192
def test_dle_returns_distinct_but_consistent_results_when_reused(regression_data, direct_error_estimator):
    reference, analysis = regression_data

    # Get rid of negative values for log based metrics
    reference = reference[~(reference['y_pred'] < 0)]
    analysis = analysis[~(analysis['y_pred'] < 0)]

    direct_error_estimator.fit(reference)
    estimate1 = direct_error_estimator.estimate(analysis)
    estimate2 = direct_error_estimator.estimate(analysis)

    # Checks two distinct results are returned. Previously there was a bug causing the previous result instance to be
    # modified on subsequent estimates.
    assert estimate1 is not estimate2
    pd.testing.assert_frame_equal(estimate1.to_df(), estimate2.to_df())


# See https://github.com/NannyML/nannyml/issues/197
def test_dle_result_filter_should_preserve_data_with_default_args(estimates):
    filtered_result = estimates.filter()
    assert filtered_result.data.equals(estimates.data)


# See https://github.com/NannyML/nannyml/issues/197
def test_dle_result_filter_metrics(estimates):
    filtered_result = estimates.filter(metrics=["mae"])
    columns = tuple(set(metric for (metric, _) in filtered_result.data.columns if metric != "chunk"))
    assert columns == ("mae",)
    assert filtered_result.data.shape[0] == estimates.data.shape[0]


# See https://github.com/NannyML/nannyml/issues/197
def test_dle_result_filter_period(estimates):
    ref_period = estimates.data.loc[estimates.data.loc[:, ("chunk", "period")] == "reference", :]
    filtered_result = estimates.filter(period="reference")
    assert filtered_result.data.equals(ref_period)


@pytest.mark.parametrize(
    'custom_thresholds',
    [
        {'mae': ConstantThreshold(lower=1, upper=2)},
        {'mae': ConstantThreshold(lower=1, upper=2), 'mape': ConstantThreshold(lower=1, upper=2)},
        {
            'mae': ConstantThreshold(lower=1, upper=2),
            'mape': ConstantThreshold(lower=1, upper=2),
            'mse': ConstantThreshold(lower=1, upper=2),
        },
        {
            'mae': ConstantThreshold(lower=1, upper=2),
            'mape': ConstantThreshold(lower=1, upper=2),
            'mse': ConstantThreshold(lower=1, upper=2),
            'msle': ConstantThreshold(lower=1, upper=2),
        },
        {
            'mae': ConstantThreshold(lower=1, upper=2),
            'mape': ConstantThreshold(lower=1, upper=2),
            'mse': ConstantThreshold(lower=1, upper=2),
            'msle': ConstantThreshold(lower=1, upper=2),
            'rmse': ConstantThreshold(lower=1, upper=2),
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
def test_cbpe_with_custom_thresholds(custom_thresholds, regression_feature_columns):
    est = DLE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=10000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    sut = est.thresholds
    expected_thresholds = DEFAULT_THRESHOLDS
    expected_thresholds.update(**custom_thresholds)
    assert sut == expected_thresholds


def test_cbpe_with_default_thresholds(regression_feature_columns):
    est = DLE(
        timestamp_column_name='timestamp',
        y_pred='y_pred',
        y_true='y_true',
        feature_column_names=regression_feature_columns,
        chunk_size=10000,
        metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    )
    sut = est.thresholds

    assert sut == DEFAULT_THRESHOLDS


@pytest.mark.parametrize('metric', ['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'])
def test_result_plot_with_string_metric_returns_plotly_figure(estimates, direct_error_estimator, metric):
    _ = MetricFactory.create(
        key=metric,
        problem_type=ProblemType.REGRESSION,
        feature_column_names=direct_error_estimator.feature_column_names,
        y_true=direct_error_estimator.y_true,
        y_pred=direct_error_estimator.y_pred,
        chunker=direct_error_estimator.chunker,
        tune_hyperparameters=direct_error_estimator.tune_hyperparameters,
        hyperparameter_tuning_config=direct_error_estimator.hyperparameter_tuning_config,
        hyperparameters=direct_error_estimator.hyperparameters,
        threshold=StandardDeviationThreshold(),
    )

    sut = estimates.plot(metric=metric).to_dict()
    assert 'Metric' in sut['data'][3]['name']


@pytest.mark.parametrize('metric', ['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'])
def test_result_plot_with_metric_object_returns_plotly_figure(estimates, direct_error_estimator, metric):
    _metric = MetricFactory.create(
        key=metric,
        problem_type=ProblemType.REGRESSION,
        feature_column_names=direct_error_estimator.feature_column_names,
        y_true=direct_error_estimator.y_true,
        y_pred=direct_error_estimator.y_pred,
        chunker=direct_error_estimator.chunker,
        tune_hyperparameters=direct_error_estimator.tune_hyperparameters,
        hyperparameter_tuning_config=direct_error_estimator.hyperparameter_tuning_config,
        hyperparameters=direct_error_estimator.hyperparameters,
        threshold=StandardDeviationThreshold(),
    )

    sut = estimates.plot(metric=_metric)
    assert 'Metric' in sut.to_dict()['data'][3]['name']


@pytest.mark.parametrize('metric', ['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'])
def test_result_plot_contains_reference_data_when_plot_reference_set_to_true(estimates, metric):
    sut = estimates.plot(metric=metric, plot_reference=True)
    assert len(sut.to_dict()['data'][2]['x']) > 0
    assert len(sut.to_dict()['data'][2]['y']) > 0


@pytest.mark.parametrize(
    'estimator_args, plot_args',
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
def test_binary_classification_result_plots_raise_no_exceptions(estimator_args, plot_args):  # noqa: D103
    reference, analysis, analysis_targets = load_synthetic_car_price_dataset()
    est = DLE(
        feature_column_names=[col for col in reference.columns if col not in ['y_true', 'y_pred', 'timestamp']],
        y_true='y_true',
        y_pred='y_pred',
        metrics=['mae', 'mape'],
        **estimator_args,
    ).fit(reference)
    sut = est.estimate(analysis)

    try:
        _ = sut.plot(**plot_args)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")
