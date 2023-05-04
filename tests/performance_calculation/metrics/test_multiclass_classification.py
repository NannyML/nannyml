#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0
#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml import PerformanceCalculator
from nannyml._typing import ProblemType
from nannyml.chunk import DefaultChunker
from nannyml.datasets import load_synthetic_multiclass_classification_dataset
from nannyml.performance_calculation.metrics.base import MetricFactory
from nannyml.performance_calculation.metrics.multiclass_classification import (
    MulticlassClassificationAccuracy,
    MulticlassClassificationAUROC,
    MulticlassClassificationF1,
    MulticlassClassificationPrecision,
    MulticlassClassificationRecall,
    MulticlassClassificationSpecificity,
)
from nannyml.thresholds import ConstantThreshold, StandardDeviationThreshold


@pytest.fixture(scope='module')
def multiclass_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_multiclass_classification_dataset()

    return ref_df, ana_df, tgt_df


@pytest.fixture(scope='module')
def performance_calculator() -> PerformanceCalculator:
    return PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba={
            'prepaid_card': 'y_pred_proba_prepaid_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'upmarket_card': 'y_pred_proba_upmarket_card',
        },
        y_pred='y_pred',
        y_true='y_true',
        metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
        problem_type='classification_multiclass',
    )


@pytest.fixture(scope='module')
def realized_performance_metrics(multiclass_data) -> pd.DataFrame:
    performance_calculator = PerformanceCalculator(
        y_pred_proba={
            'prepaid_card': 'y_pred_proba_prepaid_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'upmarket_card': 'y_pred_proba_upmarket_card',
        },
        y_pred='y_pred',
        y_true='y_true',
        metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
        problem_type='classification_multiclass',
    ).fit(multiclass_data[0])
    results = performance_calculator.calculate(
        multiclass_data[1].merge(multiclass_data[2], left_index=True, right_index=True)
    ).filter(period='analysis')
    return results.to_df()


@pytest.fixture(scope='module')
def no_timestamp_metrics(performance_calculator, multiclass_data) -> pd.DataFrame:
    performance_calculator.fit(multiclass_data[0])
    results = performance_calculator.calculate(
        multiclass_data[1].merge(multiclass_data[2], left_index=True, right_index=True)
    ).filter(period='analysis')
    return results.data


@pytest.mark.parametrize(
    'key,problem_type,metric',
    [
        ('roc_auc', ProblemType.CLASSIFICATION_MULTICLASS, MulticlassClassificationAUROC),
        ('f1', ProblemType.CLASSIFICATION_MULTICLASS, MulticlassClassificationF1),
        ('precision', ProblemType.CLASSIFICATION_MULTICLASS, MulticlassClassificationPrecision),
        ('recall', ProblemType.CLASSIFICATION_MULTICLASS, MulticlassClassificationRecall),
        ('specificity', ProblemType.CLASSIFICATION_MULTICLASS, MulticlassClassificationSpecificity),
        ('accuracy', ProblemType.CLASSIFICATION_MULTICLASS, MulticlassClassificationAccuracy),
    ],
)
def test_metric_factory_returns_correct_metric_given_key_and_problem_type(key, problem_type, metric):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_multiclass',
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
        y_true=calc.y_true, y_pred=calc.y_pred, y_pred_proba=calc.y_pred_proba, threshold=StandardDeviationThreshold
    )


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('roc_auc', [0.90759, 0.91053, 0.90941, 0.91158, 0.90753, 0.74859, 0.75114, 0.7564, 0.75856, 0.75394]),
        ('f1', [0.7511, 0.76305, 0.75849, 0.75894, 0.75796, 0.55711, 0.55915, 0.56506, 0.5639, 0.56164]),
        ('precision', [0.75127, 0.76313, 0.7585, 0.75897, 0.75795, 0.5597, 0.56291, 0.56907, 0.56667, 0.56513]),
        ('recall', [0.75103, 0.76315, 0.75848, 0.75899, 0.75798, 0.55783, 0.56017, 0.56594, 0.56472, 0.56277]),
        ('specificity', [0.87555, 0.88151, 0.87937, 0.87963, 0.87899, 0.77991, 0.78068, 0.78422, 0.78342, 0.78243]),
        ('accuracy', [0.75117, 0.763, 0.75867, 0.75917, 0.758, 0.56083, 0.56233, 0.56983, 0.56783, 0.566]),
    ],
)
def test_metric_values_are_calculated_correctly(realized_performance_metrics, metric, expected):
    metric_values = realized_performance_metrics.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('roc_auc', [0.90759, 0.91053, 0.90941, 0.91158, 0.90753, 0.74859, 0.75114, 0.7564, 0.75856, 0.75394]),
        ('f1', [0.7511, 0.76305, 0.75849, 0.75894, 0.75796, 0.55711, 0.55915, 0.56506, 0.5639, 0.56164]),
        ('precision', [0.75127, 0.76313, 0.7585, 0.75897, 0.75795, 0.5597, 0.56291, 0.56907, 0.56667, 0.56513]),
        ('recall', [0.75103, 0.76315, 0.75848, 0.75899, 0.75798, 0.55783, 0.56017, 0.56594, 0.56472, 0.56277]),
        ('specificity', [0.87555, 0.88151, 0.87937, 0.87963, 0.87899, 0.77991, 0.78068, 0.78422, 0.78342, 0.78243]),
        ('accuracy', [0.75117, 0.763, 0.75867, 0.75917, 0.758, 0.56083, 0.56233, 0.56983, 0.56783, 0.566]),
    ],
)
def test_metric_values_without_timestamps_are_calculated_correctly(no_timestamp_metrics, metric, expected):
    metric_values = no_timestamp_metrics.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()


@pytest.mark.parametrize(
    'metric_cls',
    [
        MulticlassClassificationAUROC,
        MulticlassClassificationF1,
        MulticlassClassificationPrecision,
        MulticlassClassificationRecall,
        MulticlassClassificationSpecificity,
        MulticlassClassificationAccuracy,
    ],
)
def test_metric_logs_warning_when_lower_threshold_is_overridden_by_metric_limits(caplog, metric_cls, multiclass_data):
    reference = multiclass_data[0]
    metric = metric_cls(
        y_pred_proba={
            'prepaid_card': 'y_pred_proba_prepaid_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'upmarket_card': 'y_pred_proba_upmarket_card',
        },
        y_pred='y_pred',
        y_true='y_true',
        threshold=ConstantThreshold(lower=-1),
    )
    metric.fit(reference, chunker=DefaultChunker())

    assert (
        f'{metric.display_name} lower threshold value -1 overridden by '
        f'lower threshold value limit {metric.lower_threshold_value_limit}' in caplog.messages
    )


@pytest.mark.parametrize(
    'metric_cls',
    [
        MulticlassClassificationAUROC,
        MulticlassClassificationF1,
        MulticlassClassificationPrecision,
        MulticlassClassificationRecall,
        MulticlassClassificationSpecificity,
        MulticlassClassificationAccuracy,
    ],
)
def test_metric_logs_warning_when_upper_threshold_is_overridden_by_metric_limits(caplog, metric_cls, multiclass_data):
    reference = multiclass_data[0]
    metric = metric_cls(
        y_pred_proba={
            'prepaid_card': 'y_pred_proba_prepaid_card',
            'highstreet_card': 'y_pred_proba_highstreet_card',
            'upmarket_card': 'y_pred_proba_upmarket_card',
        },
        y_pred='y_pred',
        y_true='y_true',
        threshold=ConstantThreshold(upper=2),
    )
    metric.fit(reference, chunker=DefaultChunker())

    assert (
        f'{metric.display_name} upper threshold value 2 overridden by '
        f'upper threshold value limit {metric.upper_threshold_value_limit}' in caplog.messages
    )
