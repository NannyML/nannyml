#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for performance metrics."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml import PerformanceCalculator
from nannyml._typing import ProblemType
from nannyml.chunk import DefaultChunker
from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.performance_calculation.metrics.base import MetricFactory
from nannyml.performance_calculation.metrics.binary_classification import (
    BinaryClassificationAccuracy,
    BinaryClassificationAUROC,
    BinaryClassificationBusinessValue,
    BinaryClassificationConfusionMatrix,
    BinaryClassificationF1,
    BinaryClassificationPrecision,
    BinaryClassificationRecall,
    BinaryClassificationSpecificity,
)
from nannyml.thresholds import ConstantThreshold, StandardDeviationThreshold


@pytest.fixture(scope='module')
def binary_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


def performance_calculator(timestamp_column_name: Optional[str] = 'timestamp') -> PerformanceCalculator:
    return PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        metrics=[
            'roc_auc',
            'f1',
            'precision',
            'recall',
            'specificity',
            'accuracy',
            'business_value',
            'confusion_matrix',
        ],
        business_value_matrix=[[0, -10], [5, 0]],
        problem_type='classification_binary',
    )


@pytest.fixture(scope='module')
def realized_performance_metrics(binary_data) -> pd.DataFrame:
    calculator = performance_calculator().fit(binary_data[0])
    results = calculator.calculate(binary_data[1].merge(binary_data[2], on='identifier')).filter(period='analysis')
    return results.data

@pytest.fixture(scope='module')
def realized_performance_alt_cm_pred(binary_data) -> pd.DataFrame:
    calculator = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        metrics=[
            'business_value',
            'confusion_matrix',
        ],
        business_value_matrix=[[-50, -10], [-5, 10]],
        normalize_business_value='per_prediction',
        normalize_confusion_matrix='pred',
        problem_type='classification_binary',
    ).fit(binary_data[0])
    results = calculator.calculate(binary_data[1].merge(binary_data[2], on='identifier')).filter(period='analysis')
    return results.data

@pytest.fixture(scope='module')
def realized_performance_alt_cm_true(binary_data) -> pd.DataFrame:
    calculator = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='work_home_actual',
        metrics=[
            'confusion_matrix',
        ],
        # business_value_matrix=[[-50, -10], [-5, 10]],
        # normalize_business_value='per_prediction',
        normalize_confusion_matrix='true',
        problem_type='classification_binary',
    ).fit(binary_data[0])
    results = calculator.calculate(binary_data[1].merge(binary_data[2], on='identifier')).filter(period='analysis')
    return results.data

@pytest.fixture(scope='module')
def no_timestamp_metrics(binary_data):
    calc = performance_calculator(timestamp_column_name=None).fit(binary_data[0])
    results = calc.calculate(binary_data[1].merge(binary_data[2], on='identifier')).filter(period='analysis')
    return results.data


@pytest.fixture(scope='module')
def partial_target_metrics(binary_data):
    partial_targets = binary_data[2][: len(binary_data[2]) // 2]
    analysis_data = binary_data[1].merge(partial_targets, on='identifier', how='left')

    calc = performance_calculator().fit(binary_data[0])
    results = calc.calculate(analysis_data).filter(period='analysis')
    return results.data


@pytest.mark.parametrize(
    'key,problem_type,metric',
    [
        ('roc_auc', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationAUROC),
        ('f1', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationF1),
        ('precision', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationPrecision),
        ('recall', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationRecall),
        ('specificity', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationSpecificity),
        ('accuracy', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationAccuracy),
        ('business_value', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationBusinessValue),
        ('confusion_matrix', ProblemType.CLASSIFICATION_BINARY, BinaryClassificationConfusionMatrix),
    ],
)
def test_metric_factory_returns_correct_metric_given_key_and_problem_type(key, problem_type, metric):  # noqa: D103
    calc = PerformanceCalculator(
        timestamp_column_name='timestamp',
        y_pred_proba='y_pred_proba',
        y_pred='y_pred',
        y_true='y_true',
        metrics=['roc_auc', 'f1'],
        problem_type='classification_binary',
    )
    sut = MetricFactory.create(
        key,
        problem_type,
        y_true=calc.y_true,
        y_pred=calc.y_pred,
        y_pred_proba=calc.y_pred_proba,
        threshold=StandardDeviationThreshold(),
        business_value_matrix=[[0, -10], [5, 0]],
    )
    assert sut == metric(
        y_true=calc.y_true,
        y_pred=calc.y_pred,
        y_pred_proba=calc.y_pred_proba,
        threshold=StandardDeviationThreshold,
        business_value_matrix=[[0, -10], [5, 0]],
    )


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('roc_auc', [0.97096, 0.97025, 0.97628, 0.96772, 0.96989, 0.96005, 0.95853, 0.95904, 0.96309, 0.95756]),
        ('f1', [0.92186, 0.92124, 0.92678, 0.91684, 0.92356, 0.87424, 0.87672, 0.86806, 0.883, 0.86775]),
        ('precision', [0.96729, 0.96607, 0.96858, 0.96819, 0.9661, 0.94932, 0.95777, 0.95012, 0.95718, 0.94271]),
        ('recall', [0.88051, 0.88039, 0.88843, 0.87067, 0.8846, 0.81017, 0.80832, 0.79904, 0.8195, 0.80383]),
        ('specificity', [0.9681, 0.9701, 0.97277, 0.9718, 0.96864, 0.95685, 0.96364, 0.95795, 0.96386, 0.94879]),
        ('accuracy', [0.9228, 0.926, 0.9318, 0.9216, 0.9264, 0.8836, 0.8852, 0.8784, 0.8922, 0.8746]),
        ('business_value', [775, 710, 655, 895, 670, 1290, 1520, 1465, 1330, 1260]),
        ('true_positive', [2277, 2164, 2158, 2161, 2223, 2023, 2041, 2000, 2034, 2057]),
        ('false_positive', [77, 76, 70, 71, 78, 108, 90, 105, 91, 125]),
        ('true_negative', [2337, 2466, 2501, 2447, 2409, 2395, 2385, 2392, 2427, 2316]),
        ('false_negative', [309, 294, 271, 321, 290, 474, 484, 503, 448, 502]),
    ],
)
def test_metric_values_are_calculated_correctly(realized_performance_metrics, metric, expected):
    metric_values = realized_performance_metrics.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()

@pytest.mark.parametrize(
    'metric, expected',
    [
        ('business_value', [-35.39910, -35.88509, -36.22902, -35.41763, -35.84284, -33.57896, -33.25318, -33.17897, -33.84430, -33.12941]),
        ('true_positive', [0.96729, 0.96607, 0.96858, 0.96819, 0.96610, 0.94932, 0.95777, 0.95012, 0.95718, 0.94271]),
        ('false_positive', [0.03271, 0.03393, 0.03142, 0.03181, 0.03390, 0.05068, 0.04223, 0.04988, 0.04282, 0.05729]),
        ('true_negative', [0.88322, 0.89348, 0.90224, 0.88403, 0.89255, 0.83479, 0.83130, 0.82625, 0.84417, 0.82186]),
        ('false_negative', [0.11678, 0.10652, 0.09776, 0.11597, 0.10745, 0.16521, 0.16870, 0.17375, 0.15583, 0.17814]),
    ],
)
def test_alt_cm_pred_values_are_calculated_correctly(realized_performance_alt_cm_pred, metric, expected):
    metric_values = realized_performance_alt_cm_pred.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()

@pytest.mark.parametrize(
    'metric, expected',
    [
        ('true_positive', [0.88051, 0.88039, 0.88843, 0.87067, 0.88460, 0.81017, 0.80832, 0.79904, 0.81950, 0.80383]),
        ('false_positive', [0.03190, 0.02990, 0.02723, 0.02820, 0.03136, 0.04315, 0.03636, 0.04205, 0.03614, 0.05121]),
        ('true_negative', [0.96810, 0.97010, 0.97277, 0.97180, 0.96864, 0.95685, 0.96364, 0.95795, 0.96386, 0.94879]),
        ('false_negative', [0.11949, 0.11961, 0.11157, 0.12933, 0.11540, 0.18983, 0.19168, 0.20096, 0.18050, 0.19617]),
    ],
)
def test_alt_cm_true_values_are_calculated_correctly(realized_performance_alt_cm_true, metric, expected):
    metric_values = realized_performance_alt_cm_true.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('roc_auc', [0.97096, 0.97025, 0.97628, 0.96772, 0.96989, 0.96005, 0.95853, 0.95904, 0.96309, 0.95756]),
        ('f1', [0.92186, 0.92124, 0.92678, 0.91684, 0.92356, 0.87424, 0.87672, 0.86806, 0.883, 0.86775]),
        ('precision', [0.96729, 0.96607, 0.96858, 0.96819, 0.9661, 0.94932, 0.95777, 0.95012, 0.95718, 0.94271]),
        ('recall', [0.88051, 0.88039, 0.88843, 0.87067, 0.8846, 0.81017, 0.80832, 0.79904, 0.8195, 0.80383]),
        ('specificity', [0.9681, 0.9701, 0.97277, 0.9718, 0.96864, 0.95685, 0.96364, 0.95795, 0.96386, 0.94879]),
        ('accuracy', [0.9228, 0.926, 0.9318, 0.9216, 0.9264, 0.8836, 0.8852, 0.8784, 0.8922, 0.8746]),
        ('business_value', [775, 710, 655, 895, 670, 1290, 1520, 1465, 1330, 1260]),
        ('true_positive', [2277, 2164, 2158, 2161, 2223, 2023, 2041, 2000, 2034, 2057]),
        ('false_positive', [77, 76, 70, 71, 78, 108, 90, 105, 91, 125]),
        ('true_negative', [2337, 2466, 2501, 2447, 2409, 2395, 2385, 2392, 2427, 2316]),
        ('false_negative', [309, 294, 271, 321, 290, 474, 484, 503, 448, 502]),
    ],
)
def test_metric_values_without_timestamp_are_calculated_correctly(no_timestamp_metrics, metric, expected):
    metric_values = no_timestamp_metrics.loc[:, (metric, 'value')]
    assert (round(metric_values, 5) == expected).all()


@pytest.mark.parametrize(
    'metric, expected',
    [
        ('roc_auc', [0.97096, 0.97025, 0.97628, 0.96772, 0.96989, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('f1', [0.92186, 0.92124, 0.92678, 0.91684, 0.92356, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('precision', [0.96729, 0.96607, 0.96858, 0.96819, 0.9661, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('recall', [0.88051, 0.88039, 0.88843, 0.87067, 0.8846, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('specificity', [0.9681, 0.9701, 0.97277, 0.9718, 0.96864, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('accuracy', [0.9228, 0.926, 0.9318, 0.9216, 0.9264, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('business_value', [775, 710, 655, 895, 670, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('true_positive', [2277, 2164, 2158, 2161, 2223, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('false_positive', [77, 76, 70, 71, 78, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('true_negative', [2337, 2466, 2501, 2447, 2409, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
        ('false_negative', [309, 294, 271, 321, 290, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]),
    ],
)
def test_metric_values_with_partial_targets_are_calculated_correctly(partial_target_metrics, metric, expected):
    metric_values = partial_target_metrics.loc[:, (metric, 'value')]
    assert np.array_equal(round(metric_values, 5), expected, equal_nan=True)


@pytest.mark.parametrize(
    'metric_cls',
    [
        BinaryClassificationAUROC,
        BinaryClassificationF1,
        BinaryClassificationPrecision,
        BinaryClassificationRecall,
        BinaryClassificationSpecificity,
        BinaryClassificationAccuracy,
    ],
)
def test_metric_logs_warning_when_lower_threshold_is_overridden_by_metric_limits(caplog, metric_cls, binary_data):
    reference = binary_data[0]
    metric = metric_cls(
        y_pred_proba='y_pred_proba', y_pred='y_pred', y_true='work_home_actual', threshold=ConstantThreshold(lower=-1)
    )
    metric.fit(reference, chunker=DefaultChunker())

    assert (
        f'{metric.display_name} lower threshold value -1 overridden by '
        f'lower threshold value limit {metric.lower_threshold_value_limit}' in caplog.messages
    )


@pytest.mark.parametrize(
    'metric_cls',
    [
        BinaryClassificationAUROC,
        BinaryClassificationF1,
        BinaryClassificationPrecision,
        BinaryClassificationRecall,
        BinaryClassificationSpecificity,
        BinaryClassificationAccuracy,
    ],
)
def test_metric_logs_warning_when_upper_threshold_is_overridden_by_metric_limits(caplog, metric_cls, binary_data):
    reference = binary_data[0]
    metric = metric_cls(
        y_pred_proba='y_pred_proba', y_pred='y_pred', y_true='work_home_actual', threshold=ConstantThreshold(upper=2)
    )
    metric.fit(reference, chunker=DefaultChunker())

    assert (
        f'{metric.display_name} upper threshold value 2 overridden by '
        f'upper threshold value limit {metric.upper_threshold_value_limit}' in caplog.messages
    )
