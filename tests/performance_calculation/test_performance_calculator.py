#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the PerformanceCalculator."""
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from nannyml.datasets import load_synthetic_sample
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata import ModelMetadata, extract_metadata
from nannyml.performance_calculation import PerformanceCalculator
from nannyml.performance_calculation.metrics import AUROC, F1, Metric


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_sample()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture
def metadata(data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(data[0])
    md.target_column_name = 'work_home_actual'
    return md


class DummyMetric(Metric):
    """Dummy metric for testing."""

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _minimum_chunk_size(self) -> int:
        return 100

    def __init__(self):  # noqa: D107
        super().__init__(display_name='dummy_metric', column_name='dummy_metric')

    def _calculate(self, data: pd.DataFrame):
        return 0.5


@pytest.fixture
def dummy_metric() -> Metric:  # noqa: D103
    return DummyMetric()


def test_calculator_init_with_empty_metrics_should_not_fail(metadata):  # noqa: D103
    try:
        _ = PerformanceCalculator(model_metadata=metadata, metrics=[])
    except Exception as exc:
        pytest.fail(f'unexpected exception: {exc}')


def test_calculator_init_should_set_metrics(metadata):  # noqa: D103
    sut = PerformanceCalculator(model_metadata=metadata, metrics=['roc_auc', 'f1']).metrics
    assert len(sut) == 2
    assert sut[0] == AUROC()
    assert sut[1] == F1()


def test_calculator_init_should_deal_with_mixed_metric_keys(metadata):  # noqa: D103
    sut = PerformanceCalculator(model_metadata=metadata, metrics=['roc_auc', Metric('custom', f1_score)]).metrics
    assert len(sut) == 2
    assert sut[0] == AUROC()
    assert sut[1] == Metric('custom', f1_score)


def test_calculator_calculate_should_raise_invalid_args_exception_when_no_target_data_present(  # noqa: D103
    data, metadata
):
    calc = PerformanceCalculator(model_metadata=metadata, metrics=['roc_auc', 'f1'])
    calc.fit(reference_data=data[0])
    with pytest.raises(InvalidArgumentsException):
        _ = calc.calculate(analysis_data=data[1])


def test_calculator_calculate_should_include_chunk_information_columns(data, metadata, dummy_metric):  # noqa: D103
    calc = PerformanceCalculator(model_metadata=metadata, metrics=[dummy_metric, 'roc_auc'], chunk_size=5000)
    calc.fit(reference_data=data[0])
    ref_with_tgt = data[0].join(data[2], on='identifier', rsuffix='_r')
    sut = calc.calculate(analysis_data=ref_with_tgt)

    assert 'key' in sut.data.columns
    assert 'start_index' in sut.data.columns
    assert 'end_index' in sut.data.columns
    assert 'start_date' in sut.data.columns
    assert 'end_date' in sut.data.columns
    assert 'partition' in sut.data.columns


def test_calculator_calculate_should_include_target_completeness_rate(data, metadata, dummy_metric):  # noqa: D103
    # Let's artificially modify the target completeness of different chunks.
    ref_data = data[0]

    # Drop 10% of the target values in the first chunk
    ref_data.loc[0:499, metadata.target_column_name] = np.NAN

    # Drop 90% of the target values in the second chunk
    ref_data.loc[5000:9499, metadata.target_column_name] = np.NAN

    calc = PerformanceCalculator(model_metadata=metadata, metrics=[dummy_metric], chunk_size=5000)
    calc.fit(reference_data=ref_data)
    sut = calc.calculate(analysis_data=ref_data)

    assert 'targets_missing_rate' in sut.data.columns
    assert sut.data.loc[0, 'targets_missing_rate'] == 0.1
    assert sut.data.loc[1, 'targets_missing_rate'] == 0.9


def test_calculator_calculates_minimum_chunk_size_for_each_metric(data, metadata):  # noqa: D103
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']
    for metric in metrics:
        calc = PerformanceCalculator(model_metadata=metadata, metrics=[metric])
        calc.fit(reference_data=data[0])
        assert calc._minimum_chunk_size > 0
        assert calc._minimum_chunk_size <= len(data[0])
