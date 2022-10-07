"""Unit tests for the PerformanceCalculator."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.drift.model_inputs.univariate.distance import DistanceDriftCalculator


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture()
def distance_drift_calculator(data) -> DistanceDriftCalculator:
    return DistanceDriftCalculator(
        feature_column_names=[
            col for col in data[0].columns if col not in ['y_pred_proba', 'y_pred', 'timestamp', 'work_home_actual']
        ],
        timestamp_column_name='timestamp',
        metrics=['jensen_shannon'],
    )


def test_rando(data):
    try:
        calc = DistanceDriftCalculator(
            feature_column_names=[
                col for col in data[0].columns if col not in ['y_pred_proba', 'y_pred', 'timestamp', 'work_home_actual']
            ],
            timestamp_column_name='timestamp',
            metrics=['jensen_shannon'],
        )
        _ = calc.fit(data[1])

    except Exception as exc:
        pytest.fail(f'Unexpected error occured: {exc}')
