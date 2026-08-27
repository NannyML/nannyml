import numpy as np
import pandas as pd
import pytest

from nannyml import DataReconstructionDriftCalculator, DomainClassifierCalculator


def test_data_reconstruction_exposes_per_row_reconstruction_norms():
    rng = np.random.default_rng(13)
    reference = pd.DataFrame({'x': rng.normal(size=24), 'y': rng.normal(size=24)})
    analysis = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'x': [0.0, 1.0, 4.0, 8.0], 'y': [0.0, 1.0, 4.0, -8.0]})
    calculator = DataReconstructionDriftCalculator(column_names=['x', 'y'], n_components=1, chunk_size=4).fit(reference)

    result = calculator.calculate(analysis, include_instance_metrics=True, identifier_column_name='id')
    instance = result.to_instance_df()

    assert instance['id'].tolist() == analysis['id'].tolist()
    chunk_value = result.filter(period='analysis').to_df()[('reconstruction_error', 'value')].iloc[0]
    assert instance['reconstruction_error_contribution'].mean() == pytest.approx(chunk_value)


def test_domain_classifier_exposes_analysis_oof_mann_whitney_scores():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({'x': rng.normal(0, 1, 60)})
    analysis = pd.DataFrame({'id': [f'row-{i}' for i in range(20)], 'x': rng.normal(2, 1, 20)})
    calculator = DomainClassifierCalculator(
        feature_column_names=['x'],
        chunk_size=20,
        cv_folds_num=2,
        hyperparameters={'n_estimators': 10, 'num_leaves': 7, 'random_state': 13, 'verbose': -1},
    ).fit(reference)

    result = calculator.calculate(analysis, include_instance_metrics=True, identifier_column_name='id')
    instance = result.to_instance_df()

    assert instance['id'].tolist() == analysis['id'].tolist()
    assert instance['domain_classifier_auroc_contribution'].between(0, 1).all()
    chunk_value = result.filter(period='analysis').to_df()[('domain_classifier_auroc', 'value')].iloc[0]
    assert instance['domain_classifier_auroc_contribution'].mean() == pytest.approx(chunk_value)
