#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Multivariate Domain Classifier package."""

from typing import Tuple

import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_car_loan_dataset

# from nannyml._typing import Result
from nannyml.drift.multivariate.domain_classifier.calculator import DomainClassifierCalculator

column_names1 = [
    'salary_range',
    'repaid_loan_on_prev_car',
    'size_of_downpayment',
    'car_value',
    'debt_to_income_ratio',
    'loan_length',
    'driver_tenure',
]


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_car_loan_dataset()
    return ref_df.head(15_000), ana_df.tail(10_000)


def test_default_cdd_run(binary_classification_data):
    """Test a default run of DC."""
    (
        reference,
        analysis,
    ) = binary_classification_data
    calc = DomainClassifierCalculator(feature_column_names=column_names1, chunk_size=5_000)
    calc.fit(reference)
    results = calc.calculate(analysis)
    assert list(results.to_df().loc[:, ("domain_classifier_auroc", "value")].round(4)) == [
        0.5020,
        0.5002,
        0.5174,
        0.9108,
        0.9136,
    ]
    assert list(results.to_df().loc[:, ("domain_classifier_auroc", "alert")]) == [False, False, False, True, True]


def test_cdd_run_w_timestamp(binary_classification_data):
    """Test a default run of DC."""
    (
        reference,
        analysis,
    ) = binary_classification_data
    calc = DomainClassifierCalculator(
        feature_column_names=column_names1,
        chunk_size=5_000,
        timestamp_column_name='timestamp'
    )
    calc.fit(reference.sample(frac=1).reset_index(drop=True))
    results = calc.calculate(analysis)
    assert list(results.to_df().loc[:, ("domain_classifier_auroc", "value")].round(4)) == [
        0.5020,
        0.5002,
        0.5174,
        0.9108,
        0.9136,
    ]
    assert list(results.to_df().loc[:, ("domain_classifier_auroc", "alert")]) == [False, False, False, True, True]
