#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Tests for Multivariate Classifier for Drift Detection package."""

import numpy as np
import pandas as pd
import pytest
# from sklearn.impute import SimpleImputer
from typing import Tuple

from nannyml._typing import Result
from nannyml.chunk import PeriodBasedChunker, SizeBasedChunker
from nannyml.drift.multivariate.classifier_for_drift_detection.calculator import ClassifierForDriftDetectionCalculator
from nannyml.datasets import (
    load_synthetic_car_loan_dataset,
    # load_synthetic_multiclass_classification_dataset,
)
# from nannyml.drift.univariate import UnivariateDriftCalculator
# from nannyml.performance_estimation.confidence_based import CBPE

column_names1 = [
    'salary_range',
    'repaid_loan_on_prev_car',
    'size_of_downpayment',
    'car_value',
    'debt_to_income_ratio',
    'loan_length',
    'driver_tenure'
]


@pytest.fixture
def binary_classification_data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_car_loan_dataset()
    return ref_df.head(15_000), ana_df.tail(10_000)

def test_default_cdd_run(binary_classification_data):
    reference, analysis, = binary_classification_data
    calc = ClassifierForDriftDetectionCalculator(
        feature_column_names=column_names1,
        chunk_size=5_000
    )
    calc.fit(reference)
    results = calc.calculate(analysis)
    print(results.categorical_column_names)
    print(results.to_df())
    # print(results.to_df()[:,6:])
    assert list(results.to_df().loc[:,("cdd_discrimination", "value")].round(4)) == [0.5020, 0.5002, 0.5174, 0.9108, 0.9136]
    assert list(results.to_df().loc[:,("cdd_discrimination", "alert")]) == [False, False, False, True, True]
