#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit testing for CBPE."""
from typing import Tuple

import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_sample
from nannyml.metadata import ModelMetadata, extract_metadata
from nannyml.performance_estimation import CBPE


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, _ = load_synthetic_sample()
    return ref_df, ana_df


@pytest.fixture
def metadata(data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(data[0])
    md.target_column_name = 'work_home_actual'
    return md


def test_estimator_will_calibrate_scores_when_needed(metadata, data):  # noqa: D103
    ref_df = data[0]

    sut = CBPE(model_metadata=metadata)
    sut.fit(ref_df)

    assert sut.needs_calibration is True


def test_estimator_will_not_calibrate_scores_when_not_needed(metadata, data):  # noqa: D103
    ref_df = data[0]
    # If predictions equal targets no calibration will be required
    ref_df[metadata.predicted_probability_column_name] = ref_df[metadata.target_column_name]

    sut = CBPE(model_metadata=metadata)
    sut.fit(ref_df)

    assert sut.needs_calibration is False


def test_estimator_will_not_fail_on_work_from_home_sample(metadata, data):  # noqa: D103
    reference, analysis = data
    try:
        estimator = CBPE(model_metadata=metadata)
        estimator.fit(reference)
        _ = estimator.estimate(analysis)
    except Exception as exc:
        pytest.fail(f'unexpected exception was raised: {exc}')
