#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Unit tests for the preprocessing module."""
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from nannyml.datasets import load_synthetic_binary_classification_dataset
from nannyml.exceptions import InvalidArgumentsException, InvalidReferenceDataException, MissingMetadataException
from nannyml.metadata import extract_metadata
from nannyml.metadata.base import NML_METADATA_COLUMNS, ModelMetadata
from nannyml.preprocessing import preprocess


@pytest.fixture
def data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: D103
    ref_df, ana_df, tgt_df = load_synthetic_binary_classification_dataset()
    ref_df['y_pred'] = ref_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)
    ana_df['y_pred'] = ana_df['y_pred_proba'].map(lambda p: p >= 0.8).astype(int)

    return ref_df, ana_df, tgt_df


@pytest.fixture
def metadata(data) -> ModelMetadata:  # noqa: D103
    md = extract_metadata(data[0], model_type='classification_binary')
    md.target_column_name = 'work_home_actual'
    return md


def test_preprocess_raises_missing_metadata_exception_when_metadata_is_not_complete(data, metadata):  # noqa: D103
    analysis_data = data[0]
    metadata.partition_column_name = None
    analysis_data.drop(columns=['partition'], inplace=True)

    with pytest.raises(MissingMetadataException):
        _ = preprocess(data=analysis_data, metadata=metadata)


def test_preprocess_adds_metadata_columns_to_result(data, metadata):  # noqa: D103
    reference_data = data[0]
    sut = preprocess(reference_data, metadata)
    for col in NML_METADATA_COLUMNS:
        assert col in sut.columns


def test_preprocess_should_raise_warning_when_predicted_probabilities_outside_of_bounds(data, metadata):  # noqa: D103
    analysis_data = data[0]
    analysis_data.loc[10, 'output'] = 5
    metadata.predicted_probability_column_name = 'output'

    with pytest.warns(
        UserWarning,
        match="the predicted probabilities column 'output' contains "
        "values outside of the accepted \\[0, 1\\] interval",
    ):
        _ = preprocess(analysis_data, metadata)


def test_preprocess_should_raise_warning_when_predicted_probabilities_have_too_few_unique_values(  # noqa: D103
    data, metadata
):
    analysis_data = data[1]
    analysis_data[metadata.predicted_probability_column_name] = 0.20
    with pytest.warns(
        UserWarning,
        match=f"the predicted probabilities column '{metadata.predicted_probability_column_name}' "
        "contains fewer than 2 "
        "unique values.",
    ):
        _ = preprocess(analysis_data, metadata)


def test_preprocess_should_not_fail_when_no_predicted_probabilities_were_set(data, metadata):  # noqa: D103
    metadata.predicted_probability_column_name = None
    try:
        _ = preprocess(data[0], metadata)
    except Exception as exc:
        pytest.fail(f"an unexpected exception occurred: {exc}")


def test_preprocess_should_raise_invalid_ref_data_exception_when_contains_nan_predictions(data, metadata):  # noqa: D103
    ref_data = data[0]
    ref_data.loc[:10, metadata.prediction_column_name] = np.NAN

    with pytest.raises(
        InvalidReferenceDataException,
        match=f"prediction column '{metadata.prediction_column_name}' contains NaN values.",
    ):
        preprocess(ref_data, metadata, reference=True)


def test_preprocess_should_raise_invalid_ref_data_exception_when_contains_nan_pred_proba(data, metadata):  # noqa: D103
    ref_data = data[0]
    ref_data.loc[:10, metadata.predicted_probability_column_name] = np.NAN

    with pytest.raises(
        InvalidReferenceDataException,
        match=f"predicted probability column '{metadata.predicted_probability_column_name}' " f"contains NaN values.",
    ):
        preprocess(ref_data, metadata, reference=True)


def test_preprocess_should_raise_invalid_ref_data_exception_when_contains_nan_target(data, metadata):  # noqa: D103
    ref_data = data[0]
    ref_data.loc[:10, metadata.target_column_name] = np.NAN

    with pytest.raises(
        InvalidReferenceDataException, match=f"target column '{metadata.target_column_name}' " f"contains NaN values."
    ):
        preprocess(ref_data, metadata, reference=True)


def test_preprocess_should_raise_invalid_ref_data_exception_when_ref_data_emtpy(data, metadata):  # noqa: D103
    ref_data = pd.DataFrame(columns=data[0].columns)

    with pytest.raises(InvalidArgumentsException, match="provided data cannot be empty."):
        preprocess(ref_data, metadata, reference=True)
