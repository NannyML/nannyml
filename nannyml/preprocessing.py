#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Preprocessing pipeline for incoming data."""
import logging
import warnings

import pandas as pd

from nannyml.exceptions import MissingMetadataException
from nannyml.metadata import ModelMetadata

logger = logging.getLogger(__name__)

PREDICTED_PROBABILITIES_UNIQUE_VALUES_THRESHOLD = 2


def preprocess(data: pd.DataFrame, model_metadata: ModelMetadata) -> pd.DataFrame:
    """Analyse and prepare incoming data for further use downstream.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing model inputs, scores, targets and other metadata.
    model_metadata: ModelMetadata
        Optional ModelMetadata instance that might have been manually constructed
        or contains non-default values

    Returns
    -------
    prepped_data: Optional[DataFrame]
        A copy of the uploaded data with added copies of metadata columns
        Will be ``None`` when the extracted/provided metadata was not complete.

    """
    # Check metadata for completeness.
    metadata_complete, missing_properties = model_metadata.is_complete()  # type: ignore
    if not metadata_complete:
        raise MissingMetadataException(
            f'metadata is still missing values for {missing_properties}.\n'
            'Please rectify by renaming columns following automated extraction conventions\n'
            'and re-running preprocessing or set metadata properties manually.\n'
            'See https://docs.nannyml.com/metadata-extraction for more information\n'
        )

    # If complete then add copies of metadata columns
    prepped_data = model_metadata.enrich(data)

    # Check if predicted probability values don't contain (binary) prediction values instead
    _check_predicted_probabilities_are_probabilities(model_metadata, data)

    return prepped_data


def _check_predicted_probabilities_are_probabilities(model_metadata: ModelMetadata, data: pd.DataFrame):
    if model_metadata.predicted_probability_column_name is None:
        return

    predicted_probabilities = data[model_metadata.predicted_probability_column_name]

    values_within_bounds = predicted_probabilities.between(0, 1).all()
    if not values_within_bounds:
        warnings.warn(
            message=f"the predicted probabilities column '{model_metadata.predicted_probability_column_name}'"
            f" contains values outside of the accepted [0, 1] interval. "
            f"Please ensure you are not providing predictions instead."
        )

    enough_unique_values = predicted_probabilities.nunique() > PREDICTED_PROBABILITIES_UNIQUE_VALUES_THRESHOLD
    if not enough_unique_values:
        warnings.warn(
            message=f"the predicted probabilities column '{model_metadata.predicted_probability_column_name}'"
            f" contains fewer than {PREDICTED_PROBABILITIES_UNIQUE_VALUES_THRESHOLD} unique values. "
            f"Please ensure you are not providing predictions instead."
        )
