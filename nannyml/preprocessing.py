#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Preprocessing pipeline for incoming data."""
import logging
import warnings
from typing import Union

import pandas as pd

from nannyml.exceptions import InvalidArgumentsException, InvalidReferenceDataException, MissingMetadataException
from nannyml.metadata import BinaryClassificationMetadata, MulticlassClassificationMetadata
from nannyml.metadata.base import ModelMetadata

logger = logging.getLogger(__name__)

PREDICTED_PROBABILITIES_UNIQUE_VALUES_THRESHOLD = 2


def preprocess(data: pd.DataFrame, metadata: ModelMetadata, reference: bool = False) -> pd.DataFrame:
    """Analyse and prepare incoming data for further use downstream.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing model inputs, scores, targets and other metadata.
    metadata: ModelMetadata
        Optional ModelMetadata instance that might have been manually constructed
        or contains non-default values
    reference: bool
        Boolean indicating whether additional checks for reference data should be executed.

    Returns
    -------
    prepped_data: Optional[DataFrame]
        A copy of the uploaded data with added copies of metadata columns
        Will be ``None`` when the extracted/provided metadata was not complete.

    """
    # Check metadata for completeness.
    metadata_complete, missing_properties = metadata.is_complete()  # type: ignore
    if not metadata_complete:
        raise MissingMetadataException(
            f'metadata is still missing values for {missing_properties}.\n'
            'Please rectify by renaming columns following automated extraction conventions\n'
            'and re-running preprocessing or set metadata properties manually.\n'
            'See https://docs.nannyml.com/metadata-extraction for more information\n'
        )

    if data.empty:
        raise InvalidArgumentsException("provided data cannot be empty.")

    # If complete then add copies of metadata columns
    prepped_data = metadata.enrich(data)

    # TODO refactor this into a proper pattern
    if isinstance(metadata, MulticlassClassificationMetadata):
        ok, missing = metadata.validate_predicted_class_labels_in_class_probability_mapping(data)
        if not ok:
            raise InvalidArgumentsException(
                f"class labels {missing} in the prediction column "
                f"'{metadata.prediction_column_name}' have no corresponding predicted "
                f"class probability column. "
                f"Please review the prediction column values and the metadata "
                f"'predicted_probabilities_column_names' property."
            )

    # Check if predicted probability values don't contain (binary) prediction values instead
    _check_predicted_probabilities_are_probabilities(metadata, data)

    # When dealing with reference data, perform some additional, stricter checks.
    if reference:
        _validate_reference_data(data, metadata)

    return prepped_data


# TODO: move this to calculators or metadata subclasses?
def _validate_reference_data(reference_data: pd.DataFrame, metadata: Union[ModelMetadata]):

    if not isinstance(metadata, BinaryClassificationMetadata):
        return

    if (
        metadata.predicted_probability_column_name
        and reference_data[metadata.predicted_probability_column_name].hasnans
    ):
        raise InvalidReferenceDataException(
            f"predicted probability column '{metadata.predicted_probability_column_name}' contains NaN values."
            "Please ensure any NaN values are removed or replaced."
        )

    if metadata.prediction_column_name and reference_data[metadata.prediction_column_name].hasnans:
        raise InvalidReferenceDataException(
            f"prediction column '{metadata.prediction_column_name}' contains NaN values."
            "Please ensure any NaN values are removed or replaced."
        )

    if metadata.target_column_name and reference_data[metadata.target_column_name].hasnans:
        raise InvalidReferenceDataException(
            f"target column '{metadata.target_column_name}' contains NaN values."
            "Please ensure any NaN values are removed or replaced."
        )


# TODO: move this to calculators or metadata subclasses?
def _check_predicted_probabilities_are_probabilities(metadata: ModelMetadata, data: pd.DataFrame):
    if not isinstance(metadata, BinaryClassificationMetadata):
        return

    if metadata.predicted_probability_column_name is None:
        return

    predicted_probabilities = data[metadata.predicted_probability_column_name]

    values_within_bounds = predicted_probabilities.between(0, 1).all()
    if not values_within_bounds:
        warnings.warn(
            message=f"the predicted probabilities column '{metadata.predicted_probability_column_name}'"
            f" contains values outside of the accepted [0, 1] interval. "
            f"Please ensure you are not providing predictions instead."
        )

    enough_unique_values = predicted_probabilities.nunique() > PREDICTED_PROBABILITIES_UNIQUE_VALUES_THRESHOLD
    if not enough_unique_values:
        warnings.warn(
            message=f"the predicted probabilities column '{metadata.predicted_probability_column_name}'"
            f" contains fewer than {PREDICTED_PROBABILITIES_UNIQUE_VALUES_THRESHOLD} unique values. "
            f"Please ensure you are not providing predictions instead."
        )
