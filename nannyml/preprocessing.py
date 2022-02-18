#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Preprocessing pipeline for incoming data."""
import logging

import pandas as pd

from nannyml.exceptions import MissingMetadataException
from nannyml.metadata import ModelMetadata

logger = logging.getLogger(__name__)


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
    metadata: ModelMetadata
        Metadata gathered by automated extraction.
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

    return prepped_data
