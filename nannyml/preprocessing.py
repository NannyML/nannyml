#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Preprocessing pipeline for incoming data."""
import logging
from typing import Optional, Tuple

import pandas as pd

from nannyml.calibration import NML_CALIBRATED_SCORE_COLUMN_NAME, calibrated_scores
from nannyml.metadata import ModelMetadata, extract_metadata

logger = logging.getLogger(__name__)


def preprocess(
    data: pd.DataFrame, model_name: str, model_metadata: ModelMetadata = None
) -> Tuple[ModelMetadata, Optional[pd.DataFrame]]:
    """Analyse and prepare incoming data for further use downstream.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing model inputs, scores, targets and other metadata.
    model_name: str
        A name human-readable name to identify your model by.
    model_metadata: ModelMetadata
        Optional ModelMetadata instance that might have been manually constructed
        or contains non-default values

    Returns
    -------
    metadata: ModelMetadata
        Metadata gathered by automated extraction.
    prepped_data: Optional[DataFrame]
        A copy of the uploaded data with added copies of metadata columns
        and other calculated values such as calibrated model scores.
        Will be ``None`` when the extracted/provided metadata was not complete.

    """
    # Extract metadata
    if model_metadata is None:
        model_metadata = extract_metadata(data=data, model_name=model_name)

    # Check metadata for completeness.
    metadata_complete, missing_properties = model_metadata.is_complete()  # type: ignore
    if not metadata_complete:
        logger.error(
            f'metadata is still missing values for {missing_properties}.\n'
            'Please rectify by renaming columns following automated extraction conventions\n'
            'and re-running preprocessing or set metadata properties manually.\n'
            'See https://docs.nannyml.com/metadata-extraction for more information\n'
        )
        return model_metadata, None

    # If complete then add copies of metadata columns
    prepped_data = model_metadata.enrich(data)

    # Calibrate model score when required
    prepped_data[NML_CALIBRATED_SCORE_COLUMN_NAME] = calibrated_scores(
        y_true=prepped_data[model_metadata.ground_truth_column_name],
        y_pred_proba=prepped_data[model_metadata.prediction_column_name],
    )

    return model_metadata, prepped_data
