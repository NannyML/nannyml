#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing the metadata for binary classification models."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from nannyml.metadata.base import (
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
    ModelMetadata,
    ModelType,
    _check_for_nan,
    _extract_features,
)
from nannyml.metadata.feature import FeatureType

NML_METADATA_PREDICTION_COLUMN_NAME = 'nml_meta_prediction'
NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME = 'nml_meta_predicted_proba'


class BinaryClassificationMetadata(ModelMetadata):
    """Contains the metadata for multiclass classification models.

    An extension of :class:`nannyml.metadata.base.ModelMetadata` that contains properties relating to binary
    classification models:
    - ``prediction_column_name``: the name of the column containing the predicted label
    - ``predicted_probability_column_name``: the name of the column containing the predicted score or probability.
    """

    def __init__(
        self, prediction_column_name: str = None, predicted_probability_column_name: str = None, *args, **kwargs
    ):
        """Creates a new instance of BinaryClassificationMetadata.

        Parameters
        ----------
        prediction_column_name : str
            The name of the column that contains the models' predictions. Optional, defaults to ``None``.
        predicted_probability_column_name: str
            The name of the column containing the predicted score or probability. Optional, defaults to ``None``.

        Warnings
        --------
        Whilst at least one of these two properties must be given for the metadata to be deemed complete, most
        performance-related calculators and estimators will require both predicted labels and predicted probabilities
        to be given.
        """
        super().__init__(ModelType.CLASSIFICATION_BINARY, *args, **kwargs)
        self._prediction_column_name = prediction_column_name
        self._predicted_probability_column_name = predicted_probability_column_name

    @property
    def prediction_column_name(self):  # noqa: D102
        return self._prediction_column_name

    @prediction_column_name.setter
    def prediction_column_name(self, column_name: str):  # noqa: D102
        self._prediction_column_name = column_name
        self._remove_from_features(column_name)

    @property
    def predicted_probability_column_name(self):  # noqa: D102
        return self._predicted_probability_column_name

    @predicted_probability_column_name.setter
    def predicted_probability_column_name(self, column_name: str):  # noqa: D102
        self._predicted_probability_column_name = column_name
        self._remove_from_features(column_name)

    @property
    def metadata_columns(self):
        """Returns all metadata columns that are added to the data by the ``enrich`` method."""
        return [
            NML_METADATA_PARTITION_COLUMN_NAME,
            NML_METADATA_PREDICTION_COLUMN_NAME,
            NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
            NML_METADATA_TARGET_COLUMN_NAME,
            NML_METADATA_TIMESTAMP_COLUMN_NAME,
        ]

    def is_complete(self) -> Tuple[bool, List[str]]:
        """Flags if the ModelMetadata is considered complete or still missing values.

        Returns
        -------
        complete: bool
            True when all required fields are present, False otherwise
        missing: List[str]
            A list of all missing properties. Empty when metadata is complete.

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata('work_from_home', target_column_name='work_home_actual')
        >>> metadata.features = [
        >>>     Feature('cat1', 'cat1', FeatureType.CATEGORICAL), Feature('cat2', 'cat2', FeatureType.CATEGORICAL),
        >>>     Feature('cont1', 'cont1', FeatureType.CONTINUOUS), Feature('cont2', 'cont2', FeatureType.UNKNOWN)]
        >>> # missing either predicted labels or predicted probabilities, 'cont2' has an unknown feature type
        >>> metadata.is_complete()
        (False, ['predicted_probability_column_name', 'prediction_column_name'])
        >>> metadata.predicted_probability_column_name = 'y_pred_proba'  # fix the missing value
        >>> metadata.feature(feature='cont2').feature_type = FeatureType.CONTINUOUS
        >>> metadata.is_complete()
        (True, [])
        """
        ok, missing = super().is_complete()

        # Either predicted probabilities or predicted labels should be specified
        if self.prediction_column_name is None and self.predicted_probability_column_name is None:
            ok = False
            missing.append('predicted_probability_column_name')
            missing.append('prediction_column_name')

        return ok, missing

    def extract(self, data: pd.DataFrame, model_name: str = None, exclude_columns: List[str] = None):
        """Tries to extract model metadata from a given data set.

        Manually constructing model metadata can be cumbersome, especially if you have hundreds of features.
        NannyML includes this helper function that tries to do the boring stuff for you using some simple rules.

        By default, all columns in the given dataset are considered to be either model features or metadata. Use the
        ``exclude_columns`` parameter to prevent columns from being interpreted as metadata or features.

        Parameters
        ----------
        data : DataFrame
            The dataset containing model inputs and outputs, enriched with the required metadata.
        model_name : str
            A human-readable name for the model.
        exclude_columns: List[str], default=None
            A list of column names that are to be skipped during metadata extraction,
            preventing them from being interpreted as either model metadata or model features.

        Returns
        -------
        metadata: BinaryClassificationMetadata
            A fully initialized BinaryClassificationMetadata instance.

        Notes
        -----
        This method is most often not used directly, but by calling the
        :func:`nannyml.metadata.extraction.extract_metadata` function that will delegate to this method.
        """
        if super().extract(data, model_name, exclude_columns) is None:
            return None

        predictions = _guess_predictions(data)
        _check_for_nan(data, predictions)
        self.prediction_column_name = None if len(predictions) == 0 else predictions[0]  # type: ignore

        predicted_probabilities = _guess_predicted_probabilities(data)
        _check_for_nan(data, predicted_probabilities)
        self.predicted_probability_column_name = (
            None if len(predicted_probabilities) == 0 else predicted_probabilities[0]
        )

        not_feature_cols = []
        if exclude_columns:
            not_feature_cols = exclude_columns

        if self.prediction_column_name:
            not_feature_cols += [self.prediction_column_name]

        if self.predicted_probability_column_name:
            not_feature_cols += [self.predicted_probability_column_name]

        self.features = _extract_features(data, exclude_columns=not_feature_cols)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Represents a MulticlassClassificationMetadata instance as a dictionary."""
        res = super().to_dict()
        res['prediction_column_name'] = self.prediction_column_name
        res['predicted_probability_column_name'] = self.predicted_probability_column_name
        return res

    def to_df(self) -> pd.DataFrame:
        """Represents a MulticlassClassificationMetadata instance as a DataFrame."""
        df_base = super().to_df()
        df_binary = pd.DataFrame(
            [
                {
                    'label': 'prediction_column_name',
                    'column_name': self.prediction_column_name,
                    'type': FeatureType.CONTINUOUS.value,
                    'description': 'predicted label',
                },
                {
                    'label': 'predicted_probability_column_name',
                    'column_name': self.predicted_probability_column_name,
                    'type': FeatureType.CONTINUOUS.value,
                    'description': 'predicted score/probability',
                },
            ]
        )
        return df_base.append(df_binary, ignore_index=True).reset_index(drop=True)

    def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates copies of all metadata columns with fixed names.

        Parameters
        ----------
        data: DataFrame
            The data to enrich

        Returns
        -------
        enriched_data: DataFrame
            A DataFrame that has all metadata present in NannyML-specific columns.
        """
        df = super().enrich(data)
        if self.prediction_column_name in data.columns:
            df[NML_METADATA_PREDICTION_COLUMN_NAME] = data[self.prediction_column_name]
        else:
            df[NML_METADATA_PREDICTION_COLUMN_NAME] = np.NaN
        if self.predicted_probability_column_name in data.columns:
            df[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME] = data[self.predicted_probability_column_name]
        else:
            df[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME] = np.NaN
        return df


def _guess_predictions(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['p', 'pred', 'prediction', 'out', 'output', 'y_pred']

    return [col for col in data.columns if _guess_if_prediction(data[col])]


def _guess_predicted_probabilities(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['y_pred_proba']

    return [col for col in data.columns if _guess_if_prediction(data[col])]
