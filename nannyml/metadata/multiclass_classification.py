#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing the metadata for multiclass classification models."""

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

PREDICTED_PROBABILITIES_PATTERN = 'y_pred_proba_'

NML_METADATA_PREDICTION_COLUMN_NAME = 'nml_meta_prediction'
NML_METADATA_PREDICTED_CLASS_PROBABILITY_COLUMN_NAME = 'nml_meta_predicted_proba'


class MulticlassClassificationMetadata(ModelMetadata):
    """Contains the metadata for multiclass classification models.

    An extension of :class:`nannyml.metadata.base.ModelMetadata` that contains properties relating to multiclass
    classification models.

    The main differentiator with the :class:`~nannyml.metadata.binary_classification.BinaryClassificationMetadata` class
    is that the predicted probabilities are represented by multiple columns, one for each result class.
    It is to be provided explicitly as a class-to-column-name mapping (a dictionary mapping a class string to a column
    name containing predicted probabilities for that class) or will be extracted automatically by NannyML.
    """

    def __init__(
        self,
        prediction_column_name: str = None,
        predicted_probabilities_column_names: Dict[Any, str] = None,
        *args,
        **kwargs,
    ):
        """Creates a new instance of MulticlassClassificationMetadata.

        Parameters
        ----------
        prediction_column_name : string
            The name of the column that contains the models' predictions. Optional, defaults to ``None``.
        predicted_probabilities_column_names: Dict[str, str], default=None
            A dictionary mapping a model result class to the name of the column in the data that contains
            the predicted probabilities for that class.
        """
        super().__init__(ModelType.CLASSIFICATION_MULTICLASS, *args, **kwargs)
        self._prediction_column_name = prediction_column_name
        self._predicted_probabilities_column_names = predicted_probabilities_column_names

    @property
    def prediction_column_name(self):  # noqa: D102
        return self._prediction_column_name

    @prediction_column_name.setter
    def prediction_column_name(self, column_name: str):  # noqa: D102
        self._prediction_column_name = column_name
        self._remove_from_features(column_name)

    @property
    def predicted_probabilities_column_names(self):  # noqa: D102
        return self._predicted_probabilities_column_names

    @predicted_probabilities_column_names.setter
    def predicted_probabilities_column_names(self, class_to_column_mapping: Dict[Any, str]):  # noqa: D102
        self._predicted_probabilities_column_names = class_to_column_mapping
        if class_to_column_mapping is None or len(class_to_column_mapping) == 0:
            return
        for _, column_name in class_to_column_mapping.items():
            self._remove_from_features(column_name)

    @property
    def metadata_columns(self):
        """Returns all metadata columns that are added to the data by the ``enrich`` method."""
        return [
            NML_METADATA_PARTITION_COLUMN_NAME,
            NML_METADATA_PREDICTION_COLUMN_NAME,
            NML_METADATA_TARGET_COLUMN_NAME,
            NML_METADATA_TIMESTAMP_COLUMN_NAME,
        ] + list(self.predicted_class_probability_metadata_columns().values())

    def predicted_class_probability_metadata_columns(self) -> Dict[Any, str]:
        """Returns the names of the class probability columns added to the data by the ``enrich`` method."""
        return {
            clazz: f'{NML_METADATA_PREDICTED_CLASS_PROBABILITY_COLUMN_NAME}_{clazz}'
            for clazz in self.predicted_probabilities_column_names.keys()
        }

    def class_labels(self) -> List:
        """Returns a sorted list of class labels based on the class probability mapping."""
        return [class_label for class_label in sorted(list(self.predicted_probabilities_column_names.keys()))]

    def to_dict(self) -> Dict[str, Any]:
        """Represents a MulticlassClassificationMetadata instance as a dictionary."""
        res = super().to_dict()
        res['prediction_column_name'] = self.prediction_column_name
        res['predicted_probabilities_column_names'] = self.predicted_probabilities_column_names
        return res

    def to_df(self) -> pd.DataFrame:
        """Represents a MulticlassClassificationMetadata instance as a DataFrame.

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata(model_type='classification_multiclass',
        target_column_name='work_home_actual')
        >>> metadata.features = [Feature(column_name='dist_from_office', label='office_distance',
        description='Distance from home to the office', feature_type=FeatureType.CONTINUOUS),
        >>> Feature(column_name='salary_range', label='salary_range', feature_type=FeatureType.CATEGORICAL)]
        >>> metadata.to_df()
        """
        df_base = super().to_df()

        multiclass_entries = [
            {
                'label': 'prediction_column_name',
                'column_name': self.prediction_column_name,
                'type': FeatureType.CONTINUOUS.value,
                'description': 'predicted label',
            }
        ]
        if self.predicted_probabilities_column_names:
            multiclass_entries += [
                {
                    'label': f'predicted_probability_column_name_{clazz}',
                    'column_name': column_name,
                    'type': FeatureType.CONTINUOUS.value,
                    'description': f"predicted score/probability for class '{clazz}'",
                }
                for clazz, column_name in self.predicted_probabilities_column_names.items()
            ]

        df_multiclass = pd.DataFrame(multiclass_entries)
        return df_base.append(df_multiclass, ignore_index=True).reset_index(drop=True)

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

        for clazz, column_name in self.predicted_probabilities_column_names.items():
            if column_name in data.columns:
                df[f'{NML_METADATA_PREDICTED_CLASS_PROBABILITY_COLUMN_NAME}_{clazz}'] = data[column_name]
            else:
                df[f'{NML_METADATA_PREDICTED_CLASS_PROBABILITY_COLUMN_NAME}_{clazz}'] = np.NaN

        return df

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
        >>> from nannyml.metadata import MulticlassClassificationMetadata, Feature, FeatureType
        >>> metadata = MulticlassClassificationMetadata(target_column_name='work_home_actual')
        >>> metadata.features = [
        >>>     Feature('cat1', 'cat1', FeatureType.CATEGORICAL), Feature('cat2', 'cat2', FeatureType.CATEGORICAL),
        >>>     Feature('cont1', 'cont1', FeatureType.CONTINUOUS), Feature('cont2', 'cont2', FeatureType.UNKNOWN)]
        >>> # missing either predicted labels or predicted probabilities, 'cont2' has an unknown feature type
        >>> metadata.is_complete()
        (False, ['predicted_probabilities_column_names', 'prediction_column_name'])
        >>> metadata.predicted_probabilities_column_names = {'A': 'y_pred_proba_A', 'B': 'y_pred_proba_B'}
        >>> metadata.feature(feature='cont2').feature_type = FeatureType.CONTINUOUS
        >>> metadata.is_complete()
        (True, [])
        """
        ok, missing = super().is_complete()

        # Either predicted probabilities or predicted labels should be specified
        if self.prediction_column_name is None and (
            self.predicted_probabilities_column_names is None or len(self.predicted_probabilities_column_names) == 0
        ):
            ok = False
            missing.append('predicted_probabilities_column_names')
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
        metadata: MulticlassClassificationMetadata
            A fully initialized MultiClassClassificationMetadata instance.

        Notes
        -----
        This method is most often not used directly, but by calling the
        :func:`nannyml.metadata.extraction.extract_metadata` function that will delegate to this method.
        """
        md = super().extract(data, model_name, exclude_columns)

        predictions = _guess_predictions(data)
        _check_for_nan(data, predictions)
        self.prediction_column_name = None if len(predictions) == 0 else predictions[0]  # type: ignore

        predicted_probabilities = _guess_predicted_probabilities(data)
        class_to_column_mapping = _extract_class_to_column_mapping(predicted_probabilities)
        _check_for_nan(data, list(class_to_column_mapping.values()))
        self.predicted_probabilities_column_names = class_to_column_mapping

        # exclude columns from feature candidates
        not_feature_cols = []
        if exclude_columns:
            not_feature_cols = exclude_columns

        if self.prediction_column_name:
            not_feature_cols += [self.prediction_column_name]

        if self.predicted_probabilities_column_names:
            not_feature_cols += [
                predicted_class_probability_column
                for predicted_class_probability_column in self.predicted_probabilities_column_names.values()
            ]

        self.features = _extract_features(data, exclude_columns=not_feature_cols)

        return md

    def validate_predicted_class_labels_in_class_probability_mapping(self, data: pd.DataFrame) -> Tuple[bool, List]:
        """Checks if all predicted class labels have a corresponding predicted class probability column.

        Parameters
        ----------
        data: pd.DataFrame
            A pd.DataFrame that contains both the prediction column and the predicted class probability columns.

        Returns
        -------
        ok: bool
            Boolean indicating validity. ``True`` when no class probability columns are missing, ``False`` otherwise.
        missing: List
            A list of predicted classes for which a corresponding probability column is missing.
        """
        predicted_class_labels = list(data[self.prediction_column_name].unique())
        missing = [
            predicted_class_label
            for predicted_class_label in predicted_class_labels
            if predicted_class_label not in self.class_labels()
        ]

        return len(missing) == 0, missing


def _extract_class_to_column_mapping(column_names: List[str]) -> Dict[Any, str]:
    import re

    mapping = {}
    for column_name in column_names:
        match = re.search(f'(?<={PREDICTED_PROBABILITIES_PATTERN})\\w+', column_name)
        if match:
            class_name: Any
            try:
                class_name = int(match.group(0))
            except ValueError:
                class_name = match.group(0)
            mapping[class_name] = column_name

    return mapping


def _guess_predictions(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['p', 'pred', 'prediction', 'out', 'output', 'y_pred']

    return [col for col in data.columns if _guess_if_prediction(data[col])]


def _guess_predicted_probabilities(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return str(col.name).startswith(PREDICTED_PROBABILITIES_PATTERN)

    return [col for col in data.columns if _guess_if_prediction(data[col])]
