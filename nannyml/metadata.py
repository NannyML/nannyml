#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""NannyML module providing classes and utilities for dealing with model metadata."""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nannyml.exceptions import MissingMetadataException

NML_METADATA_PARTITION_COLUMN_NAME = 'nml_meta_partition'
NML_METADATA_PREDICTION_COLUMN_NAME = 'nml_meta_prediction'
NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME = 'nml_meta_predicted_proba'
NML_METADATA_TARGET_COLUMN_NAME = 'nml_meta_target'
NML_METADATA_IDENTIFIER_COLUMN_NAME = 'nml_meta_identifier'
NML_METADATA_TIMESTAMP_COLUMN_NAME = 'nml_meta_timestamp'

NML_METADATA_REFERENCE_PARTITION_NAME = 'reference'
NML_METADATA_ANALYSIS_PARTITION_NAME = 'analysis'

NML_METADATA_COLUMNS = [
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_PREDICTION_COLUMN_NAME,
    NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    NML_METADATA_IDENTIFIER_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
]

logger = logging.getLogger(__name__)


# TODO wording
class FeatureType(str, Enum):
    """An enum indicating what kind of variable a given feature represents.

    The FeatureType enum is a property of a Feature. NannyML uses this information to select the best drift detection
    algorithms for each individual feature.

    We consider the following feature types:

    CONTINUOUS: numeric variables that have an infinite number of values between any two values.
    CATEGORICAL: has two or more categories, but there is no intrinsic ordering to the categories.
    ORDINAL: similar to a categorical variable, but there is a clear ordering of the categories.
    UNKNOWN: indicates NannyML couldn't detect the feature type with a high enough degree of certainty.
    """

    CONTINUOUS = 'continuous'
    CATEGORICAL = 'categorical'
    ORDINAL = 'ordinal'
    UNKNOWN = 'unknown'


class Feature:
    """Representation of a model feature.

    NannyML requires both model inputs and outputs to perform drift calculation and performance metrics.
    It needs to understand what features a model is made of and what kind of data they might contain.
    The Feature class allows you to provide this information.
    """

    def __init__(self, column_name: str, label: str, feature_type: FeatureType, description: str = None):
        """Creates a new Feature instance.

        The ModelMetadata class contains a list of Features that describe the values that serve as model input.

        Parameters
        ----------
        column_name : str
            The name of the column where the feature is found in the (to be provided) model input/output data.
        label : str
            A (human-friendly) label for the feature.
        feature_type : FeatureType
            The kind of values the data for this feature are.
        description : str
            Some additional information to display within results and visualizations.

        Returns
        -------
        feature: Feature

        """
        self.column_name = column_name
        self.label = label
        self.description = description
        self.feature_type = feature_type

    def to_dict(self) -> Dict[str, Any]:
        """Converts the feature into a Dictionary representation."""
        return {
            'label': self.label,
            'column_name': self.column_name,
            'type': self.feature_type.value,
            'description': self.description,
        }

    def __repr__(self):
        """String representation of a single Feature."""
        return f'Feature({self.to_dict()})'

    def __str__(self):  # pragma: no cover
        """String representation of a single Feature."""
        return f'Feature({self.to_dict()})'

    def print(self):
        """String representation of a single Feature."""
        strs = [
            f"Feature: {self.label}",
            '',
            f"{'Column name':25} {self.column_name:25}",
            f"{'Description':25} {self.description:25}",
            f"{'Type':25} {self.feature_type:25}",
            '',
        ]
        return str.join('\n', strs)


# TODO wording
class ModelMetadata:
    """The ModelMetadata class contains all the information nannyML requires.

    To understand the model inputs and outputs you wish it to process, nannyML needs to understand what your model
    (and hence also your model inputs/outputs) look like.
    The ModelMetadata class combines all the information about your model it might need. We call this the model
    metadata, since it does not concern the actual model (e.g .weights or coefficients) but generic information about
    your model.

    These properties are:
    - `model_name` : a human-readable name for the model
    - `model_purpose` : an optional description of the use for your model
    - `model_problem` : the kind of problem your model is trying to solve.
    We currently only support `binary_classification` problems but are planning to support more very soon!
    - `features` : the list of Features for the model
    - `identifier_column_name` : name of the column that contains a value that acts as an identifier for the
    observation, i.e. it is unique over all observations.
    - `prediction_column_name` : name of the column that contains the models' predictions
    - `target_column_name` : name of the column that contains the ground truth / target / actual.
    - `partition_column_name` : name of the column that contains the partition the observation belongs to.
    Allowed partition values are 'reference' and 'analysis'.
    - `timestamp_column_name` : name of the column that contains the timestamp indicating when the observation occurred.

    """

    # TODO wording
    def __init__(
        self,
        model_name: str = None,
        model_problem: str = 'binary_classification',
        features: List[Feature] = None,
        identifier_column_name: str = 'id',
        prediction_column_name: str = None,
        predicted_probability_column_name: str = None,
        target_column_name: str = 'target',
        partition_column_name: str = 'partition',
        timestamp_column_name: str = 'date',
    ):
        """Creates a new ModelMetadata instance.

        Parameters
        ----------
        model_name : string
            A human-readable name for the model. Required.
        model_problem : string
            The kind of problem your model is trying to solve. Optional, defaults to `binary_classification`.
        features : List[Feature]
            The list of Features for the model. Optional, defaults to `None`.
        identifier_column_name : string
            The name of the column that contains a value that acts as an identifier for the
            observation, i.e. it is unique over all observations. Optional, defaults to `id`
        prediction_column_name : string
            The name of the column that contains the models' predictions. Optional, defaults to ``None``.
        predicted_probability_column_name: string
            The name of the column that contains the models' predicted probabilities.
            Optional, defaults to ``None``.
        target_column_name : string
            The name of the column that contains the ground truth / target / actual. Optional, defaults to `target`
        partition_column_name : string
            The name of the column that contains the partition the observation belongs to.
            Allowed partition values are 'reference' and 'analysis'. Optional, defaults to `partition`
        timestamp_column_name : string
            The name of the column that contains the timestamp indicating when the observation occurred.
            Optional, defaults to `date`.

        Returns
        -------
        metadata: ModelMetadata
        """
        self.id: int

        self.name = model_name
        self.model_problem = model_problem

        self._identifier_column_name = identifier_column_name
        self._prediction_column_name = prediction_column_name
        self._predicted_probability_column_name = predicted_probability_column_name
        self._target_column_name = target_column_name
        self._partition_column_name = partition_column_name
        self._timestamp_column_name = timestamp_column_name

        self.features = [] if features is None else features

    @property
    def identifier_column_name(self):  # noqa: D102
        return self._identifier_column_name

    @identifier_column_name.setter
    def identifier_column_name(self, column_name: str):  # noqa: D102
        self._identifier_column_name = column_name
        self.__remove_from_features(column_name)

    @property
    def prediction_column_name(self):  # noqa: D102
        return self._prediction_column_name

    @prediction_column_name.setter
    def prediction_column_name(self, column_name: str):  # noqa: D102
        self._prediction_column_name = column_name
        self.__remove_from_features(column_name)

    @property
    def predicted_probability_column_name(self):  # noqa: D102
        return self._predicted_probability_column_name

    @predicted_probability_column_name.setter
    def predicted_probability_column_name(self, column_name: str):  # noqa: D102
        self._predicted_probability_column_name = column_name
        self.__remove_from_features(column_name)

    @property
    def target_column_name(self):  # noqa: D102
        return self._target_column_name

    @target_column_name.setter
    def target_column_name(self, column_name: str):  # noqa: D102
        self._target_column_name = column_name
        self.__remove_from_features(column_name)

    @property
    def partition_column_name(self):  # noqa: D102
        return self._partition_column_name

    @partition_column_name.setter
    def partition_column_name(self, column_name: str):  # noqa: D102
        self._partition_column_name = column_name
        self.__remove_from_features(column_name)

    @property
    def timestamp_column_name(self):  # noqa: D102
        return self._timestamp_column_name

    @timestamp_column_name.setter
    def timestamp_column_name(self, column_name: str):  # noqa: D102
        self._timestamp_column_name = column_name
        self.__remove_from_features(column_name)

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover
        """Converts a ModelMetadata instance into a Dictionary."""
        return {
            'identifier_column_name': self.identifier_column_name,
            'timestamp_column_name': self.timestamp_column_name,
            'partition_column_name': self.partition_column_name,
            'target_column_name': self.target_column_name,
            'prediction_column_name': self.prediction_column_name,
            'predicted_probability_column_name': self.predicted_probability_column_name,
            'features': repr(self.features),
        }

    def to_df(self) -> pd.DataFrame:
        """Converts a ModelMetadata instance into a read-only DataFrame."""
        return pd.DataFrame(
            [
                {
                    'label': 'identifier_column_name',
                    'column_name': self.identifier_column_name,
                    'type': FeatureType.CONTINUOUS.value,
                    'description': 'identifier',
                },
                {
                    'label': 'timestamp_column_name',
                    'column_name': self.timestamp_column_name,
                    'type': FeatureType.CONTINUOUS.value,
                    'description': 'timestamp',
                },
                {
                    'label': 'partition_column_name',
                    'column_name': self.partition_column_name,
                    'type': FeatureType.CATEGORICAL.value,
                    'description': 'partition',
                },
                {
                    'label': 'target_column_name',
                    'column_name': self.target_column_name,
                    'type': FeatureType.CATEGORICAL.value,
                    'description': 'target',
                },
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
            + [feature.to_dict() for feature in self.features],
            columns=['label', 'column_name', 'type', 'description'],
        )

    def print(self):  # pragma: no cover
        """Returns a string representation of a ModelMetadata instance."""
        UNKNOWN = "~ UNKNOWN ~"
        strs = [
            f"Metadata for model {self.name or UNKNOWN}",
            '',
            '# Warning - unable to identify all essential data',
            f'# Please identify column names for all \'{UNKNOWN}\' values',  # TODO: add link to relevant docs
            '',
            f"{'Model problem':35} {self.model_problem or UNKNOWN:35}",
            '',
            f"{'Identifier column':35} {self.identifier_column_name or UNKNOWN:35}",
            f"{'Timestamp column':35} {self.timestamp_column_name or UNKNOWN:35}",
            f"{'Partition column':35} {self.partition_column_name or UNKNOWN:35}",
            f"{'Prediction column':35} {self.prediction_column_name or UNKNOWN:35}",
            f"{'Predicted probability column':35} {self.predicted_probability_column_name or UNKNOWN:35}",
            f"{'Target column':35} {self.target_column_name or UNKNOWN:35}",
            '',
            'Features',
            '',
            f"{'Name':35} {'Column':35} {'Type':15} {'Description'}",
        ]
        for f in self.features:
            strs.append(f"{f.label:35} {f.column_name:35} {f.feature_type or 'NA':15} {f.description}")
        print(str.join('\n', strs))

    def __repr__(self):
        """Converts the ModelMetadata instance to a string representation."""
        return f'Metadata({self.to_dict()})'

    def __str__(self):  # pragma: no cover
        """Converts the ModelMetadata instance to a string representation."""
        return f'Metadata({self.to_dict()})'

    def feature(self, index: int = None, feature: str = None, column: str = None) -> Optional[Feature]:
        """A function used to access a specific model feature.

        Because a model might contain hundreds of features NannyML provides this utility method to filter through
        them and find the exact feature you need.

        Parameters
        ----------
        index : int
            Retrieve a Feature using its index in the features list.
        feature : str
            Retrieve a feature using its label.
        column : str
            Retrieve a feature using the name of the column it has in the model inputs/outputs.

        Returns
        -------
        feature: Feature
            A single Feature matching the search criteria. Returns `None` if none were found matching the criteria
            or no criteria were provided.

        """
        if feature:
            matches = [f for f in self.features if f.label == feature]
            return matches[0] if len(matches) != 0 else None

        if column:
            matches = [f for f in self.features if f.column_name == column]
            return matches[0] if len(matches) != 0 else None

        if index is not None:
            return self.features[index]

        else:
            return None

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
        data = data.copy()

        data[NML_METADATA_IDENTIFIER_COLUMN_NAME] = data[self.identifier_column_name]
        data[NML_METADATA_TIMESTAMP_COLUMN_NAME] = data[self.timestamp_column_name]
        if self.prediction_column_name in data.columns:
            data[NML_METADATA_PREDICTION_COLUMN_NAME] = data[self.prediction_column_name]
        else:
            data[NML_METADATA_PREDICTION_COLUMN_NAME] = np.NaN
        if self.predicted_probability_column_name in data.columns:
            data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME] = data[self.predicted_probability_column_name]
        else:
            data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME] = np.NaN
        data[NML_METADATA_PARTITION_COLUMN_NAME] = data[self.partition_column_name]
        if self.target_column_name in data.columns:
            data[NML_METADATA_TARGET_COLUMN_NAME] = data[self.target_column_name]
        else:
            data[NML_METADATA_TARGET_COLUMN_NAME] = np.NaN

        return data

    @property
    def categorical_features(self) -> List[Feature]:
        """Retrieves all categorical features.

        Returns
        -------
        features: List[Feature]
            A list of all categorical features
        """
        return [f for f in self.features if f.feature_type == FeatureType.CATEGORICAL]

    @property
    def continuous_features(self) -> List[Feature]:
        """Retrieves all continuous features.

        Returns
        -------
        features: List[Feature]
            A list of all continuous features
        """
        return [f for f in self.features if f.feature_type == FeatureType.CONTINUOUS]

    def is_complete(self) -> Tuple[bool, List[str]]:
        """Flags if the ModelMetadata is considered complete or still missing values.

        Returns
        -------
        complete: bool
            True when all required fields are present, False otherwise
        missing: List[str]
            A list of all missing properties. Empty when metadata is complete.
        """
        props_to_check = [
            'timestamp_column_name',
            'target_column_name',
            'timestamp_column_name',
            'partition_column_name',
        ]
        complete = True
        missing = []

        for attr in props_to_check:
            if self.__getattribute__(attr) is None:
                missing.append(attr)
                complete = False

        if self.prediction_column_name is None and self.predicted_probability_column_name is None:
            complete = False
            missing.append('predicted_probability_column_name')
            missing.append('prediction_column_name')

        features_with_unknown_type = list(filter(lambda f: f.feature_type == FeatureType.UNKNOWN, self.features))
        if len(features_with_unknown_type) > 0:
            complete = False
            missing += [f.column_name for f in features_with_unknown_type]

        return complete, missing

    def __remove_from_features(self, column_name: str):
        current_feature = self.feature(column=column_name)
        if current_feature:
            self.features.remove(current_feature)


def extract_metadata(data: pd.DataFrame, model_name: str = None):
    """Tries to extract model metadata from a given data set.

    Manually constructing model metadata can be cumbersome, especially if you have hundreds of features.
    NannyML includes this helper function that tries to do the boring stuff for you using some simple rules.

    Parameters
    ----------
    data : DataFrame
        The dataset containing model inputs and outputs, enriched with the required metadata.
    model_name : string
            A human-readable name for the model.

    Returns
    -------
    metadata: ModelMetadata
        A fully initialized ModelMetadata instance.

    Notes
    -----
    NannyML can only make educated guesses as to what kind of data lives where. When NannyML feels to unsure
    about a guess, it will not use it.
    Be sure to always review the results of this method for their correctness and completeness.
    Adjust and complete as you see fit.
    """

    def check_for_nan(column_names):
        number_of_nan = data[column_names].isnull().sum().sum()
        if number_of_nan > 0:
            raise MissingMetadataException(f'found {number_of_nan} NaN values in one of these columns: {column_names}')

    if len(data.columns) == 0:
        return None

    metadata = ModelMetadata(model_name=model_name)

    identifiers = _guess_identifiers(data)
    check_for_nan(identifiers)
    metadata.identifier_column_name = None if len(identifiers) == 0 else identifiers[0]  # type: ignore

    predictions = _guess_predictions(data)
    check_for_nan(predictions)
    metadata.prediction_column_name = None if len(predictions) == 0 else predictions[0]  # type: ignore

    predicted_probabilities = _guess_predicted_probabilities(data)
    check_for_nan(predicted_probabilities)
    metadata.predicted_probability_column_name = (
        None if len(predicted_probabilities) == 0 else predicted_probabilities[0]
    )

    targets = _guess_targets(data)
    check_for_nan(targets)
    metadata.target_column_name = None if len(targets) == 0 else targets[0]  # type: ignore

    partitions = _guess_partitions(data)
    check_for_nan(partitions)
    metadata.partition_column_name = None if len(partitions) == 0 else partitions[0]  # type: ignore

    timestamps = _guess_timestamps(data)
    check_for_nan(timestamps)
    metadata.timestamp_column_name = None if len(timestamps) == 0 else timestamps[0]  # type: ignore

    metadata.features = _extract_features(data)

    return metadata


def _guess_identifiers(data: pd.DataFrame) -> List[str]:
    def _guess_if_identifier(col: pd.Series) -> bool:
        return col.name in ['id', 'ident', 'identity', 'identifier', 'uid', 'uuid']

    return [col for col in data.columns if _guess_if_identifier(data[col])]


def _guess_timestamps(data: pd.DataFrame) -> List[str]:
    def _guess_if_timestamp(col: pd.Series) -> bool:
        return col.name in ['date', 'timestamp', 'ts', 'date', 'time']

    return [col for col in data.columns if _guess_if_timestamp(data[col])]


def _guess_predictions(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['p', 'pred', 'prediction', 'out', 'output', 'y_pred']

    return [col for col in data.columns if _guess_if_prediction(data[col])]


def _guess_predicted_probabilities(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['y_pred_proba']

    return [col for col in data.columns if _guess_if_prediction(data[col])]


def _guess_targets(data: pd.DataFrame) -> List[str]:
    def _guess_if_ground_truth(col: pd.Series) -> bool:
        return col.name in ['target', 'ground_truth', 'actual', 'actuals']

    return [col for col in data.columns if _guess_if_ground_truth(data[col])]


def _guess_partitions(data: pd.DataFrame) -> List[str]:
    def _guess_if_partition(col: pd.Series) -> bool:
        return 'partition' in col.name

    return [col for col in data.columns if _guess_if_partition(data[col])]


def _guess_features(data: pd.DataFrame) -> List[str]:
    def _guess_if_feature(col: pd.Series) -> bool:
        return (
            col.name
            not in _guess_identifiers(data)
            + _guess_partitions(data)
            + _guess_predictions(data)
            + _guess_predicted_probabilities(data)
            + _guess_timestamps(data)
            + _guess_targets(data)
        ) and (col.name not in NML_METADATA_COLUMNS)

    return [col for col in data.columns if _guess_if_feature(data[col])]


def _extract_features(data: pd.DataFrame) -> List[Feature]:
    feature_columns = _guess_features(data)
    if len(feature_columns) == 0:
        return []

    feature_types = _predict_feature_types(data[feature_columns])

    return [
        Feature(
            label=col,
            column_name=col,
            description=f'extracted feature: {col}',
            feature_type=feature_types.loc[col, 'predicted_feature_type'],
        )
        for col in feature_columns
    ]


INFERENCE_NUM_ROWS_THRESHOLD = 5
INFERENCE_HIGH_CARDINALITY_THRESHOLD = 0.1
INFERENCE_MEDIUM_CARDINALITY_THRESHOLD = 0.01
INFERENCE_LOW_CARDINALITY_THRESHOLD = 0.0
INFERENCE_INT_NUNIQUE_THRESHOLD = 40


def _predict_feature_types(df: pd.DataFrame):
    def _determine_type(data_type, row_count, unique_count, unique_fraction):
        if row_count < INFERENCE_NUM_ROWS_THRESHOLD:
            return FeatureType.UNKNOWN

        elif data_type == 'float64':
            return FeatureType.CONTINUOUS

        elif data_type == 'int64' and unique_count >= INFERENCE_INT_NUNIQUE_THRESHOLD:
            return FeatureType.CONTINUOUS

        elif data_type == 'object':
            return FeatureType.CATEGORICAL

        elif unique_fraction >= INFERENCE_HIGH_CARDINALITY_THRESHOLD:
            return FeatureType.CONTINUOUS

        elif INFERENCE_LOW_CARDINALITY_THRESHOLD <= unique_fraction <= INFERENCE_MEDIUM_CARDINALITY_THRESHOLD:
            return FeatureType.CATEGORICAL

        else:
            return FeatureType.UNKNOWN

    # nunique: number of unique values
    # count: number of not-None values
    # size: number of values (including None)
    stats = df.agg(['nunique', 'count']).T
    stats['column_data_type'] = df.dtypes

    stats['unique_count_fraction'] = stats['nunique'] / stats['count']
    stats['predicted_feature_type'] = stats.apply(
        lambda r: _determine_type(
            data_type=r['column_data_type'],
            row_count=r['count'],
            unique_count=r['nunique'],
            unique_fraction=r['unique_count_fraction'],
        ),
        axis=1,
    )

    # Just for serialization purposes
    stats['column_data_type'] = str(stats['column_data_type'])

    return stats
