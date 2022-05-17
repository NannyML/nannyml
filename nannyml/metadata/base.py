#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""NannyML module providing classes and utilities for dealing with model metadata."""
import abc
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata.feature import Feature, FeatureType

NML_METADATA_PARTITION_COLUMN_NAME = 'nml_meta_partition'
NML_METADATA_TARGET_COLUMN_NAME = 'nml_meta_target'
NML_METADATA_TIMESTAMP_COLUMN_NAME = 'nml_meta_timestamp'

NML_METADATA_REFERENCE_PARTITION_NAME = 'reference'
NML_METADATA_ANALYSIS_PARTITION_NAME = 'analysis'

NML_METADATA_COLUMNS = [
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    NML_METADATA_TIMESTAMP_COLUMN_NAME,
]

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """A listing of all possible model types.

    The model type will determine which specific metadata properties are required by NannyML.
    Each :class:`~nannyml.metadata.metadata.ModelMetadata` subclass will be associated with a specific ModelType.

    """

    CLASSIFICATION_BINARY = 'classification_binary'
    CLASSIFICATION_MULTICLASS = 'classification_multiclass'
    REGRESSION = 'regression'

    @staticmethod
    def parse(model_type_str: str):
        """Returns a ModelType instance from a string representation."""
        if model_type_str == 'classification_binary':
            return ModelType.CLASSIFICATION_BINARY
        elif model_type_str == 'classification_multiclass':
            return ModelType.CLASSIFICATION_MULTICLASS
        elif model_type_str == 'regression':
            return ModelType.REGRESSION
        else:
            raise InvalidArgumentsException(f"unknown model type '{model_type_str}'")


# TODO wording
class ModelMetadata(abc.ABC):
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
        model_type: ModelType,
        model_name: str = None,
        features: List[Feature] = None,
        target_column_name: str = 'target',
        partition_column_name: str = 'partition',
        timestamp_column_name: str = 'date',
    ):
        """Creates a new ModelMetadata instance.

        Parameters
        ----------
        model_type : ModelType
            The kind of problem your model is trying to solve. Used to determine which metadata properties should
            be known by NannyML.
        model_name : string, default=None
            A human-readable name for the model.
        features : List[Feature]
            The list of Features for the model. Optional, defaults to `None`.
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


        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata(model_type='classification_binary', target_column_name='work_home_actual')
        >>> metadata.features = [Feature(column_name='dist_from_office', label='office_distance',
        description='Distance from home to the office', feature_type=FeatureType.CONTINUOUS),
        >>> Feature(column_name='salary_range', label='salary_range', feature_type=FeatureType.CATEGORICAL)]
        >>> metadata.to_dict()
        {'timestamp_column_name': 'date',
         'partition_column_name': 'partition',
         'target_column_name': 'work_home_actual',
         'prediction_column_name': None,
         'predicted_probability_column_name': None,
         'features': [{'label': 'office_distance',
           'column_name': 'dist_from_office',
           'type': 'continuous',
           'description': 'Distance from home to the office'},
          {'label': 'salary_range',
           'column_name': 'salary_range',
           'type': 'categorical',
           'description': None}]}

        """
        self.model_type = model_type
        self.name = model_name

        self._target_column_name = target_column_name
        self._partition_column_name = partition_column_name
        self._timestamp_column_name = timestamp_column_name

        self.features = [] if features is None else features

    @property
    def target_column_name(self):  # noqa: D102
        return self._target_column_name

    @target_column_name.setter
    def target_column_name(self, column_name: str):  # noqa: D102
        self._target_column_name = column_name
        self._remove_from_features(column_name)

    @property
    def partition_column_name(self):  # noqa: D102
        return self._partition_column_name

    @partition_column_name.setter
    def partition_column_name(self, column_name: str):  # noqa: D102
        self._partition_column_name = column_name
        self._remove_from_features(column_name)

    @property
    def timestamp_column_name(self):  # noqa: D102
        return self._timestamp_column_name

    @timestamp_column_name.setter
    def timestamp_column_name(self, column_name: str):  # noqa: D102
        self._timestamp_column_name = column_name
        self._remove_from_features(column_name)

    @property
    @abc.abstractmethod
    def metadata_columns(self):
        """Returns all metadata columns that are added to the data by the ``enrich`` method."""
        return NML_METADATA_COLUMNS

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover
        """Converts a ModelMetadata instance into a Dictionary."""
        return {
            'timestamp_column_name': self.timestamp_column_name,
            'partition_column_name': self.partition_column_name,
            'target_column_name': self.target_column_name,
            'features': [feature.to_dict() for feature in self.features],
        }

    @abc.abstractmethod
    def to_df(self) -> pd.DataFrame:
        """Converts a ModelMetadata instance into a read-only DataFrame.

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata(model_type='classification_binary', target_column_name='work_home_actual')
        >>> metadata.features = [Feature(column_name='dist_from_office', label='office_distance',
        description='Distance from home to the office', feature_type=FeatureType.CONTINUOUS),
        >>> Feature(column_name='salary_range', label='salary_range', feature_type=FeatureType.CATEGORICAL)]
        >>> metadata.to_df()
        """
        return pd.DataFrame(
            [
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
            ]
            + [feature.to_dict() for feature in self.features],
            columns=['label', 'column_name', 'type', 'description'],
        )

    def print(self):  # pragma: no cover
        """Returns a string representation of a ModelMetadata instance.

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata(model_type='classification_binary', target_column_name='work_home_actual')
        >>> metadata.features = [Feature(column_name='dist_from_office', label='office_distance',
        >>> description='Distance to the office', feature_type=FeatureType.CONTINUOUS),
        >>> Feature(column_name='salary_range', label='salary_range', feature_type=FeatureType.CATEGORICAL)]
        >>> metadata.print()
        Metadata for model work_from_home
        --
        # Warning - unable to identify all essential data
        # Please identify column names for all '~ UNKNOWN ~' values
        --
        Model problem                       binary_classification
        Timestamp column                    date
        Partition column                    partition
        Prediction column                   ~ UNKNOWN ~
        Predicted probability column        ~ UNKNOWN ~
        Target column                       work_home_actual
        --
        Features
        --
        Name                                Column                              Type            Description
        office_distance                     dist_from_office                    continuous      Distance to the office
        salary_range                        salary_range                        categorical     None
        """
        UNKNOWN = "~ UNKNOWN ~"
        strs = [
            f"Metadata for model {self.name or UNKNOWN}",
            '',
            '# Warning - unable to identify all essential data',
            f'# Please identify column names for all \'{UNKNOWN}\' values',  # TODO: add link to relevant docs
            '',
            f"{'Model type':35} {self.model_type or UNKNOWN:35}",
            '',
            f"{'Timestamp column':35} {self.timestamp_column_name or UNKNOWN:35}",
            f"{'Partition column':35} {self.partition_column_name or UNKNOWN:35}",
            # f"{'Prediction column':35} {self.prediction_column_name or UNKNOWN:35}",
            # f"{'Predicted probability column':35} {self.predicted_probability_column_name or UNKNOWN:35}",
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

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata(model_type='classification_binary', target_column_name='work_home_actual')
        >>> metadata.features = [Feature(column_name='dist_from_office', label='office_distance',
        >>> description='Distance from home to the office', feature_type=FeatureType.CONTINUOUS),
        >>> Feature(column_name='salary_range', label='salary_range', feature_type=FeatureType.CATEGORICAL)]
        >>> metadata.feature(index=1)
        Feature({'label': 'salary_range', 'column_name': 'salary_range', 'type': 'categorical', 'description': None})
        >>> metadata.feature(feature='office_distance')
        Feature({'label': 'office_distance', 'column_name': 'dist_from_office', 'type': 'continuous',
        'description': 'Distance from home to the office'})
        >>> metadata.feature(column='dist_from_office')
        Feature({'label': 'office_distance', 'column_name': 'dist_from_office', 'type': 'continuous',
        'description': 'Distance from home to the office'})

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

    @abc.abstractmethod
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

        data[NML_METADATA_TIMESTAMP_COLUMN_NAME] = data[self.timestamp_column_name]
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

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata(model_type='classification_binary', target_column_name='work_home_actual')
        >>> metadata.features = [
        >>>     Feature('cat1', 'cat1', FeatureType.CATEGORICAL), Feature('cat2', 'cat2', FeatureType.CATEGORICAL),
        >>>     Feature('cont1', 'cont1', FeatureType.CONTINUOUS), Feature('cont2', 'cont2', FeatureType.CONTINUOUS)]
        >>> metadata.categorical_features
        [Feature({'label': 'cat1', 'column_name': 'cat1', 'type': 'categorical', 'description': None}),
        Feature({'label': 'cat2', 'column_name': 'cat2', 'type': 'categorical', 'description': None})]
        """
        return [f for f in self.features if f.feature_type == FeatureType.CATEGORICAL]

    @property
    def continuous_features(self) -> List[Feature]:
        """Retrieves all continuous features.

        Returns
        -------
        features: List[Feature]
            A list of all continuous features

        Examples
        --------
        >>> from nannyml.metadata import ModelMetadata, Feature, FeatureType
        >>> metadata = ModelMetadata('work_from_home', target_column_name='work_home_actual')
        >>> metadata.features = [
        >>>     Feature('cat1', 'cat1', FeatureType.CATEGORICAL), Feature('cat2', 'cat2', FeatureType.CATEGORICAL),
        >>>     Feature('cont1', 'cont1', FeatureType.CONTINUOUS), Feature('cont2', 'cont2', FeatureType.CONTINUOUS)]
        >>> metadata.continuous_features
        [Feature({'label': 'cont1', 'column_name': 'cont1', 'type': 'continuous', 'description': None}),
        Feature({'label': 'cont2', 'column_name': 'cont2', 'type': 'continuous', 'description': None})]
        """
        return [f for f in self.features if f.feature_type == FeatureType.CONTINUOUS]

    @abc.abstractmethod
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

        features_with_unknown_type = list(filter(lambda f: f.feature_type == FeatureType.UNKNOWN, self.features))
        if len(features_with_unknown_type) > 0:
            complete = False
            missing += [f.column_name for f in features_with_unknown_type]

        return complete, missing

    @abc.abstractmethod
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
        metadata: ModelMetadata
            A fully initialized ModelMetadata subclass instance.

        Notes
        -----
        This method is most often not used directly, but by calling the
        :func:`nannyml.metadata.extraction.extract_metadata` function that will delegate to this method.

        This particular abstract method provides common functionality for its subclasses and is always called there
        using a ``super().extract()`` call.
        """
        if len(data.columns) == 0:
            return None

        if exclude_columns is not None and len(exclude_columns) != 0:
            data = data.drop(columns=exclude_columns)

        targets = _guess_targets(data)
        _check_for_nan(data, targets)
        self.target_column_name = None if len(targets) == 0 else targets[0]  # type: ignore

        partitions = _guess_partitions(data)
        _check_for_nan(data, partitions)
        self.partition_column_name = None if len(partitions) == 0 else partitions[0]  # type: ignore

        timestamps = _guess_timestamps(data)
        _check_for_nan(data, timestamps)
        self.timestamp_column_name = None if len(timestamps) == 0 else timestamps[0]  # type: ignore

        # self.features = extract_features(data)

        return self

    def _remove_from_features(self, column_name: str):
        current_feature = self.feature(column=column_name)
        if current_feature:
            self.features.remove(current_feature)


def _guess_timestamps(data: pd.DataFrame) -> List[str]:
    def _guess_if_timestamp(col: pd.Series) -> bool:
        return col.name in ['date', 'timestamp', 'ts', 'date', 'time']

    return [col for col in data.columns if _guess_if_timestamp(data[col])]


def _guess_targets(data: pd.DataFrame) -> List[str]:
    def _guess_if_ground_truth(col: pd.Series) -> bool:
        return col.name in ['target', 'ground_truth', 'actual', 'actuals', 'y_true']

    return [col for col in data.columns if _guess_if_ground_truth(data[col])]


def _guess_partitions(data: pd.DataFrame) -> List[str]:
    def _guess_if_partition(col: pd.Series) -> bool:
        return 'partition' in col.name

    return [col for col in data.columns if _guess_if_partition(data[col])]


def _guess_features(data: pd.DataFrame) -> List[str]:
    def _guess_if_feature(col: pd.Series) -> bool:
        return (col.name not in _guess_partitions(data) + _guess_timestamps(data) + _guess_targets(data)) and (
            col.name not in NML_METADATA_COLUMNS
        )

    return [col for col in data.columns if _guess_if_feature(data[col])]


def _extract_features(data: pd.DataFrame, exclude_columns: List[str] = None) -> List[Feature]:
    if not exclude_columns:
        exclude_columns = []

    feature_columns = [col for col in _guess_features(data) if col not in exclude_columns]
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


def _check_for_nan(data: pd.DataFrame, column_names: List[str]):
    number_of_nan = data[column_names].isnull().sum().sum()
    if number_of_nan > 0:
        raise MissingMetadataException(f'found {number_of_nan} NaN values in one of these columns: {column_names}')
