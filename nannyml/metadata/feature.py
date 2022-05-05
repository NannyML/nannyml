#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""A module containing definitions and functionality concerning model features."""

# TODO wording
from enum import Enum
from typing import Any, Dict


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

        Examples
        --------
        >>> from nannyml.metadata.feature import Feature, FeatureType
        >>> feature = Feature(column_name='dist_from_office', label='office_distance',
        description='Distance from home to the office', feature_type=FeatureType.CONTINUOUS)
        >>> feature
        Feature({'label': 'office_distance', 'column_name': 'dist_from_office', 'type': 'continuous',
        'description': 'Distance from home to the office'})
        """
        self.column_name = column_name
        self.label = label
        self.description = description
        self.feature_type = feature_type

    def to_dict(self) -> Dict[str, Any]:
        """Converts the feature into a Dictionary representation.

        Examples
        --------
        >>> from nannyml.metadata.feature import Feature, FeatureType
        >>> feature = Feature(column_name='dist_from_office', label='office_distance',
        description='Distance from home to the office', feature_type=FeatureType.CONTINUOUS)
        >>> feature.to_dict()
        {'label': 'office_distance',
         'column_name': 'dist_from_office',
         'type': 'continuous',
         'description': 'Distance from home to the office'}

        """
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
