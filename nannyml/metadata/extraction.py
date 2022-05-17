#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing functionality to extract metadata for a given model type from a data sample."""

from typing import List

import pandas as pd

from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata.base import ModelType
from nannyml.metadata.binary_classification import BinaryClassificationMetadata
from nannyml.metadata.multiclass_classification import MulticlassClassificationMetadata
from nannyml.metadata.regression import RegressionMetadata


def extract_metadata(data: pd.DataFrame, model_type: str, model_name: str = None, exclude_columns: List[str] = None):
    """Tries to extract model metadata from a given data set.

    Manually constructing model metadata can be cumbersome, especially if you have hundreds of features.
    NannyML includes this helper function that tries to do the boring stuff for you using some simple rules.

    By default, all columns in the given dataset are considered to be either model features or metadata. Use the
    ``exclude_columns`` parameter to prevent columns from being interpreted as metadata or features.

    Parameters
    ----------
    data : DataFrame
        The dataset containing model inputs and outputs, enriched with the required metadata.
    model_type: str
        The kind of model to extract metadata for.
        Should be one of "classification_binary" or "classification_multiclass".
    model_name : str
        A human-readable name for the model.
    exclude_columns: List[str], default=None
        A list of column names that are to be skipped during metadata extraction, preventing them from being interpreted
        as either model metadata or model features.

    Returns
    -------
    metadata: ModelMetadata
        A fully initialized ModelMetadata instance.


    Examples
    --------
    >>> from nannyml.datasets import load_synthetic_binary_classification_dataset
    >>> from nannyml.metadata import extract_metadata
    >>> ref_df, _, _ = load_synthetic_binary_classification_dataset()
    >>> metadata = extract_metadata(ref_df, model_name='work_from_home')
    >>> metadata.is_complete()
    (False, ['target_column_name'])
    >>> metadata.to_dict()
    {'identifier_column_name': 'identifier',
     'timestamp_column_name': 'timestamp',
     'partition_column_name': 'partition',
     'target_column_name': None,
     'prediction_column_name': None,
     'predicted_probability_column_name': 'y_pred_proba',
     'features': [{'label': 'distance_from_office',
       'column_name': 'distance_from_office',
       'type': 'continuous',
       'description': 'extracted feature: distance_from_office'},
      {'label': 'salary_range',
       'column_name': 'salary_range',
       'type': 'categorical',
       'description': 'extracted feature: salary_range'},
      {'label': 'gas_price_per_litre',
       'column_name': 'gas_price_per_litre',
       'type': 'continuous',
       'description': 'extracted feature: gas_price_per_litre'},
      {'label': 'public_transportation_cost',
       'column_name': 'public_transportation_cost',
       'type': 'continuous',
       'description': 'extracted feature: public_transportation_cost'},
      {'label': 'wfh_prev_workday',
       'column_name': 'wfh_prev_workday',
       'type': 'categorical',
       'description': 'extracted feature: wfh_prev_workday'},
      {'label': 'workday',
       'column_name': 'workday',
       'type': 'categorical',
       'description': 'extracted feature: workday'},
      {'label': 'tenure',
       'column_name': 'tenure',
       'type': 'continuous',
       'description': 'extracted feature: tenure'},
      {'label': 'work_home_actual',
       'column_name': 'work_home_actual',
       'type': 'categorical',
       'description': 'extracted feature: work_home_actual'}]}


    Notes
    -----
    NannyML can only make educated guesses as to what kind of data lives where. When NannyML feels to unsure
    about a guess, it will not use it.
    Be sure to always review the results of this method for their correctness and completeness.
    Adjust and complete as you see fit.
    """
    metadata = ModelMetadataFactory.create(model_type=ModelType.parse(model_type), model_name=model_name).extract(
        data, exclude_columns=exclude_columns
    )

    return metadata


class ModelMetadataFactory:
    """A factory class that aids in the construction of :class:`~nannyml.metadata.base.ModelMetadata` subclasses."""

    mapping = {
        ModelType.CLASSIFICATION_BINARY: BinaryClassificationMetadata,
        ModelType.CLASSIFICATION_MULTICLASS: MulticlassClassificationMetadata,
        ModelType.REGRESSION: RegressionMetadata,
    }

    @classmethod
    def create(cls, model_type: ModelType, **kwargs):
        """Creates a new :class:`~nannyml.metadata.base.ModelMetadata` subclass instance for a given model type.

        Parameters
        ----------
        model_type : ModelType
            The type of model NannyML should try to extract metadata for. This type will determine the properties
            NannyML will look for in the data sample.
        kwargs :
            Any optional keyword arguments to be passed along to the :class:`~nannyml.metadata.base.ModelMetadata`
            constructor.

        Returns
        -------
        metadata: ModelMetadata
            A new :class:`~nannyml.metadata.base.ModelMetadata` subclass instance.

        """
        if model_type not in cls.mapping:
            raise InvalidArgumentsException(f"could not create metadata for model type '{model_type}'")

        metadata = cls.mapping[model_type]
        return metadata(**kwargs)
