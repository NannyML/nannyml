#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from nannyml._typing import ProblemType
from nannyml.base import AbstractEstimator, AbstractEstimatorResult, _list_missing, _split_features_by_type
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation import DEFAULT_METRICS
from nannyml.performance_estimation.direct_loss_estimation.metrics import Metric, MetricFactory
from nannyml.performance_estimation.direct_loss_estimation.result import Result


class DLE(AbstractEstimator):
    """The Direct :term:`Loss` Estimator (DLE) estimates the :term:`loss<Loss>` resulting
    from the difference between the prediction and the target before the targets become known.
    The :term:`loss<Loss>` is defined from the regression performance metric
    specified. For all metrics used the :term:`loss<Loss>` function is positive.

    It uses an internal
    `LGBMRegressor <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_
    model per metric to predict the value of the error function
    (the function returning the error for a given prediction) of the monitored model.

    The error results on the reference data become a target for those internal models.

    It is possible to specify a set of hyperparameters to instantiate these internal nanny models with using the
    `hyperparameters` parameter. You can also opt to run hyperparameter tuning using FLAML to determine hyperparameters
    for you. Tuning hyperparameters takes some time and does not guarantee better results,
    hence we don't do it by default.
    """

    def __init__(
        self,
        feature_column_names: List[str],
        y_pred: str,
        y_true: str,
        timestamp_column_name: str,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        metrics: List[str] = None,
        hyperparameters: Dict[str, Any] = None,
        tune_hyperparameters: bool = False,
        hyperparameter_tuning_config: Dict[str, Any] = None,
    ):
        """
        Creates a new Direct Loss Estimator.

        Parameters
        ----------
        feature_column_names : List[str]
            A list of column names indicating which columns contain feature values.
        y_pred : str
            A column name indicating which column contains the model predictions.
        y_true : str
            A column name indicating which column contains the target values.
        timestamp_column_name : str
            A column name indicating which column contains the timestamp of the prediction.
        chunk_size: int, default=None
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int, default=None
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str, default=None
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.
        metrics : List[str], default = ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']
            A list of metrics to calculate. When not provided it will default to include all currently supported
            metrics.
        hyperparameters : Dict[str, Any], default = None
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        tune_hyperparameters : bool, default = False
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config : Dict[str, Any], default = None
            A dictionary that allows you to provide a custom hyperparameter tuning configuration when
            `tune_hyperparameters` has been set to `True`.
            The following dictionary is the default tuning configuration. It can be used as a template to modify::

                {
                    "time_budget": 15,
                    "metric": "mse",
                    "estimator_list": ['lgbm'],
                    "eval_method": "cv",
                    "hpo_method": "cfo",
                    "n_splits": 5,
                    "task": 'regression',
                    "seed": 1,
                    "verbose": 0,
                }

            For an overview of possible parameters for the tuning process check out the
            `FLAML documentation <https://microsoft.github.io/FLAML/docs/reference/automl#automl-objects>`_.

        Returns
        -------
        estimator: DLE
            A new DLE instance to be fitted on reference data.

        Examples
        --------
         Without hyperparameter tuning:

        >>> import nannyml as nml
        >>> reference_df, analysis_df, _ = nml.load_synthetic_car_price_dataset()
        >>> estimator = nml.DLE(
        ...     feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count',
        ...                           'door_count', 'fuel', 'transmission'],
        ...     y_pred='y_pred',
        ...     y_true='y_true',
        ...     timestamp_column_name='timestamp',
        ...     metrics=['rmse', 'rmsle'],
        ...     chunk_size=6000,
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)

        With hyperparameter tuning, using a custom hyperparameter tuning configuration:

        >>> import nannyml as nml
        >>> reference_df, analysis_df, _ = nml.load_synthetic_car_price_dataset()
        >>> estimator = nml.DLE(
        ...     feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count',
        ...                           'door_count', 'fuel', 'transmission'],
        ...     y_pred='y_pred',
        ...     y_true='y_true',
        ...     timestamp_column_name='timestamp',
        ...     metrics=['rmse', 'rmsle'],
        ...     chunk_size=6000,
        ...     tune_hyperparameters=True,
        ...     hyperparameter_tuning_config={
        ...         "time_budget": 60,  # run longer
        ...         "metric": "mse",
        ...         "estimator_list": ['lgbm'],
        ...         "eval_method": "cv",
        ...         "hpo_method": "cfo",
        ...         "n_splits": 5,
        ...         "task": 'regression',
        ...         "seed": 1,
        ...         "verbose": 0,
        ...     }
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)

        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name

        if metrics is None:
            metrics = DEFAULT_METRICS
        self.metrics: List[Metric] = [
            MetricFactory.create(metric, ProblemType.REGRESSION, kwargs={'estimator': self}) for metric in metrics
        ]

        if hyperparameter_tuning_config is None:
            hyperparameter_tuning_config = {
                "time_budget": 15,  # total running time in seconds
                "metric": "mse",
                "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
                "eval_method": "cv",  # resampling strategy
                "hpo_method": "cfo",  # hyperparameter optimization method, cfo is default.
                "n_splits": 5,  # Default Value is 5
                "task": 'regression',  # task type
                "seed": 1,  # random seed
                "verbose": 0,
            }
        self.hyperparameter_tuning_config = hyperparameter_tuning_config
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameters = hyperparameters

        self._categorical_imputer = SimpleImputer(strategy='constant', fill_value='NML_missing_value')
        self._categorical_encoders: defaultdict = defaultdict(LabelEncoder)

    def __str__(self):
        return (
            f"{self.__class__.__name__}[tune_hyperparameters={self.tune_hyperparameters}, "
            f"metrics={[str(m) for m in self.metrics]}]"
        )

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))

        _, categorical_feature_columns = _split_features_by_type(reference_data, self.feature_column_names)
        if len(categorical_feature_columns) > 0:
            reference_data[categorical_feature_columns] = self._categorical_imputer.fit_transform(
                reference_data[categorical_feature_columns]
            )
            reference_data[categorical_feature_columns] = reference_data[categorical_feature_columns].apply(
                lambda x: self._categorical_encoders[x.name].fit_transform(x)
            )

        for metric in self.metrics:
            metric.fit(reference_data)

        self.previous_reference_results = self._estimate(reference_data).data

        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> AbstractEstimatorResult:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred], list(data.columns))

        _, categorical_feature_columns = _split_features_by_type(data, self.feature_column_names)
        if len(categorical_feature_columns) > 0:
            data[categorical_feature_columns] = self._categorical_imputer.transform(data[categorical_feature_columns])
            data[categorical_feature_columns] = data[categorical_feature_columns].apply(
                lambda x: self._categorical_encoders[x.name].transform(x)
            )

        chunks = self.chunker.split(data, timestamp_column_name=self.timestamp_column_name)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    **self._estimate_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        res = res.reset_index(drop=True)
        return Result(results_data=res, estimator=self)

    def _estimate_chunk(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = metric.estimate(chunk.data)
            sampling_error = metric.sampling_error(chunk.data)
            estimates[f'realized_{metric.column_name}'] = metric.realized_performance(chunk.data)
            estimates[f'estimated_{metric.column_name}'] = estimated_metric
            estimates[f'upper_confidence_{metric.column_name}'] = estimated_metric + 3 * sampling_error
            estimates[f'lower_confidence_{metric.column_name}'] = estimated_metric - 3 * sampling_error
            estimates[f'sampling_error_{metric.column_name}'] = sampling_error
            estimates[f'upper_threshold_{metric.column_name}'] = metric.upper_threshold
            estimates[f'lower_threshold_{metric.column_name}'] = metric.lower_threshold
            estimates[f'alert_{metric.column_name}'] = (
                estimated_metric > metric.upper_threshold or estimated_metric < metric.lower_threshold
            )
        return estimates
