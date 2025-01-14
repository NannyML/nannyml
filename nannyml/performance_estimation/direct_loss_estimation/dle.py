#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Classs implementing Direct Loss Estimation algorithm to estimate performance for regression models."""


from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import MultiIndex
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from nannyml._typing import ProblemType, Self
from nannyml.base import AbstractEstimator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation import SUPPORTED_METRIC_VALUES
from nannyml.performance_estimation.direct_loss_estimation.metrics import Metric, MetricFactory
from nannyml.performance_estimation.direct_loss_estimation.result import Result
from nannyml.thresholds import StandardDeviationThreshold, Threshold
from nannyml.usage_logging import UsageEvent, log_usage

DEFAULT_THRESHOLDS: Dict[str, Threshold] = {
    'mae': StandardDeviationThreshold(),
    'mape': StandardDeviationThreshold(),
    'mse': StandardDeviationThreshold(),
    'msle': StandardDeviationThreshold(),
    'rmse': StandardDeviationThreshold(),
    'rmsle': StandardDeviationThreshold(),
}


class DLE(AbstractEstimator):
    """Class implementing the Direct Loss Estimation method.

    The Direct Loss Estimator (DLE) estimates the :term:`loss<Loss>` resulting
    from the difference between the prediction and the target before the targets become known.
    The loss is defined from the regression performance metric
    specified. For all metrics used the loss function is positive.

    It uses an internal
    `LGBMRegressor <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_
    model per metric to predict the value of the error function
    (the function returning the error for a given prediction) of the monitored model.

    The error results on the reference data become a target for those internal models.

    It is possible to specify a set of hyperparameters to instantiate these internal nanny models with using the
    `hyperparameters` parameter. You can also opt to run hyperparameter tuning using FLAML to determine hyperparameters
    for you. Tuning hyperparameters takes some time and does not guarantee better results,
    hence we don't do it by default.

    The estimator manages a list of :class:`~nannyml.performance_estimation.direct_loss_estimation.metrics.Metric`
    instances, constructed using the
    :class:`~nannyml.performance_estimation.direct_loss_estimation.metrics.MetricFactory`.

    The estimator is then responsible for delegating the `fit` and `estimate` method calls to each of the managed
    :class:`~nannyml.performance_estimation.direct_loss_estimation.metrics.Metric` instances and building a
    :class:`~nannyml.performance_estimation.direct_loss_estimation.results.Result` object.

    For more information, check out the `tutorial`_ and the `deep dive`_.

    .. _tutorial:
        https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/regression_performance_estimation.html

    .. _deep dive:
        https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#direct-loss-estimation-dle
    """

    def __init__(
        self,
        feature_column_names: List[str],
        y_pred: str,
        y_true: str,
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tune_hyperparameters: bool = False,
        hyperparameter_tuning_config: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, Threshold]] = None,
    ):
        """Creates a new Direct Loss Estimator.

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
        chunk_size : int, default=None
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number : int, default=None
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period : str, default=None
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.
        metrics : Optional[Union[str, List[str]]], default = ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']
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
        thresholds: dict
            The default values are::

                {
                    'mae': StandardDeviationThreshold(),
                    'mape': StandardDeviationThreshold(),
                    'mse': StandardDeviationThreshold(),
                    'msle': StandardDeviationThreshold(),
                    'rmse': StandardDeviationThreshold(),
                    'rmsle': StandardDeviationThreshold(),
                }

            A dictionary allowing users to set a custom threshold for each method. It links a `Threshold` subclass
            to a method name. This dictionary is optional.
            When a dictionary is given its values will override the default values. If no dictionary is given a default
            will be applied.

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
        >>> metric_fig = results.plot()
        >>> metric_fig.show()

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
        >>> metric_fig = results.plot()
        >>> metric_fig.show()

        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name)

        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true

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

        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(**thresholds)

        if metrics is None:
            metrics = SUPPORTED_METRIC_VALUES
        elif isinstance(metrics, str):
            metrics = [metrics]

        self.metrics: List[Metric] = []
        for metric in metrics:
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(
                    f"unknown metric key '{metric}' given. " f"Should be one of {SUPPORTED_METRIC_VALUES}."
                )

            self.metrics.append(
                MetricFactory.create(
                    metric,
                    ProblemType.REGRESSION,
                    feature_column_names=self.feature_column_names,
                    y_true=self.y_true,
                    y_pred=self.y_pred,
                    chunker=self.chunker,
                    tune_hyperparameters=self.tune_hyperparameters,
                    hyperparameter_tuning_config=self.hyperparameter_tuning_config,
                    hyperparameters=self.hyperparameters,
                    threshold=self.thresholds[metric],
                )
            )

        self._categorical_imputer = SimpleImputer(strategy='constant', fill_value='NML_missing_value')
        self._categorical_encoders: defaultdict = defaultdict(_default_encoder)

        self.result: Optional[Result] = None

    def __str__(self):
        """Get sting representing instantiated class."""
        return (
            f"{self.__class__.__name__}[tune_hyperparameters={self.tune_hyperparameters}, "
            f"metrics={[str(m) for m in self.metrics]}]"
        )

    @log_usage(UsageEvent.DLE_ESTIMATOR_FIT, metadata_from_self=['metrics'])
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        reference_data = reference_data.copy(deep=True)

        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))

        _, categorical_feature_columns = _split_features_by_type(reference_data, self.feature_column_names)
        for categorical_feature_column in categorical_feature_columns:
            reference_data[categorical_feature_column] = reference_data[categorical_feature_column].astype("object")
            reference_data[categorical_feature_column] = self._categorical_imputer.fit_transform(
                reference_data[categorical_feature_column].values.reshape(-1, 1)
            ).ravel()
            reference_data[categorical_feature_column] = self._categorical_encoders[
                categorical_feature_column
            ].fit_transform(reference_data[categorical_feature_column].values.reshape(-1, 1)).ravel()
            # LGBM treats -1 for categorical features as missing
            # https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support
            # Ordinal encoder encodes from 0 to n-1.
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
            # hence we add 1 before we pass the data to lightgbm for fitting so treating unseen values
            # is more consistent
            reference_data[categorical_feature_column] = reference_data[categorical_feature_column] + 1

        for metric in self.metrics:
            metric.categorical_column_names = categorical_feature_columns
            metric.fit(reference_data)

        self.result = self._estimate(reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    @log_usage(UsageEvent.DLE_ESTIMATOR_RUN, metadata_from_self=['metrics'])
    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        data = data.copy(deep=True)

        _list_missing([self.y_pred], list(data.columns))

        _, categorical_feature_columns = _split_features_by_type(data, self.feature_column_names)
        for categorical_feature_column in categorical_feature_columns:
            data[categorical_feature_column] = data[categorical_feature_column].astype("object")
            data[categorical_feature_column] = self._categorical_imputer.transform(
                data[categorical_feature_column].values.reshape(-1, 1)
            ).ravel()
            data[categorical_feature_column] = self._categorical_encoders[categorical_feature_column].transform(
                data[categorical_feature_column].values.reshape(-1, 1)
            ).ravel()
            # LGBM treats -1 for categorical features as missing
            # https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support
            # Ordinal encoder encodes from 0 to n-1.
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
            # hence we add 1 before we pass the data to lightgbm for fitting so treating unseen values
            # is more consistent
            data[categorical_feature_column] = data[categorical_feature_column] + 1

        chunks = self.chunker.split(data)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'period': 'analysis',
                    **self._estimate_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        multilevel_index = _create_multilevel_index([metric.column_name for metric in self.metrics])
        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self.result = Result(
                results_data=res,
                metrics=self.metrics,
                feature_column_names=self.feature_column_names,
                y_pred=self.y_pred,
                y_true=self.y_true,
                timestamp_column_name=self.timestamp_column_name,
                chunker=self.chunker,
                tune_hyperparameters=self.tune_hyperparameters,
                hyperparameter_tuning_config=self.hyperparameter_tuning_config,
                hyperparameters=self.hyperparameters,
            )
        else:
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, res], ignore_index=True)

        return self.result

    def _estimate_chunk(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            try:
                estimated_metric = metric.estimate(chunk.data)
                sampling_error = metric.sampling_error(chunk.data)
                upper_confidence_boundary = estimated_metric + 3 * sampling_error
                if metric.upper_threshold_value_limit is not None:
                    upper_confidence_boundary = min(metric.upper_threshold_value_limit, upper_confidence_boundary)
                lower_confidence_boundary = estimated_metric - 3 * sampling_error
                if metric.lower_threshold_value_limit is not None:
                    lower_confidence_boundary = max(metric.lower_threshold_value_limit, lower_confidence_boundary)
                estimates[f'sampling_error_{metric.column_name}'] = sampling_error
                estimates[f'realized_{metric.column_name}'] = metric.realized_performance(chunk.data)
                estimates[f'estimated_{metric.column_name}'] = estimated_metric
                estimates[f'upper_confidence_{metric.column_name}'] = upper_confidence_boundary
                estimates[f'lower_confidence_{metric.column_name}'] = lower_confidence_boundary
                estimates[f'upper_threshold_{metric.column_name}'] = metric.upper_threshold_value
                estimates[f'lower_threshold_{metric.column_name}'] = metric.lower_threshold_value
                estimates[f'alert_{metric.column_name}'] = metric.alert(estimated_metric)
            except Exception as exc:
                self._logger.error(
                    f"an unexpected error occurred while calculating metric {metric.display_name}: {exc}"
                )
                estimates[f'sampling_error_{metric.column_name}'] = np.nan
                estimates[f'realized_{metric.column_name}'] = np.nan
                estimates[f'estimated_{metric.column_name}'] = np.nan
                estimates[f'upper_confidence_{metric.column_name}'] = np.nan
                estimates[f'lower_confidence_{metric.column_name}'] = np.nan
                estimates[f'upper_threshold_{metric.column_name}'] = metric.upper_threshold_value
                estimates[f'lower_threshold_{metric.column_name}'] = metric.lower_threshold_value
                estimates[f'alert_{metric.column_name}'] = np.nan
        return estimates


def _create_multilevel_index(metric_names: List[str]):
    chunk_column_names = [
        'key',
        'chunk_index',
        'start_index',
        'end_index',
        'start_date',
        'end_date',
        'period',
    ]
    method_column_names = [
        'sampling_error',
        'realized',
        'value',
        'upper_confidence_boundary',
        'lower_confidence_boundary',
        'upper_threshold',
        'lower_threshold',
        'alert',
    ]
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    reconstruction_tuples = [
        (metric_name, column_name) for metric_name in metric_names for column_name in method_column_names
    ]

    tuples = chunk_tuples + reconstruction_tuples

    return MultiIndex.from_tuples(tuples)


def _default_encoder():
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
