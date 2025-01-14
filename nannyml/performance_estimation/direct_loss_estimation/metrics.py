#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""A module containing the implementations of metrics estimated by DLE class.

The :class:`~nannyml.performance_estimation.direct_loss_estimation.dle.DLE` estimator
converts a list of metric names into :class:`~nannyml.performance_estimation.direct_loss_estimation.metrics.Metric`
instances using the :class:`~nannyml.performance_estimation.direct_loss_estimation.metrics.MetricFactory`.

The :class:`~nannyml.performance_estimation.direct_loss_estimation.dle.DLE` estimator will then loop over these
:class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances to fit them on reference data
and run the estimation on analysis data.
"""

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from flaml import AutoML
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
)

from nannyml._typing import ProblemType
from nannyml.base import _raise_exception_for_negative_values, common_nan_removal
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException, InvalidReferenceDataException
from nannyml.sampling_error.regression import (
    mae_sampling_error,
    mae_sampling_error_components,
    mape_sampling_error,
    mape_sampling_error_components,
    mse_sampling_error,
    mse_sampling_error_components,
    msle_sampling_error,
    msle_sampling_error_components,
    rmse_sampling_error,
    rmse_sampling_error_components,
    rmsle_sampling_error,
    rmsle_sampling_error_components,
)
from nannyml.thresholds import Threshold, calculate_threshold_values


class Metric(abc.ABC):
    """A performance metric used to estimate regression performance."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        threshold: Threshold,
        upper_value_limit: Optional[float] = None,
        lower_value_limit: Optional[float] = 0.0,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        upper_value_limit: Optional[float], default=None,
            An optional value that serves as a limit for the upper threshold value. Any calculated upper threshold
            values that end up above this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.
        lower_value_limit: Optional[float], default=0.0,
            An optional value that serves as a limit for the lower threshold value. Any calculated lower threshold
            values that end up below this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.

        """
        self.display_name = display_name
        self.column_name = column_name

        self.feature_column_names = feature_column_names
        self.categorical_column_names: List[str] = []
        self.y_true = y_true
        self.y_pred = y_pred
        self.chunker = chunker

        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_tuning_config = hyperparameter_tuning_config
        self.hyperparameters = hyperparameters

        self.threshold = threshold
        self.upper_threshold_value: Optional[float] = None
        self.lower_threshold_value: Optional[float] = None
        self.upper_threshold_value_limit: Optional[float] = upper_value_limit
        self.lower_threshold_value_limit: Optional[float] = lower_value_limit

        self._dee_model: LGBMRegressor

        self._sampling_error_components: Tuple = ()

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
        """Get string of class name."""
        return self.__class__.__name__

    def fit(self, reference_data: pd.DataFrame):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.

        """
        self._logger.debug(f"fitting {self.__class__.__name__}")

        # Calculate alert thresholds
        reference_chunks = self.chunker.split(reference_data)
        self.lower_threshold_value, self.upper_threshold_value = self._alert_thresholds(
            reference_chunks
        )

        # Delegate to subclass
        self._fit(reference_data)

        return

    @abc.abstractmethod
    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _fit method"
        )

    def estimate(self, data: pd.DataFrame):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to estimate performance metrics for. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated).
        """
        self._logger.debug(f"estimating {self.__class__.__name__}")

        return self._estimate(data)

    @abc.abstractmethod
    def _estimate(self, data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _estimate method"
        )

    def sampling_error(self, data: pd.DataFrame):
        """Calculates the sampling error with respect to the reference data for a given chunk of data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the sampling error on, with respect to the reference data.

        Returns
        -------
        sampling_error: float
            The expected sampling error.
        """
        return self._sampling_error(data)

    @abc.abstractmethod
    def _sampling_error(self, data: pd.DataFrame) -> float:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _sampling_error method"
        )

    def _alert_thresholds(
        self, reference_chunks: List[Chunk]
    ) -> Tuple[Optional[float], Optional[float]]:
        realized_chunk_performance = np.asarray(
            [self.realized_performance(chunk.data) for chunk in reference_chunks]
        )
        lower_threshold_value, upper_threshold_value = calculate_threshold_values(
            threshold=self.threshold,
            data=realized_chunk_performance,
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=self.upper_threshold_value_limit,
            logger=self._logger,
            metric_name=self.display_name,
        )

        return lower_threshold_value, upper_threshold_value

    def alert(self, value: float) -> bool:
        """Returns True if an estimated metric value is below a lower threshold or above an upper threshold.

        Parameters
        ----------
        value: float
            Value of an estimated metric.

        Returns
        -------
        bool: bool
        """
        return (
            self.lower_threshold_value is not None
            and value < self.lower_threshold_value
        ) or (
            self.upper_threshold_value is not None
            and value > self.upper_threshold_value
        )

    @abc.abstractmethod
    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.
        """
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the realized_performance method"
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
        )

    def _train_direct_error_estimation_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]],
        categorical_column_names: List[str],
    ) -> LGBMRegressor:
        if hyperparameters:
            self._logger.debug("'hyperparameters' set: using custom hyperparameters")
            self._logger.debug(f"'hyperparameters': {hyperparameters}")

            model = LGBMRegressor(**hyperparameters)
            model.fit(X_train, y_train, categorical_feature=categorical_column_names)
        elif tune_hyperparameters:
            self._logger.debug(
                f"'tune_hyperparameters' set to '{tune_hyperparameters}': "
                f"performing hyperparameter tuning"
            )
            self._logger.debug(
                "'hyperparameters' not set: using default hyperparameters"
            )
            self._logger.debug(
                f"hyperparameter tuning configuration: {hyperparameter_tuning_config}"
            )

            automl = AutoML()
            # TODO: is this correct? // categorical_feature
            automl.fit(
                X_train,
                y_train,
                **hyperparameter_tuning_config,
                categorical_feature=categorical_column_names,
            )
            self.hyperparameters = {**automl.model.estimator.get_params()}
            model = LGBMRegressor(**automl.model.estimator.get_params())
            model.fit(X_train, y_train, categorical_feature=categorical_column_names)
        else:
            self._logger.debug(
                f"'tune_hyperparameters' set to '{tune_hyperparameters}': skipping hyperparameter tuning"
            )
            model = LGBMRegressor()
            model.fit(X_train, y_train, categorical_feature=categorical_column_names)

        return model


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Dict[ProblemType, Type[Metric]]] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, problem_type: ProblemType, **kwargs) -> Metric:
        """Returns a Metric instance for a given key.

        Parameters
        ----------
        key: str
            string representing metric key of selected metric
        problem_type: ProblemType
            Determines which method to use. Use 'regression' for regression tasks.
        """
        if not isinstance(key, str):
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'"
                "Please provide a string, function or Metric"
            )

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. "
                "Should be one of ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']."
            )

        if problem_type not in cls.registry[key]:
            raise RuntimeError(
                f"metric '{key}' is currently not supported for use case {problem_type}. "
                "Please specify another metric or use one of these supported model types for this metric: "
                f"{[md for md in cls.registry[key]]}"
            )
        metric_class = cls.registry[key][problem_type]
        return metric_class(**kwargs)

    @classmethod
    def register(cls, metric: str, problem_type: ProblemType) -> Callable:
        """Add a metric class to metric registry."""

        def inner_wrapper(wrapped_class: Type[Metric]) -> Type[Metric]:
            if metric in cls.registry:
                if problem_type in cls.registry[metric]:
                    cls._logger().warning(
                        f"re-registering Metric for metric='{metric}' "
                        f"and problem_type='{problem_type}'"
                    )
                cls.registry[metric][problem_type] = wrapped_class
            else:
                cls.registry[metric] = {problem_type: wrapped_class}
            return wrapped_class

        return inner_wrapper


@MetricFactory.register("mae", ProblemType.REGRESSION)
class MAE(Metric):
    """Estimate regression performance using Mean Absolute Error metric."""

    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        threshold: Threshold,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        """Creates a new Mean Absolute Error (MAE) Metric instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        """
        super().__init__(
            display_name="MAE",
            column_name="mae",
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            threshold=threshold,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )

        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mae_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = abs(y_true - y_pred)

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(
            X=data[self.feature_column_names + [self.y_pred]]
        )
        # clip negative predictions to 0
        observation_level_estimates = np.maximum(0, observation_level_estimates)
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        # we only expect predictions to be present and estimate sampling error based on them
        data, empty = common_nan_removal(data[[self.y_pred]], [self.y_pred])
        if empty:
            return np.nan
        else:
            return mae_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.

        Returns
        -------
        mae: float
            Mean Absolute Error
        """
        if self.y_true not in data.columns:
            return np.nan
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]
        return mean_absolute_error(y_true, y_pred)


@MetricFactory.register("mape", ProblemType.REGRESSION)
class MAPE(Metric):
    """Estimate regression performance using Mean Absolute Percentage Error metric."""

    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        threshold: Threshold,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        """Creates a new Mean Absolute Percentage Error (MAPE) Metric instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        """
        super().__init__(
            display_name="MAPE",
            column_name="mape",
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            threshold=threshold,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )

        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mape_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        epsilon = np.finfo(np.float64).eps
        observation_level_metric = abs(y_true - y_pred) / (
            np.maximum(epsilon, abs(y_true))
        )

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(
            X=data[self.feature_column_names + [self.y_pred]]
        )
        # clip negative predictions to 0
        observation_level_estimates = np.maximum(0, observation_level_estimates)
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        # we only expect predictions to be present and estimate sampling error based on them
        data, empty = common_nan_removal(data[[self.y_pred]], [self.y_pred])
        if empty:
            return np.nan
        else:
            return mape_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.

        Returns
        -------
        mape: float
            Mean Absolute Percentage Error
        """
        if self.y_true not in data.columns:
            return np.nan
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]
        return mean_absolute_percentage_error(y_true, y_pred)


@MetricFactory.register("mse", ProblemType.REGRESSION)
class MSE(Metric):
    """Estimate regression performance using Mean Squared Error metric."""

    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        threshold: Threshold,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        """Creates a new Mean Squared Error (MSE) Metric instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        """
        super().__init__(
            display_name="MSE",
            column_name="mse",
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            threshold=threshold,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )

        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mse_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = np.square(y_true - y_pred)

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(
            X=data[self.feature_column_names + [self.y_pred]]
        )
        # clip negative predictions to 0
        observation_level_estimates = np.maximum(0, observation_level_estimates)
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        # we only expect predictions to be present and estimate sampling error based on them
        data, empty = common_nan_removal(data[[self.y_pred]], [self.y_pred])
        if empty:
            return np.nan
        else:
            return mse_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.

        Returns
        -------
        mse: float
            Mean Squared Error
        """
        if self.y_true not in data.columns:
            return np.nan
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            return np.nan
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]
        return mean_squared_error(y_true, y_pred)


@MetricFactory.register("msle", ProblemType.REGRESSION)
class MSLE(Metric):
    """Estimate regression performance using Mean Squared Logarithmic Error metric."""

    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        threshold: Threshold,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        """Creates a new Mean Squared Log Error (MSLE) Metric instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        """
        super().__init__(
            display_name="MSLE",
            column_name="msle",
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            threshold=threshold,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )
        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        self._sampling_error_components = msle_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = np.square(np.log1p(y_true) - np.log1p(y_pred))

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(
            X=data[self.feature_column_names + [self.y_pred]]
        )
        # clip negative predictions to 0
        observation_level_estimates = np.maximum(0, observation_level_estimates)
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        # we only expect predictions to be present and estimate sampling error based on them
        data, empty = common_nan_removal(data[[self.y_pred]], [self.y_pred])
        if empty:
            return np.nan
        else:
            return msle_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.

        Raises
        ------
        _raise_exception_for_negative_values: when any of y_true or y_pred contain negative values.

        Returns
        -------
        msle: float
            Mean Squared Log Error
        """
        if self.y_true not in data.columns:
            return np.nan
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            return np.nan
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]
        return mean_squared_log_error(y_true, y_pred)


@MetricFactory.register("rmse", ProblemType.REGRESSION)
class RMSE(Metric):
    """Estimate regression performance using Root Mean Squared Error metric."""

    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        threshold: Threshold,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        """Creates a new Root Mean Squared Error (RMSE) Metric instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        """
        super().__init__(
            display_name="RMSE",
            column_name="rmse",
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            threshold=threshold,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )

        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = rmse_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = np.square(y_true - y_pred)

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(
            X=data[self.feature_column_names + [self.y_pred]]
        )
        # clip negative predictions to 0
        observation_level_estimates = np.maximum(0, observation_level_estimates)
        chunk_level_estimate = np.sqrt(np.mean(observation_level_estimates))
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        # we only expect predictions to be present and estimate sampling error based on them
        data, empty = common_nan_removal(data[[self.y_pred]], [self.y_pred])
        if empty:
            return np.nan
        else:
            return rmse_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.

        Returns
        -------
        rmse: float
            Root Mean Squared Error
        """
        if self.y_true not in data.columns:
            return np.nan
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            return np.nan
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        # Deal with breaking API change in sklearn 1.4
        # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_error.html
        try:
            from sklearn.metrics import root_mean_squared_error

            return root_mean_squared_error(y_true, y_pred)
        except ImportError:
            from sklearn.metrics import mean_squared_error

            return np.sqrt(mean_squared_error(y_true, y_pred, squared=False))


@MetricFactory.register("rmsle", ProblemType.REGRESSION)
class RMSLE(Metric):
    """Estimate regression performance using Root Mean Squared Logarithmic Error metric."""

    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        threshold: Threshold,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        """Creates a new Root Mean Squared Log Error (RMSLE) Metric instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_true: str,
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred: str,
            The name of the column containing your model predictions.
        chunker: Chunker,
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
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
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        threshold: Threshold,
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        """
        super().__init__(
            display_name="RMSLE",
            column_name="rmsle",
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            threshold=threshold,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data, [self.y_true, self.y_pred]
        )
        if empty:
            raise InvalidReferenceDataException(
                f"Cannot fit DLE for {self.display_name}, too many missing values for predictions and targets."
            )

        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        self._sampling_error_components = rmsle_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = np.square(np.log1p(y_true) - np.log1p(y_pred))

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,
            hyperparameters=self.hyperparameters,
            categorical_column_names=self.categorical_column_names,
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(
            X=data[self.feature_column_names + [self.y_pred]]
        )
        # clip negative predictions to 0
        observation_level_estimates = np.maximum(0, observation_level_estimates)
        chunk_level_estimate = np.sqrt(np.mean(observation_level_estimates))
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        # we only expect predictions to be present and estimate sampling error based on them
        data, empty = common_nan_removal(data[[self.y_pred]], [self.y_pred])
        if empty:
            return np.nan
        else:
            return rmsle_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        """Calculates the realized performance of a model with respect of a given chunk of data.

        The data needs to have both prediction and real targets.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate the realized performance on.

        Raises
        ------
        _raise_exception_for_negative_values: when any of y_true or y_pred contain negative values.

        Returns
        -------
        rmsle: float
            Root Mean Squared Log Error
        """
        if self.y_true not in data.columns:
            return np.nan
        data, empty = common_nan_removal(
            data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            return np.nan
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        # Deal with breaking API change in sklearn 1.4
        # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_log_error.html
        try:
            from sklearn.metrics import root_mean_squared_log_error

            return root_mean_squared_log_error(y_true, y_pred)
        except ImportError:
            from sklearn.metrics import mean_squared_log_error

            return np.sqrt(mean_squared_log_error(y_true, y_pred, squared=False))
