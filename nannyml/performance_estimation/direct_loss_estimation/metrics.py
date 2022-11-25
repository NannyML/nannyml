#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from nannyml.base import _raise_exception_for_negative_values
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
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
        upper_value_limit: Optional[float] = None,
        lower_value_limit: float = 0.0,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        """
        self.display_name = display_name
        self.column_name = column_name

        self.feature_column_names = feature_column_names
        self.y_true = y_true
        self.y_pred = y_pred
        self.chunker = chunker

        self.tune_hyperparameters = tune_hyperparameters  # type: ignore
        self.hyperparameter_tuning_config = hyperparameter_tuning_config  # type: ignore
        self.hyperparameters = hyperparameters  # type: ignore

        self.upper_threshold: Optional[float] = None
        self.lower_threshold: Optional[float] = None
        self.upper_value_limit: Optional[float] = upper_value_limit
        self.lower_value_limit: Optional[float] = lower_value_limit

        self._dee_model: LGBMRegressor

        self._sampling_error_components: Tuple = ()

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def __str__(self):
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
        reference_chunks = self.chunker.split(
            reference_data,
        )
        self.lower_threshold, self.upper_threshold = self._alert_thresholds(
            reference_chunks, lower_limit=self.lower_value_limit, upper_limit=self.upper_value_limit
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
        self,
        reference_chunks: List[Chunk],
        std_num: int = 3,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        realized_chunk_performance = [self.realized_performance(chunk.data) for chunk in reference_chunks]
        deviation = np.std(realized_chunk_performance) * std_num
        mean_realised_performance = np.mean(realized_chunk_performance)
        lower_threshold = mean_realised_performance - deviation
        if lower_limit is not None:
            lower_threshold = np.maximum(lower_threshold, lower_limit)

        # Special case... in case lower threshold equals 0, it should not be shown at all
        if lower_threshold == 0.0:
            lower_threshold = None  # type: ignore

        upper_threshold = mean_realised_performance + deviation
        if upper_limit is not None:
            upper_threshold = np.minimum(upper_threshold, upper_limit)

        return lower_threshold, upper_threshold  # type: ignore

    @abc.abstractmethod
    def realized_performance(self, data: pd.DataFrame) -> float:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the realized_performance method"
        )

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return self.display_name == other.display_name and self.column_name == other.column_name

    def _common_cleaning(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        clean_targets = self.y_true in data.columns and not data[self.y_true].isna().all()

        y_pred = data[self.y_pred]
        if clean_targets:
            y_true = data[self.y_true]
            y_pred = y_pred[~y_true.isna()]
            y_true.dropna(inplace=True)
        else:
            y_true = None

        return y_pred, y_true

    def _train_direct_error_estimation_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]],
    ) -> LGBMRegressor:
        if hyperparameters:
            self._logger.debug("'hyperparameters' set: using custom hyperparameters")
            self._logger.debug(f"'hyperparameters': {hyperparameters}")

            model = LGBMRegressor(**hyperparameters)
            model.fit(X_train, y_train)
        elif tune_hyperparameters:
            self._logger.debug(
                f"'tune_hyperparameters' set to '{tune_hyperparameters}': " f"performing hyperparameter tuning"
            )
            self._logger.debug("'hyperparameters' not set: using default hyperparameters")
            self._logger.debug(f'hyperparameter tuning configuration: {hyperparameter_tuning_config}')

            automl = AutoML()
            automl.fit(X_train, y_train, **hyperparameter_tuning_config)
            self.hyperparameters = {**automl.model.estimator.get_params()}  # type: ignore
            model = LGBMRegressor(**automl.model.estimator.get_params())
            model.fit(X_train, y_train)
        else:
            self._logger.debug(
                f"'tune_hyperparameters' set to '{tune_hyperparameters}': skipping hyperparameter tuning"
            )
            model = LGBMRegressor()
            model.fit(X_train, y_train)

        return model


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Dict[ProblemType, Metric]] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, problem_type: ProblemType, **kwargs) -> Metric:
        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'" "Please provide a string, function or Metric"
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
        return metric_class(**kwargs)  # type: ignore

    @classmethod
    def register(cls, metric: str, problem_type: ProblemType) -> Callable:
        def inner_wrapper(wrapped_class: Metric) -> Metric:
            if metric in cls.registry:
                if problem_type in cls.registry[metric]:
                    cls._logger().warning(
                        f"re-registering Metric for metric='{metric}' " f"and problem_type='{problem_type}'"
                    )
                cls.registry[metric][problem_type] = wrapped_class
            else:
                cls.registry[metric] = {problem_type: wrapped_class}
            return wrapped_class

        return inner_wrapper


@MetricFactory.register('mae', ProblemType.REGRESSION)
class MAE(Metric):
    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        super().__init__(
            display_name='MAE',
            column_name='mae',
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mae_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = abs(y_true - y_pred)

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,  # type: ignore
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,  # type: ignore
            hyperparameters=self.hyperparameters,  # type: ignore
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(X=data[self.feature_column_names + [self.y_pred]])
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mae_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        y_pred, y_true = self._common_cleaning(data)

        if y_true is None:
            return np.NaN

        return mean_absolute_error(y_true, y_pred)


@MetricFactory.register('mape', ProblemType.REGRESSION)
class MAPE(Metric):
    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        super().__init__(
            display_name='MAPE',
            column_name='mape',
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mape_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        epsilon = np.finfo(np.float64).eps
        observation_level_metric = abs(y_true - y_pred) / (np.maximum(epsilon, abs(y_true)))

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,  # type: ignore
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,  # type: ignore
            hyperparameters=self.hyperparameters,  # type: ignore
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(X=data[self.feature_column_names + [self.y_pred]])
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mape_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        y_pred, y_true = self._common_cleaning(data)

        if y_true is None:
            return np.NaN

        return mean_absolute_percentage_error(y_true, y_pred)


@MetricFactory.register('mse', ProblemType.REGRESSION)
class MSE(Metric):
    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        super().__init__(
            display_name='MSE',
            column_name='mse',
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = mse_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = np.square(y_true - y_pred)

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,  # type: ignore
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,  # type: ignore
            hyperparameters=self.hyperparameters,  # type: ignore
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(X=data[self.feature_column_names + [self.y_pred]])
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        y_pred, y_true = self._common_cleaning(data)

        if y_true is None:
            return np.NaN

        return mean_squared_error(y_true, y_pred)


@MetricFactory.register('msle', ProblemType.REGRESSION)
class MSLE(Metric):
    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        super().__init__(
            display_name='MSLE',
            column_name='msle',
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
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
            tune_hyperparameters=self.tune_hyperparameters,  # type: ignore
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,  # type: ignore
            hyperparameters=self.hyperparameters,  # type: ignore
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(X=data[self.feature_column_names + [self.y_pred]])
        chunk_level_estimate = np.mean(observation_level_estimates)
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return msle_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        y_pred, y_true = self._common_cleaning(data)

        if y_true is None:
            return np.NaN

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        return mean_squared_log_error(y_true, y_pred)


@MetricFactory.register('rmse', ProblemType.REGRESSION)
class RMSE(Metric):
    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        super().__init__(
            display_name='RMSE',
            column_name='rmse',
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
        y_true = reference_data[self.y_true]
        y_pred = reference_data[self.y_pred]

        self._sampling_error_components = rmse_sampling_error_components(
            y_true_reference=y_true, y_pred_reference=y_pred
        )

        observation_level_metric = np.square(y_true - y_pred)

        self._dee_model = self._train_direct_error_estimation_model(
            X_train=reference_data[self.feature_column_names + [self.y_pred]],
            y_train=observation_level_metric,
            tune_hyperparameters=self.tune_hyperparameters,  # type: ignore
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,  # type: ignore
            hyperparameters=self.hyperparameters,  # type: ignore
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(X=data[self.feature_column_names + [self.y_pred]])
        chunk_level_estimate = np.sqrt(np.mean(observation_level_estimates))
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return rmse_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        y_pred, y_true = self._common_cleaning(data)

        if y_true is None:
            return np.NaN

        return mean_squared_error(y_true, y_pred, squared=False)


@MetricFactory.register('rmsle', ProblemType.REGRESSION)
class RMSLE(Metric):
    def __init__(
        self,
        feature_column_names: List[str],
        y_true: str,
        y_pred: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ):
        super().__init__(
            display_name='RMSLE',
            column_name='rmsle',
            feature_column_names=feature_column_names,
            y_true=y_true,
            y_pred=y_pred,
            chunker=chunker,
            tune_hyperparameters=tune_hyperparameters,
            hyperparameter_tuning_config=hyperparameter_tuning_config,
            hyperparameters=hyperparameters,
        )

    def _fit(self, reference_data: pd.DataFrame):
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
            tune_hyperparameters=self.tune_hyperparameters,  # type: ignore
            hyperparameter_tuning_config=self.hyperparameter_tuning_config,  # type: ignore
            hyperparameters=self.hyperparameters,  # type: ignore
        )

    def _estimate(self, data: pd.DataFrame):
        observation_level_estimates = self._dee_model.predict(X=data[self.feature_column_names + [self.y_pred]])
        chunk_level_estimate = np.sqrt(np.mean(observation_level_estimates))
        return chunk_level_estimate

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return rmsle_sampling_error(self._sampling_error_components, data)

    def realized_performance(self, data: pd.DataFrame) -> float:
        y_pred, y_true = self._common_cleaning(data)

        if y_true is None:
            return np.NaN

        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)

        return mean_squared_log_error(y_true, y_pred, squared=False)
