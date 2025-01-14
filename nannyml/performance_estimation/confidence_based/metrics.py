"""A module containing the implementations of metrics estimated by CBPE.

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator converts a list of metric names into
:class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances using the
:class:`~nannyml.performance_estimation.confidence_based.metrics.MetricFactory`.

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator will then loop over these
:class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances to fit them on reference data
and run the estimation on analysis data.
"""

import abc
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer, label_binarize

import nannyml.sampling_error.binary_classification as bse
import nannyml.sampling_error.multiclass_classification as mse
from nannyml._typing import ModelOutputsType, ProblemType, class_labels, model_output_column_names
from nannyml.base import _list_missing, common_nan_removal
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import CalculatorException, InvalidArgumentsException
from nannyml.performance_estimation.confidence_based import SUPPORTED_METRIC_VALUES
from nannyml.sampling_error import SAMPLING_ERROR_RANGE
from nannyml.thresholds import Threshold, calculate_threshold_values


class Metric(abc.ABC):
    """A base class representing a performance metric to estimate."""

    def __init__(
        self,
        name: str,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        components: List[Tuple[str, str]],
        timestamp_column_name: Optional[str] = None,
        lower_threshold_value_limit: Optional[float] = None,
        upper_threshold_value_limit: Optional[float] = None,
        **kwargs,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        name: str
            The name used to indicate the metric in columns of a DataFrame.
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string referring to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name
                  containing model outputs for that class.
        y_pred: str
            The name of the column containing your model predictions.
        y_true: str
            The name of the column containing target values (that are provided in reference data during fitting).
        chunker: Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        threshold: Threshold
            The Threshold instance that determines how the lower and upper threshold values will be calculated.
        components: List[Tuple[str str]]
            A list of (display_name, column_name) tuples.
        timestamp_column_name: Optional[str], default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        lower_threshold_value_limit: Optional[float], default=None
            An optional value that serves as a limit for the lower threshold value. Any calculated lower threshold
            values that end up below this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.
        upper_threshold_value_limit: Optional[float], default=None
            An optional value that serves as a limit for the upper threshold value. Any calculated upper threshold
            values that end up above this limit will be replaced by this limit value.
            The limit is often a theoretical constraint enforced by a specific drift detection method or performance
            metric.

        Notes
        -----
        The `components` approach taken here is a quick fix to deal with metrics that return multiple values.
        Look at the `confusion_matrix` for example: a single metric produces 4 different result sets (containing values,
        thresholds, alerts, etc.).
        """
        self.name = name

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name
        self.chunker = chunker

        self.threshold = threshold
        self.lower_threshold_value: Optional[float] = None
        self.upper_threshold_value: Optional[float] = None
        self.lower_threshold_value_limit: Optional[float] = lower_threshold_value_limit
        self.upper_threshold_value_limit: Optional[float] = upper_threshold_value_limit

        self.uncalibrated_y_pred_proba = f'uncalibrated_{self.y_pred_proba}'

        # A list of (display_name, column_name) tuples
        self.components: List[Tuple[str, str]] = components

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @property
    def display_name(self) -> str:  # noqa: D102
        return self.name

    @property
    def column_name(self) -> str:  # noqa: D102
        return self.components[0][1]

    @property
    def display_names(self):  # noqa: D102
        return [c[0] for c in self.components]

    @property
    def column_names(self):  # noqa: D102
        return [c[1] for c in self.components]

    def __str__(self):  # noqa: D105
        return self.display_name

    def __repr__(self):  # noqa: D105
        return self.column_name

    def fit(self, reference_data: pd.DataFrame):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.

        """
        # Delegate to subclass
        self._fit(reference_data)

        reference_chunks = self.chunker.split(reference_data)

        # Calculate alert thresholds
        reference_chunk_results = np.asarray([self._realized_performance(chunk.data) for chunk in reference_chunks])
        self.lower_threshold_value, self.upper_threshold_value = calculate_threshold_values(
            threshold=self.threshold,
            data=reference_chunk_results,
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=self.upper_threshold_value_limit,
            logger=self._logger,
            metric_name=self.display_name,
        )
        return

    @abc.abstractmethod
    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _fit method"
        )

    @abc.abstractmethod
    def _estimate(self, data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _estimate method"
        )

    @abc.abstractmethod
    def _sampling_error(self, data: pd.DataFrame) -> float:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _sampling_error method"
        )

    @abc.abstractmethod
    def _realized_performance(self, data: pd.DataFrame) -> float:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the realized_performance method"
        )

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
        return (self.lower_threshold_value is not None and value < self.lower_threshold_value) or (
            self.upper_threshold_value is not None and value > self.upper_threshold_value
        )

    def __eq__(self, other):
        """Compares two Metric instances.

        They are considered equal when their components are equal.

        Parameters
        ----------
        other: Metric
            The other Metric instance you're comparing to.

        Returns
        -------
        is_equal: bool
        """
        return self.components == other.components

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing the performance metrics for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Raises
        ------
            NotImplementedError: occurs when a metric has multiple componets

        Returns
        -------
            chunk_record : Dict
                A dictionary of perfomance metric, value pairs.
        """
        if len(self.components) > 1:
            raise NotImplementedError(
                "cannot use default 'get_chunk_record' implementation when a metric has multiple components."
            )

        column_name = self.components[0][1]

        chunk_record = {}

        try:
            estimated_metric_value = self._estimate(chunk_data)
            metric_estimate_sampling_error = self._sampling_error(chunk_data)
            chunk_record[f'estimated_{column_name}'] = estimated_metric_value
            chunk_record[f'sampling_error_{column_name}'] = metric_estimate_sampling_error
            chunk_record[f'realized_{column_name}'] = self._realized_performance(chunk_data)
            chunk_record[f'upper_confidence_boundary_{column_name}'] = np.minimum(
                np.inf if self.upper_threshold_value_limit is None else self.upper_threshold_value_limit,
                estimated_metric_value + SAMPLING_ERROR_RANGE * metric_estimate_sampling_error,
            )
            chunk_record[f'lower_confidence_boundary_{column_name}'] = np.maximum(
                -np.inf if self.lower_threshold_value_limit is None else self.lower_threshold_value_limit,
                estimated_metric_value - SAMPLING_ERROR_RANGE * metric_estimate_sampling_error,
            )
            chunk_record[f'upper_threshold_{column_name}'] = self.upper_threshold_value
            chunk_record[f'lower_threshold_{column_name}'] = self.lower_threshold_value
            chunk_record[f'alert_{column_name}'] = self.alert(estimated_metric_value)
        except Exception as exc:
            self._logger.error(f"an unexpected error occurred while calculating metric {self.display_name}: {exc}")
            chunk_record[f'estimated_{column_name}'] = np.nan
            chunk_record[f'sampling_error_{column_name}'] = np.nan
            chunk_record[f'realized_{column_name}'] = np.nan
            chunk_record[f'upper_confidence_boundary_{column_name}'] = np.nan
            chunk_record[f'lower_confidence_boundary_{column_name}'] = np.nan
            chunk_record[f'upper_threshold_{column_name}'] = np.nan
            chunk_record[f'lower_threshold_{column_name}'] = np.nan
            chunk_record[f'alert_{column_name}'] = np.nan
        finally:
            return chunk_record


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Dict[ProblemType, Type[Metric]]] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, use_case: ProblemType, **kwargs) -> Metric:
        """Create new Metric."""
        if kwargs is None:
            kwargs = {}

        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'" "Please provide a string, function or Metric"
            )

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. " f"Should be one of {SUPPORTED_METRIC_VALUES}."
            )

        if use_case not in cls.registry[key]:
            raise RuntimeError(
                f"metric '{key}' is currently not supported for use case {use_case}. "
                "Please specify another metric or use one of these supported model types for this metric: "
                f"{[md for md in cls.registry[key]]}"
            )
        metric_class = cls.registry[key][use_case]
        return metric_class(**kwargs)

    @classmethod
    def register(cls, metric: str, use_case: ProblemType) -> Callable:
        """Register a Metric in the MetricFactory registry."""

        def inner_wrapper(wrapped_class: Type[Metric]) -> Type[Metric]:
            if metric in cls.registry:
                if use_case in cls.registry[metric]:
                    cls._logger().warning(f"re-registering Metric for metric='{metric}' and use_case='{use_case}'")
                cls.registry[metric][use_case] = wrapped_class
            else:
                cls.registry[metric] = {use_case: wrapped_class}
            return wrapped_class

        return inner_wrapper


@MetricFactory.register('roc_auc', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    """CBPE binary classification AUROC Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification AUROC Metric Class."""
        super().__init__(
            name='roc_auc',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('ROC AUC', 'roc_auc')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        data = reference_data[[self.y_true, self.y_pred_proba]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred_proba])
        if empty:
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.auroc_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_proba_reference=reference_data[self.y_pred_proba],
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.uncalibrated_y_pred_proba], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(
            data[[self.y_pred_proba, self.uncalibrated_y_pred_proba]],
            [self.y_pred_proba, self.uncalibrated_y_pred_proba],
        )
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred_proba = data[self.y_pred_proba]
        uncalibrated_y_pred_proba = data[self.uncalibrated_y_pred_proba]
        return estimate_roc_auc(y_pred_proba, uncalibrated_y_pred_proba)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.uncalibrated_y_pred_proba, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(
            data[[self.uncalibrated_y_pred_proba, self.y_true]], [self.uncalibrated_y_pred_proba, self.y_true]
        )
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        uncalibrated_y_pred_proba = data[self.uncalibrated_y_pred_proba]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"'{self.y_true}' contains a single class for chunk, " f"cannot compute realized {self.display_name}."
            )
            return np.nan
        return roc_auc_score(y_true, uncalibrated_y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.uncalibrated_y_pred_proba]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.uncalibrated_y_pred_proba])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.auroc_sampling_error(self._sampling_error_components, data)


def estimate_roc_auc(
    true_y_pred_proba: Union[pd.Series, np.ndarray],
    model_y_pred_proba: Union[pd.Series, np.ndarray]
) -> float:
    """Estimates the ROC AUC metric.

    Parameters
    ----------
    true_y_pred_proba : Union[pd.Series, np.ndarray]
        Calibrated score predictions from the model.
    model_y_pred_proba : Union[pd.Series, np.ndarray]
        Un-Calibrated score predictions from the model.

    Returns
    -------
    metric: float
        Estimated ROC AUC score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    true_y_pred_proba = np.asarray(true_y_pred_proba)
    model_y_pred_proba = np.asarray(model_y_pred_proba)

    sorted_index = np.argsort(model_y_pred_proba)[::-1]
    model_y_pred_proba = model_y_pred_proba[sorted_index]
    true_y_pred_proba = true_y_pred_proba[sorted_index]

    with np.errstate(divide='ignore', invalid='ignore'):
        tps = np.cumsum(true_y_pred_proba)
        fps = 1 + np.arange(len(true_y_pred_proba)) - tps
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        tps = np.round(tps, 5)
        fps = np.round(fps, 5)
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        metric = auc(fpr, tpr)
        return metric


@MetricFactory.register('average_precision', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAP(Metric):
    """CBPE binary classification AP Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification AP Metric Class."""
        super().__init__(
            name='average_precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Average Precision', 'average_precision')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        """Metric _fit implementation on reference data."""
        data = reference_data[[self.y_true, self.y_pred_proba]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred_proba])
        y_true = data[self.y_true]
        y_pred_proba = data[self.y_pred_proba]

        # if empty then positive class won't be part of y_true series
        if 1 not in y_true.unique():
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.ap_sampling_error_components(
                y_true_reference=y_true,
                y_pred_proba_reference=y_pred_proba,
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.uncalibrated_y_pred_proba], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(
            data[[self.y_pred_proba, self.uncalibrated_y_pred_proba]],
            [self.y_pred_proba, self.uncalibrated_y_pred_proba],
        )
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        calibrated_y_pred_proba = data[self.y_pred_proba].to_numpy()
        uncalibrated_y_pred_proba = data[self.uncalibrated_y_pred_proba].to_numpy()
        return estimate_ap(calibrated_y_pred_proba, uncalibrated_y_pred_proba)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.uncalibrated_y_pred_proba, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, _ = common_nan_removal(
            data[[self.uncalibrated_y_pred_proba, self.y_true]], [self.uncalibrated_y_pred_proba, self.y_true]
        )

        y_true = data[self.y_true]
        uncalibrated_y_pred_proba = data[self.uncalibrated_y_pred_proba]

        # if empty then positive class won't be part of y_true series
        if 1 not in y_true.unique():
            warnings.warn(
                f"'{self.y_true}' does not contain positive class for chunk, cannot calculate {self.display_name}. "
                f"Returning NaN."
            )
            return np.nan
        else:
            return average_precision_score(y_true, uncalibrated_y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.uncalibrated_y_pred_proba]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.uncalibrated_y_pred_proba])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.ap_sampling_error(self._sampling_error_components, data)


def estimate_ap(
    calibrated_y_pred_proba: Union[pd.Series, np.ndarray],
    uncalibrated_y_pred_proba: Union[pd.Series, np.ndarray]
) -> float:
    """Estimates the AP metric.

    Parameters
    ----------
    calibrated_y_pred_proba: Union[pd.Series, np.ndarray]
        Calibrated probability estimates of the sample for each class in the model.
    uncalibrated_y_pred_proba: Union[pd.Series, np.ndarray]
        Raw probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated AP score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    calibrated_y_pred_proba = np.asarray(calibrated_y_pred_proba)
    uncalibrated_y_pred_proba = np.asarray(uncalibrated_y_pred_proba)

    descending_order_index = np.argsort(uncalibrated_y_pred_proba)[::-1]
    calibrated_y_pred_proba = calibrated_y_pred_proba[descending_order_index]

    tps = np.cumsum(calibrated_y_pred_proba)
    fps = 1 + np.arange(calibrated_y_pred_proba.shape[0]) - tps
    tps = np.round(tps, 5)
    fps = np.round(fps, 5)
    ps = np.arange(1, tps.shape[0] + 1)

    precision = tps / ps
    # we add an element for (tps, fps) = (0,0) after the division to avoid error
    precision = np.r_[1, precision]
    tps = np.r_[0, tps]
    recall = tps / tps[-1]
    # reverse so (0,1) is last element
    # recall is descending from 1 to 0
    precision = precision[::-1]
    recall = recall[::-1]

    # actual AP calculation
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py#L236
    # non unique values will be eliminated because diff will be 0!
    metric = -np.sum(np.diff(recall) * precision[:-1])
    return metric


@MetricFactory.register('f1', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationF1(Metric):
    """CBPE binary classification f1 Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification f1 Metric Class."""
        super().__init__(
            name='f1',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('F1', 'f1')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans
        data = reference_data[[self.y_true, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if empty:
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.f1_sampling_error_components(
                y_true_reference=y_true,
                y_pred_reference=y_pred,
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]
        return estimate_f1(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.f1_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred, self.y_true]], [self.y_pred, self.y_true])
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in '{self.y_true}', "
                f"returning NaN as realized {self.display_name} score."
            )
            return np.nan

        if y_pred.nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in '{self.y_pred}', "
                f"returning NaN as realized {self.display_name} score."
            )
            return np.nan
        # TODO: zero_division should be np.nan
        # update when we update sklearn to 1.3+ and remove unnecessary checks.
        return f1_score(y_true=y_true, y_pred=y_pred, zero_division='warn')


def estimate_f1(y_pred: Union[pd.Series, np.ndarray], y_pred_proba: Union[pd.Series, np.ndarray]) -> float:
    """Estimates the F1 metric.

    Parameters
    ----------
    y_pred: Union[pd.Series, np.ndarray]
        Predicted class labels of the sample
    y_pred_proba: Union[pd.Series, np.ndarray]
        Probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated F1 score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    y_pred = np.asarray(y_pred)
    y_pred_proba = np.asarray(y_pred_proba)

    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    fn = np.where(y_pred == 0, y_pred_proba, 0)
    TP, FP, FN = np.sum(tp), np.sum(fp), np.sum(fn)

    denominator = TP + 0.5 * (FP + FN)
    return TP / denominator if denominator != 0 else 0


@MetricFactory.register('precision', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationPrecision(Metric):
    """CBPE binary classification precision Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification precision Metric Class."""
        super().__init__(
            name='precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Precision', 'precision')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans
        data = reference_data[[self.y_true, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if empty:
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.precision_sampling_error_components(
                y_true_reference=y_true,
                y_pred_reference=y_pred,
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]
        return estimate_precision(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred, self.y_true]], [self.y_pred, self.y_true])
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in '{self.y_true}', "
                f"returning NaN as realized {self.display_name} score."
            )
            return np.nan

        if y_pred.nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in '{self.y_pred}', "
                f"returning NaN as realized {self.display_name} score."
            )
            return np.nan
        # TODO: zero_division should be np.nan
        # update when we update sklearn to 1.3+ and remove unnecessary checks.
        return precision_score(y_true=y_true, y_pred=y_pred, zero_division='warn')


def estimate_precision(y_pred: Union[pd.Series, np.ndarray], y_pred_proba: Union[pd.Series, np.ndarray]) -> float:
    """Estimates the Precision metric.

    Parameters
    ----------
    y_pred: Union[pd.Series, np.ndarray]
        Predicted class labels of the sample
    y_pred_proba: Union[pd.Series, np.ndarray]
        Probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated Precision score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    y_pred = np.asarray(y_pred)
    y_pred_proba = np.asarray(y_pred_proba)

    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    TP, FP = np.sum(tp), np.sum(fp)

    denominator = TP + FP
    return TP / denominator if denominator != 0 else 0


@MetricFactory.register('recall', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationRecall(Metric):
    """CBPE binary classification recall Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification recall Metric Class."""
        super().__init__(
            name='recall',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Recall', 'recall')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans
        data = reference_data[[self.y_true, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if empty:
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.recall_sampling_error_components(
                y_true_reference=y_true,
                y_pred_reference=y_pred,
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]
        return estimate_recall(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.recall_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred, self.y_true]], [self.y_pred, self.y_true])
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if y_true.nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in '{self.y_true}', "
                f"returning NaN as realized {self.display_name} score."
            )
            return np.nan

        if y_pred.nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in '{self.y_pred}', "
                f"returning NaN as realized {self.display_name} score."
            )
            return np.nan
        # TODO: zero_division should be np.nan
        # update when we update sklearn to 1.3+ and remove unnecessary checks.
        return recall_score(y_true=y_true, y_pred=y_pred, zero_division='warn')


def estimate_recall(y_pred: Union[pd.Series, np.ndarray], y_pred_proba: Union[pd.Series, np.ndarray]) -> float:
    """Estimates the Recall metric.

    Parameters
    ----------
    y_pred: Union[pd.Series, np.ndarray]
        Predicted class labels of the sample
    y_pred_proba: Union[pd.Series, np.ndarray]
        Probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated Recall score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    y_pred = np.asarray(y_pred)
    y_pred_proba = np.asarray(y_pred_proba)

    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fn = np.where(y_pred == 0, y_pred_proba, 0)
    TP, FN = np.sum(tp), np.sum(fn)

    denominator = TP + FN
    return TP / denominator if denominator != 0 else 0


@MetricFactory.register('specificity', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationSpecificity(Metric):
    """CBPE binary classification specificity Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification specificity Metric Class."""
        super().__init__(
            name='specificity',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Specificity', 'specificity')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

        # Set labels expected in y_true/y_pred. Currently hard-coded to 0, 1 for binary classification
        self._labels = [0, 1]

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans
        data = reference_data[[self.y_true, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if empty:
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.specificity_sampling_error_components(
                y_true_reference=y_true,
                y_pred_reference=y_pred,
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]
        return estimate_specificity(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.specificity_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred, self.y_true]], [self.y_pred, self.y_true])
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=self._labels).ravel()
        denominator = tn + fp
        if denominator == 0:
            return np.nan
        else:
            return tn / denominator


def estimate_specificity(y_pred: Union[pd.Series, np.ndarray], y_pred_proba: Union[pd.Series, np.ndarray]) -> float:
    """Estimates the Specificity metric.

    Parameters
    ----------
    y_pred: Union[pd.Series, np.ndarray]
        Predicted class labels of the sample
    y_pred_proba: Union[pd.Series, np.ndarray]
        Probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated Specificity score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    y_pred = np.asarray(y_pred)
    y_pred_proba = np.asarray(y_pred_proba)

    tn = np.where(y_pred == 0, 1 - y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    TN, FP = np.sum(tn), np.sum(fp)

    denominator = TN + FP
    return TN / denominator if denominator != 0 else 0


@MetricFactory.register('accuracy', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAccuracy(Metric):
    """CBPE binary classification accuracy Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification accuracy Metric Class."""
        super().__init__(
            name='accuracy',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Accuracy', 'accuracy')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans
        data = reference_data[[self.y_true, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if empty:
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, 0
        else:
            self._sampling_error_components = bse.accuracy_sampling_error_components(
                y_true_reference=y_true,
                y_pred_reference=y_pred,
            )

    def _estimate(self, data: pd.DataFrame):
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]
        return estimate_accuracy(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.accuracy_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred, self.y_true]], [self.y_pred, self.y_true])
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        return accuracy_score(y_true=y_true, y_pred=y_pred)


def estimate_accuracy(y_pred: Union[pd.Series, np.ndarray], y_pred_proba: Union[pd.Series, np.ndarray]) -> float:
    """Estimates the accuracy metric.

    Parameters
    ----------
    y_pred: Union[pd.Series, np.ndarray]
        Predicted class labels of the sample
    y_pred_proba: Union[pd.Series, np.ndarray]
        Probability estimates of the sample for each class in the model.

    Returns
    -------
    metric: float
        Estimated accuracy score.
    """
    # TODO: Update Code to only accept np.ndarray (and add checkand remove code below)
    y_pred = np.asarray(y_pred)
    y_pred_proba = np.asarray(y_pred_proba)

    tp = np.where(y_pred == 1, y_pred_proba, 0)
    tn = np.where(y_pred == 0, 1 - y_pred_proba, 0)
    TP, TN = np.sum(tp), np.sum(tn)
    metric = (TP + TN) / len(y_pred)
    return metric


@MetricFactory.register('confusion_matrix', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationConfusionMatrix(Metric):
    """CBPE binary classification confusion matrix Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        normalize_confusion_matrix: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification confusion matrix Metric Class."""
        super().__init__(
            name='confusion_matrix',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[
                ('True Positive', 'true_positive'),
                ('True Negative', 'true_negative'),
                ('False Positive', 'false_positive'),
                ('False Negative', 'false_negative'),
            ],
            lower_threshold_value_limit=0,
        )

        self.normalize_confusion_matrix: Optional[str] = normalize_confusion_matrix
        if self.normalize_confusion_matrix is not None:
            self.upper_threshold_value_limit = 1
        self.true_positive_lower_threshold: Optional[float] = None
        self.true_positive_upper_threshold: Optional[float] = None
        self.true_negative_lower_threshold: Optional[float] = None
        self.true_negative_upper_threshold: Optional[float] = None
        self.false_positive_lower_threshold: Optional[float] = None
        self.false_positive_upper_threshold: Optional[float] = None
        self.false_negative_lower_threshold: Optional[float] = None
        self.false_negative_upper_threshold: Optional[float] = None

        # Set labels expected in y_true/y_pred. Currently hard-coded to 0, 1 for binary classification
        self._labels = [0, 1]

    def fit(self, reference_data: pd.DataFrame):  # override the superclass fit method
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.
        """
        # Calculate alert thresholds
        reference_chunks = self.chunker.split(
            reference_data,
        )

        self.true_positive_lower_threshold, self.true_positive_upper_threshold = self._true_positive_alert_thresholds(
            reference_chunks
        )
        self.true_negative_lower_threshold, self.true_negative_upper_threshold = self._true_negative_alert_thresholds(
            reference_chunks
        )
        (
            self.false_positive_lower_threshold,
            self.false_positive_upper_threshold,
        ) = self._false_positive_alert_thresholds(reference_chunks)
        (
            self.false_negative_lower_threshold,
            self.false_negative_upper_threshold,
        ) = self._false_negative_alert_thresholds(reference_chunks)

        # Delegate to confusion matrix subclass
        self._fit(reference_data)  # could probably put _fit functionality here since overide fit method

        return

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._true_positive_sampling_error_components = np.nan, 0.0, self.normalize_confusion_matrix
            self._true_negative_sampling_error_components = np.nan, 0.0, self.normalize_confusion_matrix
            self._false_positive_sampling_error_components = np.nan, 0.0, self.normalize_confusion_matrix
            self._false_negative_sampling_error_components = np.nan, 0.0, self.normalize_confusion_matrix
        else:
            self._true_positive_sampling_error_components = bse.true_positive_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
                normalize_confusion_matrix=self.normalize_confusion_matrix,
            )
            self._true_negative_sampling_error_components = bse.true_negative_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
                normalize_confusion_matrix=self.normalize_confusion_matrix,
            )
            self._false_positive_sampling_error_components = bse.false_positive_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
                normalize_confusion_matrix=self.normalize_confusion_matrix,
            )
            self._false_negative_sampling_error_components = bse.false_negative_sampling_error_components(
                y_true_reference=reference_data[self.y_true],
                y_pred_reference=reference_data[self.y_pred],
                normalize_confusion_matrix=self.normalize_confusion_matrix,
            )

    def _true_positive_alert_thresholds(self, reference_chunks: List[Chunk]) -> Tuple[Optional[float], Optional[float]]:
        realized_chunk_performance = np.asarray(
            [self._true_positive_realized_performance(chunk.data) for chunk in reference_chunks]
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

    def _true_negative_alert_thresholds(self, reference_chunks: List[Chunk]) -> Tuple[Optional[float], Optional[float]]:
        realized_chunk_performance = np.asarray(
            [self._true_negative_realized_performance(chunk.data) for chunk in reference_chunks]
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

    def _false_positive_alert_thresholds(
        self, reference_chunks: List[Chunk]
    ) -> Tuple[Optional[float], Optional[float]]:
        realized_chunk_performance = np.asarray(
            [self._false_positive_realized_performance(chunk.data) for chunk in reference_chunks]
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

    def _false_negative_alert_thresholds(
        self, reference_chunks: List[Chunk]
    ) -> Tuple[Optional[float], Optional[float]]:
        realized_chunk_performance = np.asarray(
            [self._false_negative_realized_performance(chunk.data) for chunk in reference_chunks]
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

    def _true_positive_realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn("Too many missing values, cannot calculate true_positives. " "Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        _, _, _, tp = confusion_matrix(
            y_true, y_pred, labels=self._labels, normalize=self.normalize_confusion_matrix
        ).ravel()
        return tp

    def _true_negative_realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn("Too many missing values, cannot calculate true_negatives. " "Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        tn, _, _, _ = confusion_matrix(
            y_true, y_pred, labels=self._labels, normalize=self.normalize_confusion_matrix
        ).ravel()
        return tn

    def _false_positive_realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn("Too many missing values, cannot calculate false_positives. " "Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        _, fp, _, _ = confusion_matrix(
            y_true, y_pred, labels=self._labels, normalize=self.normalize_confusion_matrix
        ).ravel()
        return fp

    def _false_negative_realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            warnings.warn("Too many missing values, cannot calculate false_negatives. " "Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        _, _, fn, _ = confusion_matrix(
            y_true, y_pred, labels=self._labels, normalize=self.normalize_confusion_matrix
        ).ravel()
        return fn

    def get_true_positive_estimate(self, chunk_data: pd.DataFrame) -> float:
        """Estimates the true positive rate for a given chunk of data.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        normalized_est_tp_ratio : float
            Estimated true positive rate.
        """
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(chunk_data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]

        est_tp_ratio = np.mean(np.where(y_pred == 1, y_pred_proba, 0))
        est_fp_ratio = np.mean(np.where(y_pred == 1, 1 - y_pred_proba, 0))
        est_fn_ratio = np.mean(np.where(y_pred == 0, y_pred_proba, 0))

        if self.normalize_confusion_matrix is None:
            normalized_est_tp_ratio = est_tp_ratio * len(y_pred)

        elif self.normalize_confusion_matrix == 'all':
            normalized_est_tp_ratio = est_tp_ratio

        elif self.normalize_confusion_matrix == 'true':
            normalizer = 1 / (est_tp_ratio + est_fn_ratio)
            normalized_est_tp_ratio = est_tp_ratio * normalizer

        elif self.normalize_confusion_matrix == 'pred':
            normalizer = 1 / (est_tp_ratio + est_fp_ratio)
            normalized_est_tp_ratio = est_tp_ratio * normalizer

        else:
            raise InvalidArgumentsException(
                f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
                f"but got '{self.normalize_confusion_matrix}"
            )

        return normalized_est_tp_ratio

    def get_true_negative_estimate(self, chunk_data: pd.DataFrame) -> float:
        """Estimates the true negative rate for a given chunk of data.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        normalized_est_tn_ratio : float
            Estimated true negative rate.
        """
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(chunk_data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]

        est_tn_ratio = np.mean(np.where(y_pred == 0, 1 - y_pred_proba, 0))
        est_fp_ratio = np.mean(np.where(y_pred == 1, 1 - y_pred_proba, 0))
        est_fn_ratio = np.mean(np.where(y_pred == 0, y_pred_proba, 0))

        if self.normalize_confusion_matrix is None:
            normalized_est_tn_ratio = est_tn_ratio * len(y_pred)

        elif self.normalize_confusion_matrix == 'all':
            normalized_est_tn_ratio = est_tn_ratio

        elif self.normalize_confusion_matrix == 'true':
            normalizer = 1 / (est_tn_ratio + est_fp_ratio)
            normalized_est_tn_ratio = est_tn_ratio * normalizer

        elif self.normalize_confusion_matrix == 'pred':
            normalizer = 1 / (est_tn_ratio + est_fn_ratio)
            normalized_est_tn_ratio = est_tn_ratio * normalizer

        else:
            raise InvalidArgumentsException(
                f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
                f"but got '{self.normalize_confusion_matrix}"
            )

        return normalized_est_tn_ratio

    def get_false_positive_estimate(self, chunk_data: pd.DataFrame) -> float:
        """Estimates the false positive rate for a given chunk of data.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        normalized_est_fp_ratio : float
            Estimated false positive rate.
        """
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(chunk_data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]

        est_tp_ratio = np.mean(np.where(y_pred == 1, y_pred_proba, 0))
        est_fp_ratio = np.mean(np.where(y_pred == 1, 1 - y_pred_proba, 0))
        est_tn_ratio = np.mean(np.where(y_pred == 0, 1 - y_pred_proba, 0))

        if self.normalize_confusion_matrix is None:
            normalized_est_fp_ratio = est_fp_ratio * len(y_pred)

        elif self.normalize_confusion_matrix == 'all':
            normalized_est_fp_ratio = est_fp_ratio

        elif self.normalize_confusion_matrix == 'true':
            normalizer = 1 / (est_tn_ratio + est_fp_ratio)
            normalized_est_fp_ratio = est_fp_ratio * normalizer

        elif self.normalize_confusion_matrix == 'pred':
            normalizer = 1 / (est_tp_ratio + est_fp_ratio)
            normalized_est_fp_ratio = est_fp_ratio * normalizer

        else:
            raise InvalidArgumentsException(
                f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
                f"but got '{self.normalize_confusion_matrix}"
            )

        return normalized_est_fp_ratio

    def get_false_negative_estimate(self, chunk_data: pd.DataFrame) -> float:
        """Estimates the false negative rate for a given chunk of data.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        normalized_est_fn_ratio : float
            Estimated false negative rate.
        """
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(chunk_data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]

        est_tp_ratio = np.mean(np.where(y_pred == 1, y_pred_proba, 0))
        est_fn_ratio = np.mean(np.where(y_pred == 0, y_pred_proba, 0))
        est_tn_ratio = np.mean(np.where(y_pred == 0, 1 - y_pred_proba, 0))

        if self.normalize_confusion_matrix is None:
            normalized_est_fn_ratio = est_fn_ratio * len(y_pred)

        elif self.normalize_confusion_matrix == 'all':
            normalized_est_fn_ratio = est_fn_ratio

        elif self.normalize_confusion_matrix == 'true':
            normalizer = 1 / (est_tp_ratio + est_fn_ratio)
            normalized_est_fn_ratio = est_fn_ratio * normalizer

        elif self.normalize_confusion_matrix == 'pred':
            normalizer = 1 / (est_tn_ratio + est_fn_ratio)
            normalized_est_fn_ratio = est_fn_ratio * normalizer

        else:
            raise InvalidArgumentsException(
                f"'normalize_confusion_matrix' should be None, 'true', 'pred' or 'all' "
                f"but got '{self.normalize_confusion_matrix}"
            )

        return normalized_est_fn_ratio

    def get_true_pos_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the true positives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        true_pos_info : Dict
            A dictionary of true positive's information and its value pairs.
        """
        true_pos_info: Dict[str, Any] = {}

        # we check for nans inside
        estimated_true_positives = self.get_true_positive_estimate(chunk_data)
        realized_true_positives = self._true_positive_realized_performance(chunk_data)
        # we do sampling error nan checks here because we don't have dedicated sampling error function
        # TODO: Refactor similarly to multiclass so code can be re-used.
        # filter nans here - for realized performance both columns are expected
        chunk_data, empty = common_nan_removal(
            chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred]
        )
        if empty:
            warnings.warn("Too many missing values, cannot calculate true positive sampling error. " "Returning NaN.")
            sampling_error_true_positives = np.nan
        else:
            sampling_error_true_positives = bse.true_positive_sampling_error(
                self._true_positive_sampling_error_components, chunk_data
            )
        # TODO: NaN removal is duplicated to an extent. Upon refactor consider if we can do it only once

        true_pos_info['estimated_true_positive'] = estimated_true_positives
        true_pos_info['sampling_error_true_positive'] = sampling_error_true_positives
        true_pos_info['realized_true_positive'] = realized_true_positives

        true_pos_info['upper_confidence_boundary_true_positive'] = np.minimum(
            np.inf if self.upper_threshold_value_limit is None else self.upper_threshold_value_limit,
            estimated_true_positives + SAMPLING_ERROR_RANGE * sampling_error_true_positives,
        )

        true_pos_info['lower_confidence_boundary_true_positive'] = np.maximum(
            -np.inf if self.lower_threshold_value_limit is None else self.lower_threshold_value_limit,
            estimated_true_positives - SAMPLING_ERROR_RANGE * sampling_error_true_positives,
        )

        true_pos_info['upper_threshold_true_positive'] = self.true_positive_upper_threshold
        true_pos_info['lower_threshold_true_positive'] = self.true_positive_lower_threshold

        true_pos_info['alert_true_positive'] = (
            self.true_positive_upper_threshold is not None
            and estimated_true_positives > self.true_positive_upper_threshold
        ) or (
            self.true_positive_lower_threshold is not None
            and estimated_true_positives < self.true_positive_lower_threshold
        )

        return true_pos_info

    def get_true_neg_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the true negatives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        true_neg_info : Dict
            A dictionary of true negative's information and its value pairs.
        """
        true_neg_info: Dict[str, Any] = {}

        # we check for nans inside
        estimated_true_negatives = self.get_true_negative_estimate(chunk_data)
        realized_true_negatives = self._true_negative_realized_performance(chunk_data)
        # we do sampling error nan checks here because we don't have dedicated sampling error function
        # TODO: Refactor similarly to multiclass so code can be re-used.
        # filter nans here - for realized performance both columns are expected
        chunk_data, empty = common_nan_removal(
            chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred]
        )
        if empty:
            warnings.warn("Too many missing values, cannot calculate true positive sampling error. " "Returning NaN.")
            sampling_error_true_negatives = np.nan
        else:
            sampling_error_true_negatives = bse.true_negative_sampling_error(
                self._true_negative_sampling_error_components, chunk_data
            )
        # TODO: NaN removal is duplicated to an extent. Upon refactor consider if we can do it only once

        true_neg_info['estimated_true_negative'] = estimated_true_negatives
        true_neg_info['sampling_error_true_negative'] = sampling_error_true_negatives
        true_neg_info['realized_true_negative'] = realized_true_negatives

        true_neg_info['upper_confidence_boundary_true_negative'] = np.minimum(
            np.inf if self.upper_threshold_value_limit is None else self.upper_threshold_value_limit,
            estimated_true_negatives + SAMPLING_ERROR_RANGE * sampling_error_true_negatives,
        )

        true_neg_info['lower_confidence_boundary_true_negative'] = np.maximum(
            -np.inf if self.lower_threshold_value_limit is None else self.lower_threshold_value_limit,
            estimated_true_negatives - SAMPLING_ERROR_RANGE * sampling_error_true_negatives,
        )

        true_neg_info['upper_threshold_true_negative'] = self.true_negative_upper_threshold
        true_neg_info['lower_threshold_true_negative'] = self.true_negative_lower_threshold

        true_neg_info['alert_true_negative'] = (
            self.true_negative_upper_threshold is not None
            and estimated_true_negatives > self.true_negative_upper_threshold
        ) or (
            self.true_negative_lower_threshold is not None
            and estimated_true_negatives < self.true_negative_lower_threshold
        )

        return true_neg_info

    def get_false_pos_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the false positives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        false_pos_info : Dict
            A dictionary of false positive's information and its value pairs.
        """
        false_pos_info: Dict[str, Any] = {}

        # we check for nans inside
        estimated_false_positives = self.get_false_positive_estimate(chunk_data)
        realized_false_positives = self._false_positive_realized_performance(chunk_data)
        # we do sampling error nan checks here because we don't have dedicated sampling error function
        # TODO: Refactor similarly to multiclass so code can be re-used.
        # filter nans here - for realized performance both columns are expected
        chunk_data, empty = common_nan_removal(
            chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred]
        )
        if empty:
            warnings.warn("Too many missing values, cannot calculate true positive sampling error. " "Returning NaN.")
            sampling_error_false_positives = np.nan
        else:
            sampling_error_false_positives = bse.false_positive_sampling_error(
                self._false_positive_sampling_error_components, chunk_data
            )
        # TODO: NaN removal is duplicated to an extent. Upon refactor consider if we can do it only once

        false_pos_info['estimated_false_positive'] = estimated_false_positives
        false_pos_info['sampling_error_false_positive'] = sampling_error_false_positives
        false_pos_info['realized_false_positive'] = realized_false_positives

        false_pos_info['upper_confidence_boundary_false_positive'] = np.minimum(
            np.inf if self.upper_threshold_value_limit is None else self.upper_threshold_value_limit,
            estimated_false_positives + SAMPLING_ERROR_RANGE * sampling_error_false_positives,
        )

        false_pos_info['lower_confidence_boundary_false_positive'] = np.maximum(
            -np.inf if self.lower_threshold_value_limit is None else self.lower_threshold_value_limit,
            estimated_false_positives - SAMPLING_ERROR_RANGE * sampling_error_false_positives,
        )

        false_pos_info['upper_threshold_false_positive'] = self.false_positive_upper_threshold
        false_pos_info['lower_threshold_false_positive'] = self.false_positive_lower_threshold

        false_pos_info['alert_false_positive'] = (
            self.false_positive_upper_threshold is not None
            and estimated_false_positives > self.false_positive_upper_threshold
        ) or (
            self.false_positive_lower_threshold is not None
            and estimated_false_positives < self.false_positive_lower_threshold
        )

        return false_pos_info

    def get_false_neg_info(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing infomation about the false negatives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
        false_neg_info : Dict
            A dictionary of false negative's information and its value pairs.
        """
        false_neg_info: Dict[str, Any] = {}

        # we check for nans inside
        estimated_false_negatives = self.get_false_negative_estimate(chunk_data)
        realized_false_negatives = self._false_negative_realized_performance(chunk_data)
        # we do sampling error nan checks here because we don't have dedicated sampling error function
        # TODO: Refactor similarly to multiclass so code can be re-used.
        # filter nans here - for realized performance both columns are expected
        chunk_data, empty = common_nan_removal(
            chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred]
        )
        if empty:
            warnings.warn("Too many missing values, cannot calculate true positive sampling error. " "Returning NaN.")
            sampling_error_false_negatives = np.nan
        else:
            sampling_error_false_negatives = bse.false_negative_sampling_error(
                self._false_negative_sampling_error_components, chunk_data
            )
        # TODO: NaN removal is duplicated to an extent. Upon refactor consider if we can do it only once

        false_neg_info['estimated_false_negative'] = estimated_false_negatives
        false_neg_info['sampling_error_false_negative'] = sampling_error_false_negatives
        false_neg_info['realized_false_negative'] = realized_false_negatives

        false_neg_info['upper_confidence_boundary_false_negative'] = np.minimum(
            np.inf if self.upper_threshold_value_limit is None else self.upper_threshold_value_limit,
            estimated_false_negatives + SAMPLING_ERROR_RANGE * sampling_error_false_negatives,
        )

        false_neg_info['lower_confidence_boundary_false_negative'] = np.maximum(
            -np.inf if self.lower_threshold_value_limit is None else self.lower_threshold_value_limit,
            estimated_false_negatives - SAMPLING_ERROR_RANGE * sampling_error_false_negatives,
        )

        false_neg_info['upper_threshold_false_negative'] = self.false_negative_upper_threshold
        false_neg_info['lower_threshold_false_negative'] = self.false_negative_lower_threshold

        false_neg_info['alert_false_negative'] = (
            self.false_negative_upper_threshold is not None
            and estimated_false_negatives > self.false_negative_upper_threshold
        ) or (
            self.false_negative_lower_threshold is not None
            and estimated_false_negatives < self.false_negative_lower_threshold
        )

        return false_neg_info

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing the performance metrics for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
            chunk_record : Dict
                A dictionary of perfomance metric, value pairs.
        """
        chunk_record = {}

        true_pos_info = self.get_true_pos_info(chunk_data)
        chunk_record.update(true_pos_info)

        true_neg_info = self.get_true_neg_info(chunk_data)
        chunk_record.update(true_neg_info)

        false_pos_info = self.get_false_pos_info(chunk_data)
        chunk_record.update(false_pos_info)

        false_neg_info = self.get_false_neg_info(chunk_data)
        chunk_record.update(false_neg_info)

        return chunk_record

    def _estimate(self, data: pd.DataFrame):
        pass

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return 0.0

    def _realized_performance(self, data: pd.DataFrame) -> float:
        return 0.0


@MetricFactory.register('business_value', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationBusinessValue(Metric):
    """CBPE binary classification business value Metric Class."""

    y_pred_proba: str

    def __init__(
        self,
        y_pred_proba: str,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        business_value_matrix: Union[List, np.ndarray],
        normalize_business_value: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE binary classification business value Metric Class."""
        super().__init__(
            name='business_value',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Business Value', 'business_value')],
        )

        if business_value_matrix is None:
            raise ValueError("business_value_matrix must be provided for 'business_value' metric")

        if not (isinstance(business_value_matrix, np.ndarray) or isinstance(business_value_matrix, list)):
            raise ValueError(
                f"business_value_matrix must be a numpy array or a list, but got {type(business_value_matrix)}"
            )

        if isinstance(business_value_matrix, list):
            business_value_matrix = np.array(business_value_matrix)

        if business_value_matrix.shape != (2, 2):
            raise ValueError(
                f"business_value_matrix must have shape (2,2), but got matrix of shape {business_value_matrix.shape}"
            )

        self.business_value_matrix = business_value_matrix
        self.normalize_business_value: Optional[str] = normalize_business_value

        # Set labels expected in y_true/y_pred. Currently hard-coded to 0, 1 for binary classification
        self._labels = [0, 1]

        # self.lower_threshold: Optional[float] = 0
        # self.upper_threshold: Optional[float] = 1

    def _fit(self, reference_data: pd.DataFrame):
        # filter nans
        data = reference_data[[self.y_true, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        if empty:
            self._logger.debug(f"Not enough data to compute fit {self.display_name}.")
            warnings.warn(f"Not enough data to compute fit {self.display_name}.")
            self._sampling_error_components = np.nan, self.normalize_business_value
        else:
            self._sampling_error_components = bse.business_value_sampling_error_components(
                y_true_reference=y_true,
                y_pred_reference=y_pred,
                business_value_matrix=self.business_value_matrix,
                normalize_business_value=self.normalize_business_value,
            )

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred, self.y_true], list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data[[self.y_pred, self.y_true]], [self.y_pred, self.y_true])
        if empty:
            self._logger.debug(f"Not enough data to compute realized {self.display_name}.")
            warnings.warn(f"Not enough data to compute realized {self.display_name}.")
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        tp_value = self.business_value_matrix[1, 1]
        tn_value = self.business_value_matrix[0, 0]
        fp_value = self.business_value_matrix[0, 1]
        fn_value = self.business_value_matrix[1, 0]
        bv_array = np.array([[tn_value, fp_value], [fn_value, tp_value]])

        cm = confusion_matrix(y_true, y_pred, labels=self._labels)
        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)
        return (bv_array * cm).sum()

    def _estimate(self, chunk_data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_pred_proba, self.y_pred], list(chunk_data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(chunk_data[[self.y_pred_proba, self.y_pred]], [self.y_pred_proba, self.y_pred])
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_pred = data[self.y_pred]
        y_pred_proba = data[self.y_pred_proba]

        business_value_normalization = self.normalize_business_value
        business_value_matrix = self.business_value_matrix

        return estimate_business_value(y_pred, y_pred_proba, business_value_normalization, business_value_matrix)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        data = data[[self.y_pred_proba, self.y_pred]]
        data, empty = common_nan_removal(data, [self.y_pred_proba, self.y_pred])
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " "Returning NaN."
            )
            return np.nan
        else:
            return bse.business_value_sampling_error(self._sampling_error_components, data)


def estimate_business_value(
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    normalize_business_value: Optional[str],
    business_value_matrix: np.ndarray,
) -> float:
    """Estimates the Business Value metric.

    Parameters
    ----------
    y_pred: np.ndarray
        Predicted class labels of the sample
    y_pred_proba: np.ndarray
        Probability estimates of the sample for each class in the model.
    normalize_business_value: str, default=None
        Determines how the business value will be normalized. Allowed values are None and 'per_prediction'.

        - None - the business value will not be normalized and the value returned will be the total value per chunk.
        - 'per_prediction' - the value will be normalized by the number of predictions in the chunk.

    business_value_matrix: np.ndarray
        A 2x2 matrix that specifies the value of each cell in the confusion matrix.
        The format of the business value matrix must be specified as [[value_of_TN, value_of_FP], \
        [value_of_FN, value_of_TP]].

    Returns
    -------
    business_value: float
        Estimated Business Value score.
    """
    est_tn_ratio = np.mean(np.where(y_pred == 0, 1 - y_pred_proba, 0))
    est_tp_ratio = np.mean(np.where(y_pred == 1, y_pred_proba, 0))
    est_fp_ratio = np.mean(np.where(y_pred == 1, 1 - y_pred_proba, 0))
    est_fn_ratio = np.mean(np.where(y_pred == 0, y_pred_proba, 0))
    cm = np.array([[est_tn_ratio, est_fp_ratio], [est_fn_ratio, est_tp_ratio]]) * len(y_pred)
    if normalize_business_value == 'per_prediction':
        with np.errstate(all="ignore"):
            cm = cm / cm.sum(axis=0, keepdims=True)
        cm = np.nan_to_num(cm)

    tp_value = business_value_matrix[1, 1]
    tn_value = business_value_matrix[0, 0]
    fp_value = business_value_matrix[0, 1]
    fn_value = business_value_matrix[1, 0]
    bv_array = np.array([[tn_value, fp_value], [fn_value, tp_value]])

    return (bv_array * cm).sum()


def _get_binarized_multiclass_predictions(data: pd.DataFrame, y_pred: str, y_pred_proba: ModelOutputsType):
    if not isinstance(y_pred_proba, dict):
        raise CalculatorException(
            "multiclass model outputs should be of type Dict[str, str].\n"
            f"'{y_pred_proba}' is of type '{type(y_pred_proba)}'"
        )

    classes = sorted(y_pred_proba.keys())
    y_preds = list(label_binarize(data[y_pred], classes=classes).T)

    y_pred_probas = [data[y_pred_proba[clazz]] for clazz in classes]
    return y_preds, y_pred_probas, classes


def _get_multiclass_uncalibrated_predictions(data: pd.DataFrame, y_pred: str, y_pred_proba: ModelOutputsType):
    if not isinstance(y_pred_proba, dict):
        raise CalculatorException(
            "multiclass model outputs should be of type Dict[str, str].\n"
            f"'{y_pred_proba}' is of type '{type(y_pred_proba)}'"
        )

    labels, class_probability_columns = [], []
    for label in sorted(y_pred_proba.keys()):
        labels.append(label)
        class_probability_columns.append(f'uncalibrated_{y_pred_proba[label]}')
    return data[y_pred], data[class_probability_columns], labels


@MetricFactory.register('roc_auc', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAUROC(Metric):
    """CBPE multiclass classification AUROC Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification AUROC Metric Class."""
        super().__init__(
            name='roc_auc',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('ROC AUC', 'roc_auc')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )
        self.y_pred_proba: Dict[str, str]
        self.classes: List[str] = [""]
        self.class_probability_columns: List[str]
        self.class_uncalibrated_y_pred_proba_columns: List[str]
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        self.classes = class_labels(self.y_pred_proba)
        self.class_probability_columns = [self.y_pred_proba[clazz] for clazz in self.classes]
        self.class_uncalibrated_y_pred_proba_columns = ['uncalibrated_' + el for el in self.class_probability_columns]
        _list_missing([self.y_true] + self.class_uncalibrated_y_pred_proba_columns, list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true] + self.class_uncalibrated_y_pred_proba_columns],
            [self.y_true] + self.class_uncalibrated_y_pred_proba_columns,
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clasz in self.classes]
        else:
            # test if reference data are represented correctly
            observed_classes = set(reference_data[self.y_true].unique())
            if not observed_classes == set(self.classes):
                self._logger.error(
                    "The specified classification classes are not the same as the classes observed in the reference"
                    "targets."
                )
                raise InvalidArgumentsException(
                    "y_pred_proba class and class probabilities dictionary does not match reference data."
                )
            # sampling error
            binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=self.classes).T)
            y_pred_proba = [reference_data['uncalibrated_' + self.y_pred_proba[clazz]].T for clazz in self.classes]
            self._sampling_error_components = mse.auroc_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_proba_reference=y_pred_proba
            )

    def _estimate(self, data: pd.DataFrame):
        needed_columns = self.class_probability_columns + self.class_uncalibrated_y_pred_proba_columns
        try:
            _list_missing(needed_columns, list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        _, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        _, y_pred_probas_uncalibrated, _ = _get_multiclass_uncalibrated_predictions(
            data, self.y_pred, self.y_pred_proba
        )
        ovr_estimates = []
        for el in range(len(y_pred_probas)):
            ovr_estimates.append(
                estimate_roc_auc(
                    # sorting according to classes is/should_be the same across
                    # _get_binarized_multiclass_predictions and _get_multiclass_uncalibrated_predictions
                    y_pred_probas[el],
                    y_pred_probas_uncalibrated.iloc[:, el],
                )
            )
        multiclass_roc_auc = np.mean(ovr_estimates)
        return multiclass_roc_auc

    def _sampling_error(self, data: pd.DataFrame) -> float:
        needed_columns = self.class_probability_columns + self.class_uncalibrated_y_pred_proba_columns
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. Returning NaN."
            )
            return np.nan
        else:
            return mse.auroc_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true] + self.class_uncalibrated_y_pred_proba_columns, data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true] + self.class_uncalibrated_y_pred_proba_columns)
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if set(y_true.unique()) != set(self.classes):
            _message = (
                f"'{self.y_true}' does not contain all reported classes, cannot calculate {self.display_name}. "
                "Returning NaN."
            )
            warnings.warn(_message)
            self._logger.warning(_message)
            return np.nan

        _, y_pred_probas, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return roc_auc_score(y_true, y_pred_probas, multi_class='ovr', average='macro', labels=labels)


@MetricFactory.register('f1', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationF1(Metric):
    """CBPE multiclass classification f1 Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification f1 Metric Class."""
        super().__init__(
            name='f1',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('F1', 'f1')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error:
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        classes = class_labels(self.y_pred_proba)
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
            label_binarizer = LabelBinarizer()
            binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
            binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
            self._sampling_error_components = mse.f1_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
            )

    def _estimate(self, data: pd.DataFrame):
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        try:
            _list_missing(needed_columns, list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_f1(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return mse.f1_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.display_name}.")
            return np.nan

        if data[self.y_pred].nunique() <= 1:
            warnings.warn("Too few unique values present in 'y_pred', returning NaN as realized F1 score.")
            return np.nan

        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('precision', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationPrecision(Metric):
    """CBPE multiclass classification precision Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification precision Metric Class."""
        super().__init__(
            name='precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Precision', 'precision')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        classes = class_labels(self.y_pred_proba)
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
            label_binarizer = LabelBinarizer()
            binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
            binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
            self._sampling_error_components = mse.precision_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
            )

    def _estimate(self, data: pd.DataFrame):
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        try:
            _list_missing(needed_columns, list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_precision(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return mse.precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.display_name}.")
            return np.nan

        if data[self.y_pred].nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in 'y_pred', returning NaN as realized {self.display_name} score."
            )
            return np.nan

        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)
        return precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('recall', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationRecall(Metric):
    """CBPE multiclass classification recall Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification recall Metric Class."""
        super().__init__(
            name='recall',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Recall', 'recall')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        classes = class_labels(self.y_pred_proba)
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
            label_binarizer = LabelBinarizer()
            binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
            binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
            self._sampling_error_components = mse.recall_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
            )

    def _estimate(self, data: pd.DataFrame):
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        try:
            _list_missing(needed_columns, list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_recall(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)
        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return mse.recall_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.display_name}.")
            return np.nan

        if data[self.y_pred].nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in 'y_pred', returning NaN as realized {self.display_name} score."
            )
            return np.nan

        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('specificity', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationSpecificity(Metric):
    """CBPE multiclass classification specificity Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification specificity Metric Class."""
        super().__init__(
            name='specificity',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Specificity', 'specificity')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        classes = class_labels(self.y_pred_proba)
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in classes]
        else:
            label_binarizer = LabelBinarizer()
            binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
            binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
            self._sampling_error_components = mse.specificity_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
            )

    def _estimate(self, data: pd.DataFrame):
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        try:
            _list_missing(needed_columns, list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_specificity(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return mse.specificity_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.display_name}.")
            return np.nan

        if data[self.y_pred].nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in 'y_pred', returning NaN as realized {self.display_name} score."
            )
            return np.nan

        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        tn_sum = mcm[:, 0, 0]
        fp_sum = mcm[:, 0, 1]
        class_wise_specificity = tn_sum / (tn_sum + fp_sum)
        return np.mean(class_wise_specificity)


@MetricFactory.register('accuracy', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAccuracy(Metric):
    """CBPE multiclass classification accuracy Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification accuracy Metric Class."""
        super().__init__(
            name='accuracy',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Accuracy', 'accuracy')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._sampling_error_components = (np.nan,)
        else:
            label_binarizer = LabelBinarizer()
            binarized_y_true = label_binarizer.fit_transform(reference_data[self.y_true])
            binarized_y_pred = label_binarizer.transform(reference_data[self.y_pred])

            self._sampling_error_components = mse.accuracy_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
            )

    def _estimate(self, data: pd.DataFrame):
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        try:
            _list_missing(needed_columns, list(data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        y_preds_array = np.asarray(y_preds).T
        y_pred_probas_array = np.asarray(y_pred_probas).T
        probability_of_predicted = np.max(y_preds_array * y_pred_probas_array, axis=1)
        return np.mean(probability_of_predicted)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return mse.accuracy_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.display_name}.")
            return np.nan

        if data[self.y_pred].nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in 'y_pred', returning NaN as realized {self.display_name} score."
            )
            return np.nan

        y_pred, _, _ = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)
        return accuracy_score(y_true, y_pred)


@MetricFactory.register('confusion_matrix', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationConfusionMatrix(Metric):
    """CBPE multiclass classification confusion matrix Metric Class."""

    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_pred_proba: Dict[str, str],
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        normalize_confusion_matrix: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification confusion matrix Metric Class."""
        if isinstance(y_pred_proba, str):
            raise ValueError(
                "y_pred_proba must be a dictionary with class labels as keys and pred_proba column names as values"
            )

        self.classes: List[str] = sorted(list(y_pred_proba.keys()))

        super().__init__(
            name='confusion_matrix',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=self._get_components(self.classes),
            lower_threshold_value_limit=0,
        )

        self.normalize_confusion_matrix: Optional[str] = normalize_confusion_matrix
        if self.normalize_confusion_matrix is None:
            # overwrite default upper bound setting.
            self.upper_threshold_value_limit = None
        else:
            self.upper_threshold_value_limit = 1

    def _get_components(self, classes: List[str]) -> List[Tuple[str, str]]:
        components = []

        for true_class in classes:
            for pred_class in classes:
                components.append(
                    (
                        f"true class: '{true_class}', predicted class: '{pred_class}'",
                        f'true_{true_class}_pred_{pred_class}',
                    )
                )

        return components

    def fit(self, reference_data: pd.DataFrame):  # override the superclass fit method
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.
        """
        # Calculate alert thresholds
        reference_chunks = self.chunker.split(
            reference_data,
        )

        self.alert_thresholds = self._multiclass_confusion_matrix_alert_thresholds(reference_chunks)

        # Delegate to confusion matrix subclass
        self._fit(reference_data)  # could probably put _fit functionality here since overide fit method

        return

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], reference_data)
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred]
        )
        if empty:
            self._confusion_matrix_sampling_error_components = (
                np.full((len(self.classes), len(self.classes)), np.nan),
                0,
            )
        else:
            # sampling error
            self._confusion_matrix_sampling_error_components = (
                mse.multiclass_confusion_matrix_sampling_error_components(  # noqa: E501
                    y_true_reference=reference_data[self.y_true],
                    y_pred_reference=reference_data[self.y_pred],
                    normalize_confusion_matrix=self.normalize_confusion_matrix,
                )
            )

    def _multiclass_confusion_matrix_alert_thresholds(
        self, reference_chunks: List[Chunk]
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        realized_chunk_performance = np.asarray(
            [self._multi_class_confusion_matrix_realized_performance(chunk.data) for chunk in reference_chunks]
        )

        alert_thresholds = {}

        num_classes = len(self.classes)

        for i in range(num_classes):
            for j in range(num_classes):
                lower_threshold_value, upper_threshold_value = calculate_threshold_values(
                    threshold=self.threshold,
                    data=realized_chunk_performance[:, i, j],
                    lower_threshold_value_limit=self.lower_threshold_value_limit,
                    upper_threshold_value_limit=self.upper_threshold_value_limit,
                )
                alert_thresholds[f'true_{self.classes[i]}_pred_{self.classes[j]}'] = (
                    lower_threshold_value,
                    upper_threshold_value,
                )

        return alert_thresholds

    def _multi_class_confusion_matrix_realized_performance(self, data: pd.DataFrame) -> Union[np.ndarray, float]:
        # Create appropriate nan array to return in case of error
        num_classes = len(self.classes)
        nan_array = np.full(shape=(num_classes, num_classes), fill_value=np.nan)
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return nan_array
            else:
                raise ex

        data, empty = common_nan_removal(data, [self.y_true, self.y_pred])
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return nan_array

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.display_name}.")
            return nan_array
        if data[self.y_pred].nunique() <= 1:
            warnings.warn(
                f"Too few unique values present in 'y_pred', returning NaN as realized {self.display_name} score."
            )
            return nan_array

        cm = confusion_matrix(
            data[self.y_true], data[self.y_pred], labels=self.classes, normalize=self.normalize_confusion_matrix
        )
        return cm

    def _get_multiclass_confusion_matrix_estimate(self, chunk_data: pd.DataFrame) -> np.ndarray:
        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        try:
            _list_missing(needed_columns, list(chunk_data.columns))
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.debug(str(ex))
                return np.full((len(self.classes), len(self.classes)), np.nan)
            else:
                raise ex

        chunk_data, empty = common_nan_removal(chunk_data, needed_columns)
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.full((len(self.classes), len(self.classes)), np.nan)

        y_pred_proba = {key: chunk_data[value] for key, value in self.y_pred_proba.items()}

        y_pred = chunk_data[self.y_pred]

        num_classes = len(self.classes)

        est_confusion_matrix = np.zeros((num_classes, num_classes))

        for i in range(num_classes):
            for j in range(num_classes):
                est_confusion_matrix[i, j] = np.mean(
                    np.where(
                        (y_pred == self.classes[j]),
                        y_pred_proba[self.classes[i]],
                        0,
                    )
                )

        if self.normalize_confusion_matrix is None:
            normalized_est_confusion_matrix = est_confusion_matrix * len(y_pred)

        elif self.normalize_confusion_matrix == 'true':
            normalized_est_confusion_matrix = est_confusion_matrix / np.sum(est_confusion_matrix, axis=1)[:, None]

        elif self.normalize_confusion_matrix == 'pred':
            normalized_est_confusion_matrix = est_confusion_matrix / np.sum(est_confusion_matrix, axis=0)[None, :]
        elif self.normalize_confusion_matrix == 'all':
            normalized_est_confusion_matrix = est_confusion_matrix / np.sum(est_confusion_matrix)
        else:
            raise ValueError(
                f'normalize_confusion_matrix should be one of None, "true", \
                    "pred", or "all", but got {self.normalize_confusion_matrix}'
            )

        return normalized_est_confusion_matrix

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
        """Returns a dictionary containing the performance metrics for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.

        Returns
        -------
            chunk_record : Dict
                A dictionary of perfomance metric, value pairs.
        """
        chunk_record = {}

        estimated_cm = self._get_multiclass_confusion_matrix_estimate(chunk_data)
        realized_cm = self._multi_class_confusion_matrix_realized_performance(chunk_data)

        class_y_pred_proba_columns = model_output_column_names(self.y_pred_proba)
        needed_columns = class_y_pred_proba_columns + [self.y_pred]
        _list_missing(needed_columns, chunk_data)
        # filter nans here
        chunk_data, empty = common_nan_removal(chunk_data[needed_columns], needed_columns)
        if empty:
            sampling_error = np.full((len(self.classes), len(self.classes)), np.nan)
        else:
            sampling_error = mse.multiclass_confusion_matrix_sampling_error(
                self._confusion_matrix_sampling_error_components,
                chunk_data,
            )

        for true_class in self.classes:
            for pred_class in self.classes:
                chunk_record[f'estimated_true_{true_class}_pred_{pred_class}'] = estimated_cm[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ]

                chunk_record[f'sampling_error_true_{true_class}_pred_{pred_class}'] = sampling_error[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ]

                # check if realized_cm is nan
                if isinstance(realized_cm, np.ndarray):
                    chunk_record[f'realized_true_{true_class}_pred_{pred_class}'] = realized_cm[
                        self.classes.index(true_class), self.classes.index(pred_class)
                    ]
                else:
                    chunk_record[f'realized_true_{true_class}_pred_{pred_class}'] = realized_cm

                upper_confidence_boundary = (
                    estimated_cm[self.classes.index(true_class), self.classes.index(pred_class)]
                    + SAMPLING_ERROR_RANGE
                    * sampling_error[self.classes.index(true_class), self.classes.index(pred_class)]
                )
                chunk_record[f'upper_confidence_boundary_true_{true_class}_pred_{pred_class}'] = min(
                    np.inf if self.upper_threshold_value_limit is None else self.upper_threshold_value_limit,
                    upper_confidence_boundary,
                )

                lower_confidence_boundary = (
                    estimated_cm[self.classes.index(true_class), self.classes.index(pred_class)]
                    - SAMPLING_ERROR_RANGE
                    * sampling_error[self.classes.index(true_class), self.classes.index(pred_class)]
                )
                chunk_record[f'lower_confidence_boundary_true_{true_class}_pred_{pred_class}'] = max(
                    -np.inf if self.lower_threshold_value_limit is None else self.lower_threshold_value_limit,
                    lower_confidence_boundary,
                )

                chunk_record[f'upper_threshold_true_{true_class}_pred_{pred_class}'] = self.alert_thresholds[
                    f'true_{true_class}_pred_{pred_class}'
                ][1]
                chunk_record[f'lower_threshold_true_{true_class}_pred_{pred_class}'] = self.alert_thresholds[
                    f'true_{true_class}_pred_{pred_class}'
                ][0]

                # do alerts
                chunk_record[f'alert_true_{true_class}_pred_{pred_class}'] = (
                    self.alert_thresholds is not None
                    and (
                        estimated_cm[self.classes.index(true_class), self.classes.index(pred_class)]
                        > self.alert_thresholds[f'true_{true_class}_pred_{pred_class}'][1]
                    )
                    or (
                        self.alert_thresholds is not None
                        and (
                            estimated_cm[self.classes.index(true_class), self.classes.index(pred_class)]
                            < self.alert_thresholds[f'true_{true_class}_pred_{pred_class}'][0]
                        )
                    )
                )

        return chunk_record

    def _estimate(self, data: pd.DataFrame):
        pass

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return 0.0

    def _realized_performance(self, data: pd.DataFrame) -> float:
        return 0.0


@MetricFactory.register('average_precision', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAP(Metric):
    """CBPE multiclass classification AP Metric Class."""

    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification AP Metric Class."""
        super().__init__(
            name='average_precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Average Precision', 'average_precision')],
            lower_threshold_value_limit=0,
            upper_threshold_value_limit=1,
        )
        # FIXME: Should we check the y_pred_proba argument here to ensure it's a dict?
        self.y_pred_proba: Dict[str, str]

        # sampling error
        self._sampling_error_components: List[Tuple] = []

        # classes and class probability columns
        self.classes: List[str]
        self.class_probability_columns: List[str]
        self.class_uncalibrated_y_pred_proba_columns: List[str]

    def _fit(self, reference_data: pd.DataFrame):
        # set up sorted classes and prob_column_names to use across metric class
        self.classes = class_labels(self.y_pred_proba)
        self.class_probability_columns = [self.y_pred_proba[clazz] for clazz in self.classes]
        self.class_uncalibrated_y_pred_proba_columns = ['uncalibrated_' + el for el in self.class_probability_columns]

        _list_missing([self.y_true] + self.class_uncalibrated_y_pred_proba_columns, list(reference_data.columns))
        # filter nans here
        reference_data, empty = common_nan_removal(
            reference_data[[self.y_true] + self.class_uncalibrated_y_pred_proba_columns],
            [self.y_true] + self.class_uncalibrated_y_pred_proba_columns,
        )
        if empty:
            self._sampling_error_components = [(np.nan, 0) for clazz in self.classes]
        else:
            # sampling error
            binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=self.classes).T)
            y_pred_proba = [reference_data['uncalibrated_' + self.y_pred_proba[clazz]].T for clazz in self.classes]
            self._sampling_error_components = mse.average_precision_sampling_error_components(
                y_true_reference=binarized_y_true, y_pred_proba_reference=y_pred_proba
            )

    def _estimate(self, data: pd.DataFrame):
        needed_columns = self.class_probability_columns + self.class_uncalibrated_y_pred_proba_columns
        try:
            data, empty = common_nan_removal(data, needed_columns)
        except InvalidArgumentsException as ex:
            if "not all present in provided data columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex
        if empty:
            self._logger.debug(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        _, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        _, y_pred_probas_uncalibrated, _ = _get_multiclass_uncalibrated_predictions(
            data, self.y_pred, self.y_pred_proba
        )
        ovr_estimates = []
        for el in range(len(y_pred_probas)):
            ovr_estimates.append(
                estimate_ap(
                    # sorting according to classes is/should_be the same across
                    # _get_binarized_multiclass_predictions and _get_multiclass_uncalibrated_predictions
                    y_pred_probas[el],
                    y_pred_probas_uncalibrated.iloc[:, el],
                )
            )
        multiclass_ap = np.mean(ovr_estimates)
        return multiclass_ap

    def _sampling_error(self, data: pd.DataFrame) -> float:
        needed_columns = self.class_probability_columns + self.class_uncalibrated_y_pred_proba_columns
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            warnings.warn(
                f"Too many missing values, cannot calculate {self.display_name} sampling error. " f"Returning NaN."
            )
            return np.nan
        else:
            return mse.average_precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            data, empty = common_nan_removal(data, [self.y_true] + self.class_uncalibrated_y_pred_proba_columns)
        except InvalidArgumentsException as ex:
            if "not all present in provided data columns" in str(ex):
                self._logger.debug(str(ex))
                return np.nan
            else:
                raise ex
        if empty:
            warnings.warn(f"Too many missing values, cannot calculate {self.display_name}. " f"Returning NaN.")
            return np.nan

        y_true = data[self.y_true]
        if y_true.nunique() <= 1:
            warnings.warn("Too few unique values present in 'y_true', returning NaN as realized AP.")
            return np.nan

        _, y_pred_probas, _ = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
        # average_precision_score always performs OVR averaging
        return average_precision_score(y_true, y_pred_probas, average='macro')


@MetricFactory.register('business_value', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationBusinessValue(Metric):
    """CBPE multiclass classification Business Value Metric Class."""

    y_pred_proba: Dict[str, str]

    def __init__(
        self,
        y_pred_proba: Dict[str, str],
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        business_value_matrix: Union[List, np.ndarray],
        normalize_business_value: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CBPE multiclass classification Business Value Metric Class."""
        super().__init__(
            name='business_value',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Business Value', 'business_value')],
        )

        if business_value_matrix is None:
            raise ValueError("business_value_matrix must be provided for 'business_value' metric")

        if not (isinstance(business_value_matrix, np.ndarray) or isinstance(business_value_matrix, list)):
            raise ValueError(
                f"business_value_matrix must be a numpy array or a list, but got {type(business_value_matrix)}"
            )

        if isinstance(business_value_matrix, list):
            business_value_matrix = np.array(business_value_matrix)
        _rows, _columns = business_value_matrix.shape
        if _rows != _columns:
            raise InvalidArgumentsException(
                f"business_value_matrix is not a square matrix but has shape: {(_rows, _columns)}"
            )

        self.business_value_matrix = business_value_matrix
        self.normalize_business_value: Optional[str] = normalize_business_value

        self.classes: List[str] = class_labels(self.y_pred_proba)
        self.class_probability_columns: List[str]

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))
        data, empty = common_nan_removal(reference_data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            self._sampling_error_components = np.nan, self.normalize_business_value
        else:
            num_classes = len(self.classes)
            if num_classes != self.business_value_matrix.shape[0]:
                raise InvalidArgumentsException(
                    f"business_value_matrix has shape {self.business_value_matrix.shape} "
                    f"but we have {num_classes} classes!"
                )
            self._sampling_error_components = mse.business_value_sampling_error_components(
                y_true_reference=data[self.y_true],
                y_pred_reference=data[self.y_pred],
                business_value_matrix=self.business_value_matrix,
                classes=self.classes,
                normalize_business_value=self.normalize_business_value,
            )

        self.class_probability_columns = [self.y_pred_proba[clazz] for clazz in self.classes]

    def _estimate(self, data: pd.DataFrame):
        needed_columns = self.class_probability_columns + [self.y_pred]
        try:
            data, empty = common_nan_removal(data, needed_columns)
        except InvalidArgumentsException as ex:
            if "not all present in provided data columns" in str(ex):
                self._logger.warning(str(ex))
                return np.nan
            else:
                raise ex

        if empty:
            self._logger.warning(f"Not enough data to compute estimated {self.display_name}.")
            warnings.warn(f"Not enough data to compute estimated {self.display_name}.")
            return np.nan

        # TODO: put in a function? Also for MC CM.
        y_pred_proba = {key: data[value] for key, value in self.y_pred_proba.items()}
        y_pred = data[self.y_pred]
        num_classes = len(self.classes)
        est_confusion_matrix = np.zeros((num_classes, num_classes))
        # CM elements are properly ordered because y_pred_proba items are selected from self.classes[index]
        for i in range(num_classes):
            for j in range(num_classes):
                est_confusion_matrix[i, j] = np.sum(
                    np.where(
                        (y_pred == self.classes[j]),
                        y_pred_proba[self.classes[i]],
                        0,
                    )
                )

        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                est_confusion_matrix = est_confusion_matrix / est_confusion_matrix.sum(axis=0, keepdims=True)
            est_confusion_matrix = np.nan_to_num(est_confusion_matrix)

        return (self.business_value_matrix * est_confusion_matrix).sum()

    def _sampling_error(self, data: pd.DataFrame) -> float:
        needed_columns = self.class_probability_columns + [self.y_pred]
        _list_missing(needed_columns, data)
        data, empty = common_nan_removal(data[needed_columns], needed_columns)
        if empty:
            _message = f"Too many missing values, cannot calculate {self.display_name} sampling error. Returning NaN."
            self._logger.warning(_message)
            warnings.warn(_message)
            return np.nan
        else:
            return mse.business_value_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        try:
            _list_missing([self.y_true, self.y_pred], data)
        except InvalidArgumentsException as ex:
            if "missing required columns" in str(ex):
                self._logger.info(str(ex))
                return np.nan
            else:
                raise ex
        data, empty = common_nan_removal(data[[self.y_true, self.y_pred]], [self.y_true, self.y_pred])
        if empty:
            _message = f"'{self.y_true}' contains no data, cannot calculate business value. Returning NaN."
            self._logger.info(_message)
            warnings.warn(_message)
            return np.nan

        y_true = data[self.y_true]
        y_pred = data[self.y_pred]

        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)

        return (self.business_value_matrix * cm).sum()
