#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

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
from nannyml._typing import ModelOutputsType, ProblemType, class_labels
from nannyml.base import _remove_nans
from nannyml.chunk import Chunker
from nannyml.exceptions import CalculatorException, InvalidArgumentsException
from nannyml.performance_estimation.confidence_based import SUPPORTED_METRIC_VALUES
from nannyml.sampling_error import SAMPLING_ERROR_RANGE
from nannyml.thresholds import Threshold


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

        # self.confidence_deviation: Optional[float] = None

        # self.confidence_upper_bound: Optional[float] = 1.0
        # self.confidence_lower_bound: Optional[float] = 0.0

        # A list of (display_name, column_name) tuples
        self.components: List[Tuple[str, str]] = components

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    @property
    def display_name(self) -> str:  # noqa: # noqa: D102
        return self.name

    @property
    def column_name(self) -> str:  # noqa: # noqa: D102
        return self.components[0][1]

    @property
    def display_names(self):  # noqa: # noqa: D102
        return [c[0] for c in self.components]

    @property
    def column_names(self):  # noqa: # noqa: D102
        return [c[1] for c in self.components]

    def __str__(self):  # noqa: # noqa: D105
        return self.display_name

    def __repr__(self):  # noqa: # noqa: D105
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

        return

    @abc.abstractmethod
    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the _fit method"
        )

    @abc.abstractmethod
    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
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

    def _common_cleaning(
        self,
        data: pd.DataFrame,
        y_pred_proba_column_name: Optional[ModelOutputsType] = None,
        optional_column: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, Union[pd.Series, None], Union[pd.Series, None]]:
        if y_pred_proba_column_name is None:
            if not isinstance(self.y_pred_proba, str):
                raise InvalidArgumentsException(
                    f"'y_pred_proba' is of type '{type(self.y_pred_proba)}'. "
                    f"Binary use cases require 'y_pred_proba' to be a string."
                )
            y_pred_proba_column_name = self.y_pred_proba

        data = _remove_nans(data, [self.y_pred, y_pred_proba_column_name])

        clean_targets = self.y_true in data.columns and not data[self.y_true].isna().all()
        if clean_targets:
            data = _remove_nans(data, [self.y_true])

        return (
            data[y_pred_proba_column_name],
            data[self.y_pred],
            data[self.y_true] if clean_targets else None,
            data[optional_column] if optional_column else None,
        )

    def get_chunk_record(
        self,
        chunk_data_outputs: pd.DataFrame,
        reference_data_outputs: pd.DataFrame,
        reference_weights: np.ndarray
    ) -> Dict:
        """Returns a dictionary containing the performance metrics for a given chunk.

        Parameters
        ----------
        chunk_data_outputs : pd.DataFrame
            A pandas dataframe containing the output data for a given chunk.
        reference_data_outputs : pd.DataFrame
            A pandas dataframe containing the output data reference data.
        reference_weights : np.ndarray
            A numpy array containing the weights of each reference data row.

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
        estimated_metric_value = self._estimate(reference_data_outputs, reference_weights)
        metric_estimate_sampling_error = self._sampling_error(chunk_data_outputs)
        chunk_record[f'estimated_{column_name}'] = estimated_metric_value
        chunk_record[f'sampling_error_{column_name}'] = metric_estimate_sampling_error
        chunk_record[f'realized_{column_name}'] = self._realized_performance(chunk_data_outputs)
        chunk_record[f'upper_confidence_boundary_{column_name}'] = np.minimum(
            self.upper_threshold_value_limit or np.inf,
            estimated_metric_value + SAMPLING_ERROR_RANGE * metric_estimate_sampling_error,
        )
        chunk_record[f'lower_confidence_boundary_{column_name}'] = np.maximum(
            self.lower_threshold_value_limit or -np.inf,
            estimated_metric_value - SAMPLING_ERROR_RANGE * metric_estimate_sampling_error,
        )
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


@MetricFactory.register('accuracy', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAccuracy(Metric):
    """IW binary classification accuracy Metric Class."""
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
        """Initialize IW binary classification accuracy Metric Class."""
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
        self._sampling_error_components = bse.accuracy_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        # don't expect weights to have issues.
        return accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.accuracy_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return accuracy_score(y_true=y_true, y_pred=y_pred)


@MetricFactory.register('roc_auc', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    """IW binary classification AUROC Metric Class."""
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
        """Initialize IW binary classification AUROC Metric Class."""
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
        self._sampling_error_components = bse.auroc_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_proba_reference=reference_data[self.y_pred_proba],
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        reference_data_outputs['reference_weights'] = reference_weights
        y_pred_proba, _, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        # don't expect weights to have issues.
        return roc_auc_score(y_true=y_true, y_score=y_pred_proba, sample_weight=weights)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        y_pred_proba, _, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        return roc_auc_score(y_true, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.auroc_sampling_error(self._sampling_error_components, data)


@MetricFactory.register('f1', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationF1(Metric):
    """IW binary classification f1 Metric Class."""
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
        """Initialize IW binary classification f1 Metric Class."""
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
        self._sampling_error_components = bse.f1_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        # don't expect weights to have issues.
        return f1_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.f1_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return f1_score(y_true=y_true, y_pred=y_pred)


@MetricFactory.register('precision', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationPrecision(Metric):
    """IW binary classification precision Metric Class."""
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
        """Initialize IW binary classification precision Metric Class."""
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
        self._sampling_error_components = bse.precision_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        # don't expect weights to have issues.
        return precision_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return precision_score(y_true=y_true, y_pred=y_pred)


@MetricFactory.register('recall', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationRecall(Metric):
    """IW binary classification recall Metric Class."""
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
        """Initialize IW binary classification recall Metric Class."""
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
        self._sampling_error_components = bse.recall_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        # don't expect weights to have issues.
        return recall_score(y_true=y_true, y_pred=y_pred, sample_weight=weights)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.recall_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return recall_score(y_true=y_true, y_pred=y_pred)


@MetricFactory.register('specificity', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationSpecificity(Metric):
    """IW binary classification specificity Metric Class."""
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
        """Initialize IW binary classification specificity Metric Class."""
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

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.specificity_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        # don't expect weights to have issues.
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=weights).ravel()
        return tn / (tn + fp)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.specificity_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)


@MetricFactory.register('confusion_matrix', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationConfusionMatrix(Metric):
    """IW binary classification Confusion Matrix Metric Class."""
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        normalize_confusion_matrix: Optional[str] = None,
        **kwargs,
    ):
        """Initialize IW binary classification Confusion Matrix Metric Class."""
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

    def _fit(self, reference_data: pd.DataFrame):
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

    def _realized_performance_cm_elements(self, data: pd.DataFrame) -> Optional[Tuple[float, float, float, float]]:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return None
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return None
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return None
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize=self.normalize_confusion_matrix).ravel()
        return (tn, fp, fn, tp)

    def _estimate_cm_elements(
        self,
        reference_data_outputs: pd.DataFrame,
        reference_weights: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return (np.NaN, np.NaN, np.NaN, np.NaN)
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return (np.NaN, np.NaN, np.NaN, np.NaN)
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return (np.NaN, np.NaN, np.NaN, np.NaN)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            normalize=self.normalize_confusion_matrix,
            sample_weight=weights
        ).ravel()
        return (tn, fp, fn, tp)

    def get_true_pos_info(
        self,
        chunk_data: pd.DataFrame,
        est_tp: float,
        rl_tp: float
    ) -> Dict:
        """Returns a dictionary containing infomation about the true positives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.
        est_tp : float
            estimated true positive values
        rl_tp : pd.DataFrame
            realized true positive values

        Returns
        -------
        true_pos_info : Dict
            A dictionary of true positive's information and its value pairs.
        """
        true_pos_info: Dict[str, Any] = {}

        sampling_error_true_positives = bse.true_positive_sampling_error(
            self._true_positive_sampling_error_components, chunk_data
        )

        true_pos_info['estimated_true_positive'] = est_tp
        true_pos_info['sampling_error_true_positive'] = sampling_error_true_positives
        true_pos_info['realized_true_positive'] = rl_tp

        if self.normalize_confusion_matrix is None:
            true_pos_info['upper_confidence_boundary_true_positive'] = (
                est_tp + SAMPLING_ERROR_RANGE * sampling_error_true_positives
            )
        else:
            true_pos_info['upper_confidence_boundary_true_positive'] = np.minimum(
                self.upper_threshold_value_limit or np.inf,
                est_tp + SAMPLING_ERROR_RANGE * sampling_error_true_positives,
            )

        true_pos_info['lower_confidence_boundary_true_positive'] = np.maximum(
            self.lower_threshold_value_limit or -np.inf,
            est_tp - SAMPLING_ERROR_RANGE * sampling_error_true_positives
        )

        return true_pos_info

    def get_true_neg_info(
        self,
        chunk_data: pd.DataFrame,
        est_tn: float,
        rl_tn: float
    ) -> Dict:
        """Returns a dictionary containing infomation about the true negatives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.
        est_tn : float
            estimated true negative values
        rl_tn : float
            realized true negative values

        Returns
        -------
        true_neg_info : Dict
            A dictionary of true negative's information and its value pairs.
        """
        true_neg_info: Dict[str, Any] = {}

        sampling_error_true_negatives = bse.true_negative_sampling_error(
            self._true_negative_sampling_error_components, chunk_data
        )

        true_neg_info['estimated_true_negative'] = est_tn
        true_neg_info['sampling_error_true_negative'] = sampling_error_true_negatives
        true_neg_info['realized_true_negative'] = rl_tn

        if self.normalize_confusion_matrix is None:
            true_neg_info['upper_confidence_boundary_true_negative'] = (
                est_tn + SAMPLING_ERROR_RANGE * sampling_error_true_negatives
            )
        else:
            true_neg_info['upper_confidence_boundary_true_negative'] = np.minimum(
                self.upper_threshold_value_limit or np.inf,
                est_tn + SAMPLING_ERROR_RANGE * sampling_error_true_negatives,
            )

        true_neg_info['lower_confidence_boundary_true_negative'] = np.maximum(
            self.lower_threshold_value_limit or -np.inf,
            est_tn - SAMPLING_ERROR_RANGE * sampling_error_true_negatives
        )

        return true_neg_info

    def get_false_pos_info(
        self,
        chunk_data: pd.DataFrame,
        est_fp: float,
        rl_fp: float
    ) -> Dict:
        """Returns a dictionary containing infomation about the false positives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.
        est_fp : float
            estimated false positive values
        rl_fp : float
            realized false positive values

        Returns
        -------
        false_pos_info : Dict
            A dictionary of false positive's information and its value pairs.
        """
        false_pos_info: Dict[str, Any] = {}

        sampling_error_false_positives = bse.false_positive_sampling_error(
            self._false_positive_sampling_error_components, chunk_data
        )

        false_pos_info['estimated_false_positive'] = est_fp
        false_pos_info['sampling_error_false_positive'] = sampling_error_false_positives
        false_pos_info['realized_false_positive'] = rl_fp

        if self.normalize_confusion_matrix is None:
            false_pos_info['upper_confidence_boundary_false_positive'] = (
                est_fp + SAMPLING_ERROR_RANGE * sampling_error_false_positives
            )
        else:
            false_pos_info['upper_confidence_boundary_false_positive'] = np.minimum(
                self.upper_threshold_value_limit or np.inf,
                est_fp + SAMPLING_ERROR_RANGE * sampling_error_false_positives,
            )

        false_pos_info['lower_confidence_boundary_false_positive'] = np.maximum(
            self.lower_threshold_value_limit or -np.inf,
            est_fp - SAMPLING_ERROR_RANGE * sampling_error_false_positives,
        )

        return false_pos_info

    def get_false_neg_info(
        self,
        chunk_data: pd.DataFrame,
        est_fn: float,
        rl_fn: float,
    ) -> Dict:
        """Returns a dictionary containing infomation about the false negatives for a given chunk.

        Parameters
        ----------
        chunk_data : pd.DataFrame
            A pandas dataframe containing the data for a given chunk.
        est_fn : float
            estimated false negative values
        rl_fn : float
            realized false negative values

        Returns
        -------
        false_neg_info : Dict
            A dictionary of false negative's information and its value pairs.
        """
        false_neg_info: Dict[str, Any] = {}

        sampling_error_false_negatives = bse.false_negative_sampling_error(
            self._false_negative_sampling_error_components, chunk_data
        )

        false_neg_info['estimated_false_negative'] = est_fn
        false_neg_info['sampling_error_false_negative'] = sampling_error_false_negatives
        false_neg_info['realized_false_negative'] = rl_fn

        if self.normalize_confusion_matrix is None:
            false_neg_info['upper_confidence_boundary_false_negative'] = (
                est_fn + SAMPLING_ERROR_RANGE * sampling_error_false_negatives
            )
        else:
            false_neg_info['upper_confidence_boundary_false_negative'] = np.minimum(
                self.upper_threshold_value_limit or np.inf,
                est_fn + SAMPLING_ERROR_RANGE * sampling_error_false_negatives,
            )

        false_neg_info['lower_confidence_boundary_false_negative'] = np.maximum(
            self.lower_threshold_value_limit or -np.inf,
            est_fn - SAMPLING_ERROR_RANGE * sampling_error_false_negatives,
        )

        return false_neg_info

    def get_chunk_record(
        self,
        chunk_data_outputs: pd.DataFrame,
        reference_data_outputs: pd.DataFrame,
        reference_weights: np.ndarray
    ) -> Dict:
        """Returns a dictionary containing the performance metrics for a given chunk.

        Parameters
        ----------
        chunk_data_outputs : pd.DataFrame
            A pandas dataframe containing the output data for a given chunk.
        reference_data_outputs : pd.DataFrame
            A pandas dataframe containing the output data reference data.
        reference_weights : np.ndarray
            A numpy array containing the weights of each reference data row.

        Returns
        -------
            chunk_record : Dict
                A dictionary of perfomance metric, value pairs.
        """
        realized_cm_elements = self._realized_performance_cm_elements(chunk_data_outputs)
        if realized_cm_elements:
            rtn, rfp, rfn, rtp = realized_cm_elements
        else:
            rtn, rfp, rfn, rtp = np.NaN, np.NaN, np.NaN, np.NaN

        estimated_cm_elements = self._estimate_cm_elements(
            reference_data_outputs, reference_weights
        )
        if estimated_cm_elements:
            etn, efp, efn, etp = estimated_cm_elements
        else:
            etn, efp, efn, etp = np.NaN, np.NaN, np.NaN, np.NaN

        chunk_record = {}

        true_pos_info = self.get_true_pos_info(chunk_data_outputs, etp, rtp)
        chunk_record.update(true_pos_info)

        true_neg_info = self.get_true_neg_info(chunk_data_outputs, etn, rtn)
        chunk_record.update(true_neg_info)

        false_pos_info = self.get_false_pos_info(chunk_data_outputs, efp, rfp)
        chunk_record.update(false_pos_info)

        false_neg_info = self.get_false_neg_info(chunk_data_outputs, efn, rfn)
        chunk_record.update(false_neg_info)

        return chunk_record

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        pass

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return 0.0

    def _realized_performance(self, data: pd.DataFrame) -> float:
        return 0.0


@MetricFactory.register('business_value', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationBusinessValue(Metric):
    """IW binary classification business value Metric Class."""
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        business_value_matrix: Union[List, np.ndarray],
        normalize_business_value: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize IW binary classification business value Metric Class."""
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

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.business_value_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            business_value_matrix=self.business_value_matrix,
            normalize_business_value=self.normalize_business_value,
        )

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _ = self._common_cleaning(data, y_pred_proba_column_name=self.y_pred_proba)

        if y_true is None:
            warnings.warn("No 'y_true' values given for chunk, returning NaN as realized business value.")
            return np.NaN
        if y_true.shape[0] == 0:
            warnings.warn("Calculated Business Value contains NaN values.")
            return np.NaN

        tp_value = self.business_value_matrix[1, 1]
        tn_value = self.business_value_matrix[0, 0]
        fp_value = self.business_value_matrix[0, 1]
        fn_value = self.business_value_matrix[1, 0]
        bv_array = np.array([[tn_value, fp_value], [fn_value, tp_value]])

        cm = confusion_matrix(y_true, y_pred)
        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)
        return (bv_array * cm).sum()

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_column_name=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN

        cm = confusion_matrix(y_true, y_pred, sample_weight=weights)

        tp_value = self.business_value_matrix[1, 1]
        tn_value = self.business_value_matrix[0, 0]
        fp_value = self.business_value_matrix[0, 1]
        fn_value = self.business_value_matrix[1, 0]
        bv_array = np.array([[tn_value, fp_value], [fn_value, tp_value]])

        if self.normalize_business_value == 'per_prediction':
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=0, keepdims=True)
            cm = np.nan_to_num(cm)
        return (bv_array * cm).sum()

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.business_value_sampling_error(
            self._sampling_error_components,
            data,
        )


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


class _MulticlassClassificationMetric(Metric):
    """Base class for multiclass classification metrics."""

    def _common_cleaning(  # type: ignore # ignore override error
        self,
        data: pd.DataFrame,
        y_pred_proba_columns_dict: Optional[ModelOutputsType] = None,
        optional_column: Optional[str] = None
    ) -> Tuple[
        Union[pd.DataFrame, None],
        Union[pd.Series, None],
        Union[pd.Series, None],
        Union[pd.Series, None],
        List[str]
    ]:
        """Multiclass implementation of _common_cleaning."""
        if y_pred_proba_columns_dict is None:
            y_pred_proba_columns_dict = self.y_pred_proba
        if y_pred_proba_columns_dict is None:
            y_pred_proba_columns_dict = {}
        elif not isinstance(y_pred_proba_columns_dict, dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type '{type(self.y_pred_proba)}'. "
                f"Multiclass use cases require 'y_pred_proba' to be a dictionary."
            )

        labels, class_probability_columns = [], []
        for label in sorted(y_pred_proba_columns_dict.keys()):
            labels.append(label)
            class_probability_columns.append(f'{y_pred_proba_columns_dict[label]}')

        data = _remove_nans(data, [self.y_pred] + class_probability_columns)

        clean_targets = self.y_true in data.columns and not data[self.y_true].isna().all()
        if clean_targets:
            data = _remove_nans(data, [self.y_true])

        return (
            data[class_probability_columns] if len(class_probability_columns) > 0 else None,
            data[self.y_pred],
            data[self.y_true] if clean_targets else None,
            data[optional_column] if optional_column else None,
            labels
        )


@MetricFactory.register('accuracy', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAccuracy(_MulticlassClassificationMetric):
    """IW Multiclass accuracy Metric Class."""
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
        """Initialize IW Multiclass accuracy Metric Class."""
        super().__init__(
            name='accuracy',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Accuracy', 'accuracy')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        label_binarizer = LabelBinarizer()
        binarized_y_true = label_binarizer.fit_transform(reference_data[self.y_true])
        binarized_y_pred = label_binarizer.transform(reference_data[self.y_pred])

        self._sampling_error_components = mse.accuracy_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights, _ = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return accuracy_score(y_true, y_pred, sample_weight=weights)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.accuracy_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _, _ = self._common_cleaning(data)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return accuracy_score(y_true, y_pred)


@MetricFactory.register('roc_auc', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAUROC(_MulticlassClassificationMetric):
    """IW Multiclass AUROC Metric Class."""
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
        """Initialize IW Multiclass AUROC Metric Class."""
        super().__init__(
            name='roc_auc',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('ROC AUC', 'roc_auc')],
        )
        self.y_pred_proba: Dict[str, str]
        if not isinstance(self.y_pred_proba, dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type '{type(self.y_pred_proba)}'. "
                f"Multiclass use cases require 'y_pred_proba' to be a dictionary."
            )
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        classes = class_labels(self.y_pred_proba)
        binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=classes).T)
        y_pred_proba = [reference_data[self.y_pred_proba[clazz]].T for clazz in classes]
        self._sampling_error_components = mse.auroc_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_proba_reference=y_pred_proba
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        y_pred_probas, _, y_true, weights, labels = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        return roc_auc_score(
            y_true=y_true,
            y_score=y_pred_probas,
            average='macro',
            multi_class='ovr',
            sample_weight=weights,
            labels=labels
        )

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.auroc_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        y_pred_probas, _, y_true, _, labels = self._common_cleaning(data)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        return roc_auc_score(y_true=y_true, y_score=y_pred_probas, average='macro', multi_class='ovr', labels=labels)


@MetricFactory.register('f1', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationF1(_MulticlassClassificationMetric):
    """IW Multiclass f1 Metric Class."""
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
        """Initialize IW Multiclass f1 Metric Class."""
        super().__init__(
            name='f1',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('F1', 'f1')],
        )
        # sampling error:
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
        self._sampling_error_components = mse.f1_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights, labels = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro',
            sample_weight=weights,
            labels=labels
        )

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.f1_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _, labels = self._common_cleaning(data)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('precision', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationPrecision(_MulticlassClassificationMetric):
    """IW Multiclass precision Metric Class."""
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
        """Initialize IW Multiclass precision Metric Class."""
        super().__init__(
            name='precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Precision', 'precision')],
        )
        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
        self._sampling_error_components = mse.precision_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights, labels = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return precision_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro',
            sample_weight=weights,
            labels=labels
        )

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _, labels = self._common_cleaning(data)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('recall', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationRecall(_MulticlassClassificationMetric):
    """IW Multiclass recall Metric Class."""
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
        """Initialize IW Multiclass recall Metric Class."""
        super().__init__(
            name='recall',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Recall', 'recall')],
        )
        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
        self._sampling_error_components = mse.recall_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights, labels = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro',
            sample_weight=weights,
            labels=labels
        )

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.recall_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _, labels = self._common_cleaning(data)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        return recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('specificity', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationSpecificity(_MulticlassClassificationMetric):
    """IW Multiclass specificity Metric Class."""
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
        """Initialize IW Multiclass specificity Metric Class."""
        super().__init__(
            name='specificity',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            threshold=threshold,
            components=[('Specificity', 'specificity')],
        )
        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        label_binarizer = LabelBinarizer()
        binarized_y_true = list(label_binarizer.fit_transform(reference_data[self.y_true]).T)
        binarized_y_pred = list(label_binarizer.transform(reference_data[self.y_pred]).T)
        self._sampling_error_components = mse.specificity_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_reference=binarized_y_pred
        )

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray) -> float:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights, labels = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels, sample_weight=weights)
        tn_sum = mcm[:, 0, 0]
        fp_sum = mcm[:, 0, 1]
        class_wise_specificity = tn_sum / (tn_sum + fp_sum)
        return np.mean(class_wise_specificity)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.specificity_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true, _, labels = self._common_cleaning(data)
        if y_true is None:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return np.NaN
        if y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return np.NaN
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return np.NaN
        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        tn_sum = mcm[:, 0, 0]
        fp_sum = mcm[:, 0, 1]
        class_wise_specificity = tn_sum / (tn_sum + fp_sum)
        return np.mean(class_wise_specificity)


@MetricFactory.register('confusion_matrix', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationConfusionMatrix(_MulticlassClassificationMetric):
    """IW Multiclass Confusion Matrix Metric Class."""
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        threshold: Threshold,
        timestamp_column_name: Optional[str] = None,
        normalize_confusion_matrix: Optional[str] = None,
        **kwargs,
    ):
        """Initialize IW Multiclass Confusion Matrix Metric Class."""
        if isinstance(y_pred_proba, str):
            raise ValueError(
                "y_pred_proba must be a dictionary with class labels as keys and pred_proba column names as values"
            )
        self.classes = list(y_pred_proba.keys())
        self.alert_thresholds_dict: Dict[str, Any] = {}

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

    def _fit(self, reference_data: pd.DataFrame):
        self._confusion_matrix_sampling_error_components = mse.multiclass_confusion_matrix_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
            normalize_confusion_matrix=self.normalize_confusion_matrix,
        )

    def _realized_multiclass_confusion_matrix(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        if data is None or self.y_true not in data.columns:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return None
        if data[self.y_true].nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return None
        if data[self.y_pred].nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return None
        cm = confusion_matrix(
            data[self.y_true], data[self.y_pred], labels=self.classes, normalize=self.normalize_confusion_matrix
        )
        return cm

    def _estimate_multiclass_confusion_matrix(
        self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray
    ) -> Optional[np.ndarray]:
        reference_data_outputs['reference_weights'] = reference_weights
        _, y_pred, y_true, weights, labels = self._common_cleaning(
            reference_data_outputs,
            y_pred_proba_columns_dict=self.y_pred_proba,
            optional_column='reference_weights'
        )
        if reference_data_outputs is None or self.y_true not in reference_data_outputs.columns:
            warnings.warn(f"No 'y_true' values given for chunk, returning NaN as realized {self.name}.")
            return None
        if y_true is None or y_true.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_true', returning NaN as realized {self.name}.")
            return None
        if y_pred is None or y_pred.nunique() <= 1:
            warnings.warn(f"Too few unique values present in 'y_pred', returning NaN as realized {self.name}.")
            return None
        cm = confusion_matrix(
            y_true, y_pred, labels=self.classes, normalize=self.normalize_confusion_matrix, sample_weight=weights
        )
        return cm

    def get_chunk_record(
        self,
        chunk_data_outputs: pd.DataFrame,
        reference_data_outputs: pd.DataFrame,
        reference_weights: np.ndarray
    ) -> Dict:
        """Returns a dictionary containing the performance metrics for a given chunk.

        Parameters
        ----------
        chunk_data_outputs : pd.DataFrame
            A pandas dataframe containing the output data for a given chunk.
        reference_data_outputs : pd.DataFrame
            A pandas dataframe containing the output data reference data.
        reference_weights : np.ndarray
            A numpy array containing the weights of each reference data row.

        Returns
        -------
            chunk_record : Dict
                A dictionary of perfomance metric, value pairs.
        """
        chunk_record: Dict = {}
        estimated_cm = self._estimate_multiclass_confusion_matrix(
            reference_data_outputs, reference_weights
        )
        realized_cm = self._realized_multiclass_confusion_matrix(chunk_data_outputs)
        sampling_error = mse.multiclass_confusion_matrix_sampling_error(
            self._confusion_matrix_sampling_error_components,
            chunk_data_outputs,
        )
        for true_class in self.classes:
            for pred_class in self.classes:

                estimated_value = estimated_cm[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ] if estimated_cm is not None else np.NaN

                chunk_record[f'estimated_true_{true_class}_pred_{pred_class}'] = estimated_value

                chunk_record[f'sampling_error_true_{true_class}_pred_{pred_class}'] = sampling_error[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ]

                chunk_record[f'realized_true_{true_class}_pred_{pred_class}'] = realized_cm[
                    self.classes.index(true_class), self.classes.index(pred_class)
                ] if realized_cm is not None else np.NaN

                upper_confidence_boundary = (
                    estimated_value
                    + SAMPLING_ERROR_RANGE
                    * sampling_error[self.classes.index(true_class), self.classes.index(pred_class)]
                )
                if self.normalize_confusion_matrix is None:
                    chunk_record[
                        f'upper_confidence_boundary_true_{true_class}_pred_{pred_class}'
                    ] = upper_confidence_boundary
                else:
                    chunk_record[f'upper_confidence_boundary_true_{true_class}_pred_{pred_class}'] = min(
                        self.upper_threshold_value_limit, upper_confidence_boundary
                    )

                lower_confidence_boundary = (
                    estimated_value
                    - SAMPLING_ERROR_RANGE
                    * sampling_error[self.classes.index(true_class), self.classes.index(pred_class)]
                )
                if self.normalize_confusion_matrix is None:
                    chunk_record[
                        f'lower_confidence_boundary_true_{true_class}_pred_{pred_class}'
                    ] = lower_confidence_boundary
                else:
                    chunk_record[f'lower_confidence_boundary_true_{true_class}_pred_{pred_class}'] = max(
                        self.lower_threshold_value_limit, lower_confidence_boundary
                    )
        return chunk_record

    def _estimate(self, reference_data_outputs: pd.DataFrame, reference_weights: np.ndarray):
        pass

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return 0.0

    def _realized_performance(self, data: pd.DataFrame) -> float:
        return 0.0
