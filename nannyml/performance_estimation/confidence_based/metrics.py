import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score

from nannyml._typing import UseCase
from nannyml.base import AbstractEstimator
from nannyml.chunk import Chunk
from nannyml.exceptions import InvalidArgumentsException


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        estimator: AbstractEstimator,
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

        from .cbpe import CBPE

        if not isinstance(estimator, CBPE):
            raise RuntimeError(f"{estimator.__class__.__name__} is not an instance of type " f"CBPE")

        self.estimator = estimator

        self.upper_threshold: Optional[float] = None
        self.lower_threshold: Optional[float] = None
        self.confidence_deviation: Optional[float] = None

        self.reference_stability = 0.0

    def fit(self, reference_data: pd.DataFrame):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.

        """
        # Calculate alert thresholds
        reference_chunks = self.estimator.chunker.split(
            reference_data,
            timestamp_column_name=self.estimator.timestamp_column_name,
        )
        self.lower_threshold, self.upper_threshold = self._alert_thresholds(reference_chunks)

        # Calculate confidence bands
        self.confidence_deviation = self._confidence_deviation(reference_chunks)

        # Calculate reference stability
        self.reference_stability = self._reference_stability(reference_chunks)

        # Delegate to subclass
        self._fit(reference_data)

        return

    @abc.abstractmethod
    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError

    def estimate(self, data: pd.DataFrame):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate performance metrics on. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated), as well as the target data.
        """
        reference_chunks = self.estimator.chunker.split(data, self.estimator.timestamp_column_name)
        return self._estimate(reference_chunks)

    @abc.abstractmethod
    def _estimate(self, data: pd.DataFrame):
        raise NotImplementedError

    @abc.abstractmethod
    def _reference_stability(self, reference_chunks: List[Chunk]) -> float:
        raise NotImplementedError

    def _confidence_deviation(self, reference_chunks: List[Chunk]):
        return np.std([self._estimate(chunk.data) for chunk in reference_chunks])

    def _alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        realized_chunk_performance = [self._realized_performance(chunk) for chunk in reference_chunks]
        deviation = np.std(realized_chunk_performance) * std_num
        mean_realised_performance = np.mean(realized_chunk_performance)
        lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
        upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

        return lower_threshold, upper_threshold

    @abc.abstractmethod
    def _realized_performance(self, chunk: Chunk) -> float:
        raise NotImplementedError

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return self.display_name == other.display_name and self.column_name == other.column_name

    def _common_cleaning(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.estimator.y_true not in data.columns or data[self.estimator.y_true].isna().all():
            return np.NaN

        y_true = data[self.estimator.y_true]
        y_pred_proba = data[self.estimator.y_pred_proba]
        y_pred = data[self.estimator.y_pred]

        y_true = y_true[~y_pred_proba.isna()]
        y_pred_proba.dropna(inplace=True)

        y_pred_proba = y_pred_proba[~y_true.isna()]
        y_pred = y_pred[~y_true.isna()]
        y_true.dropna(inplace=True)

        return y_pred_proba, y_pred, y_true


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Dict[UseCase, Metric]] = {}

    @classmethod
    def _logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    @classmethod
    def create(cls, key: str, use_case: UseCase, kwargs: Dict[str, Any] = {}) -> Metric:
        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'" "Please provide a string, function or Metric"
            )

        if key not in cls.registry:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. "
                "Should be one of ['roc_auc', 'f1', 'precision', 'recall', 'specificity', "
                "'accuracy']."
            )

        if use_case not in cls.registry[key]:
            raise RuntimeError(
                f"metric '{key}' is currently not supported for use case {use_case}. "
                "Please specify another metric or use one of these supported model types for this metric: "
                f"{[md for md in cls.registry[key]]}"
            )
        metric_class = cls.registry[key][use_case]
        return metric_class(**kwargs)  # type: ignore

    @classmethod
    def register(cls, metric: str, use_case: UseCase) -> Callable:
        def inner_wrapper(wrapped_class: Metric) -> Metric:
            if metric in cls.registry:
                if use_case in cls.registry[metric]:
                    cls._logger().warning(f"re-registering Metric for metric='{metric}' and use_case='{use_case}'")
                cls.registry[metric][use_case] = wrapped_class
            else:
                cls.registry[metric] = {use_case: wrapped_class}
            return wrapped_class

        return inner_wrapper


@MetricFactory.register('roc_auc', UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    def __init__(self, estimator):
        super().__init__(display_name='ROC AUC', column_name='roc_auc', estimator=estimator)

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.estimator.y_pred_proba]

        thresholds = np.sort(y_pred_proba)
        one_min_thresholds = 1 - thresholds

        TP = np.cumsum(thresholds[::-1])[::-1]
        FP = np.cumsum(one_min_thresholds[::-1])[::-1]

        thresholds_with_zero = np.insert(thresholds, 0, 0, axis=0)[:-1]
        one_min_thresholds_with_zero = np.insert(one_min_thresholds, 0, 0, axis=0)[:-1]

        FN = np.cumsum(thresholds_with_zero)
        TN = np.cumsum(one_min_thresholds_with_zero)

        non_duplicated_thresholds = np.diff(np.insert(thresholds, 0, -1, axis=0)).astype(bool)
        TP = TP[non_duplicated_thresholds]
        FP = FP[non_duplicated_thresholds]
        FN = FN[non_duplicated_thresholds]
        TN = TN[non_duplicated_thresholds]

        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
        metric = auc(fpr, tpr)
        return metric

    def _realized_performance(self, chunk: Chunk) -> float:
        y_pred_proba, _, y_true = self._common_cleaning(chunk.data)
        return roc_auc_score(y_true, y_pred_proba)

    def _reference_stability(self, reference_chunks: List[Chunk]) -> float:
        return 0  # TODO: Jakub
