#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing metric utilities and implementations."""
import abc
import logging
from logging import Logger
from typing import Any, Callable, Dict, List, Tuple  # noqa: TYP001

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

from nannyml._typing import UseCase, model_output_column_names
from nannyml.base import AbstractCalculator, _list_missing
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        calculator: AbstractCalculator,
        upper_threshold: float = None,
        lower_threshold: float = None,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        display_name : str
            The name of the metric. Used to display in plots. If not given this name will be derived from the
            ``calculation_function``.
        column_name: str
            The name used to indicate the metric in columns of a DataFrame.
        calculator: AbstractCalculator
            The calculator using the Metric instance.
        upper_threshold : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.display_name = display_name
        self.column_name = column_name

        from .calculator import PerformanceCalculator

        if not isinstance(calculator, PerformanceCalculator):
            raise RuntimeError(f"{calculator.__class__.__name__} is not an instance of type " f"PerformanceCalculator")
        self.calculator = calculator
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def fit(self, reference_data: pd.DataFrame, chunker: Chunker):
        """Fits a Metric on reference data.

        Parameters
        ----------
        reference_data: pd.DataFrame
            The reference data used for fitting. Must have target data available.
        chunker: Chunker
            The :class:`~nannyml.chunk.Chunker` used to split the reference data into chunks.
            This value is provided by the calling
            :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator`.

        """
        self._fit(reference_data)

        # Calculate alert thresholds
        if self.upper_threshold is None and self.lower_threshold is None:
            reference_chunks = chunker.split(
                reference_data,
                timestamp_column_name=self.calculator.timestamp_column_name,
            )
            self.lower_threshold, self.upper_threshold = self._calculate_alert_thresholds(reference_chunks)

        return

    def _fit(self, reference_data: pd.DataFrame):
        raise NotImplementedError

    def calculate(self, data: pd.DataFrame):
        """Calculates performance metrics on data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to calculate performance metrics on. Requires presence of either the predicted labels or
            prediction scores/probabilities (depending on the metric to be calculated), as well as the target data.
        """
        return self._calculate(data)

    def _calculate(self, data: pd.DataFrame):
        raise NotImplementedError

    def _calculate_alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        chunked_reference_metric = [self.calculate(chunk.data) for chunk in reference_chunks]
        deviation = np.std(chunked_reference_metric) * std_num
        mean_reference_metric = np.mean(chunked_reference_metric)
        lower_threshold = np.maximum(mean_reference_metric - deviation, lower_limit)
        upper_threshold = np.minimum(mean_reference_metric + deviation, upper_limit)
        return lower_threshold, upper_threshold

    def __eq__(self, other):
        """Establishes equality by comparing all properties."""
        return (
            self.display_name == other.display_name
            and self.column_name == other.column_name
            and self.upper_threshold == other.upper_threshold
            and self.lower_threshold == other.lower_threshold
        )


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    registry: Dict[str, Dict[UseCase, Metric]] = {}

    @classmethod
    def _logger(cls) -> Logger:
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


@MetricFactory.register(metric='roc_auc', use_case=UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self, calculator):
        """Creates a new AUROC instance."""
        super().__init__(display_name='ROC AUC', column_name='roc_auc', calculator=calculator)

    def __str__(self):
        return "roc_auc"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred_proba], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.calculator.y_true, self.calculator.y_pred_proba], list(data.columns))

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred_proba]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)


@MetricFactory.register(metric='f1', use_case=UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationF1(Metric):
    """F1 score metric."""

    def __init__(self, calculator):
        """Creates a new F1 instance."""
        super().__init__(display_name='F1', column_name='f1', calculator=calculator)

    def __str__(self):
        return "f1"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(data.columns))

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred)


@MetricFactory.register(metric='precision', use_case=UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationPrecision(Metric):
    """Precision metric."""

    def __init__(self, calculator):
        """Creates a new Precision instance."""
        super().__init__(display_name='Precision', column_name='precision', calculator=calculator)

    def __str__(self):
        return "precision"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(data.columns))

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred)


@MetricFactory.register(metric='recall', use_case=UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    def __init__(self, calculator):
        """Creates a new Recall instance."""
        super().__init__(display_name='Recall', column_name='recall', calculator=calculator)

    def __str__(self):
        return "recall"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(data.columns))

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred)


@MetricFactory.register(metric='specificity', use_case=UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationSpecificity(Metric):
    """Specificity metric."""

    def __init__(self, calculator):
        """Creates a new F1 instance."""
        super().__init__(display_name='Specificity', column_name='specificity', calculator=calculator)

    def __str__(self):
        return "specificity"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(data.columns))

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)


@MetricFactory.register(metric='accuracy', use_case=UseCase.CLASSIFICATION_BINARY)
class BinaryClassificationAccuracy(Metric):
    """Accuracy metric."""

    def __init__(self, calculator):
        """Creates a new Accuracy instance."""
        super().__init__(display_name='Accuracy', column_name='accuracy', calculator=calculator)

    def __str__(self):
        return "accuracy"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(data.columns))

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all():
            raise InvalidArgumentsException(
                f"could not calculate metric '{self.display_name}': " "prediction column contains no data"
            )

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return (tp + tn) / (tp + tn + fp + fn)


def _common_data_cleaning(y_true, y_pred):
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    y_pred = y_pred[~y_true.isna()]
    y_true.dropna(inplace=True)

    return y_true, y_pred


@MetricFactory.register(metric='roc_auc', use_case=UseCase.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self, calculator):
        """Creates a new AUROC instance."""
        super().__init__(display_name='ROC AUC', column_name='roc_auc', calculator=calculator)
        self._min_chunk_size = 300

    def __str__(self):
        return "roc_auc"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], list(reference_data.columns))
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.calculator.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.calculator.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.calculator.y_true] + model_output_column_names(self.calculator.y_pred_proba), data)

        labels, class_probability_columns = [], []
        for label in sorted(list(self.calculator.y_pred_proba.keys())):
            labels.append(label)
            class_probability_columns.append(self.calculator.y_pred_proba[label])

        y_true = data[self.calculator.y_true]
        y_pred = data[class_probability_columns]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro', labels=labels)


@MetricFactory.register(metric='f1', use_case=UseCase.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationF1(Metric):
    """F1 score metric."""

    def __init__(self, calculator):
        """Creates a new F1 instance."""
        super().__init__(display_name='F1', column_name='f1', calculator=calculator)
        self._min_chunk_size = 300

    def __str__(self):
        return "f1"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], reference_data)
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.calculator.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.calculator.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.calculator.y_true, self.calculator.y_pred], data)

        labels = sorted(list(self.calculator.y_pred_proba.keys()))
        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred, average='macro', labels=labels)


@MetricFactory.register(metric='precision', use_case=UseCase.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationPrecision(Metric):
    """Precision metric."""

    def __init__(self, calculator):
        """Creates a new Precision instance."""
        super().__init__(display_name='Precision', column_name='precision', calculator=calculator)
        self._min_chunk_size = 300

    def __str__(self):
        return "precision"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], reference_data)
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.calculator.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.calculator.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.calculator.y_true, self.calculator.y_pred], data)

        labels = sorted(list(self.calculator.y_pred_proba.keys()))
        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred, average='macro', labels=labels)


@MetricFactory.register(metric='recall', use_case=UseCase.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    def __init__(self, calculator):
        """Creates a new Recall instance."""
        super().__init__(display_name='Recall', column_name='recall', calculator=calculator)
        self._min_chunk_size = 300

    def __str__(self):
        return "recall"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], reference_data)
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.calculator.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.calculator.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.calculator.y_true, self.calculator.y_pred], data)

        labels = sorted(list(self.calculator.y_pred_proba.keys()))
        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred, average='macro', labels=labels)


@MetricFactory.register(metric='specificity', use_case=UseCase.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationSpecificity(Metric):
    """Specificity metric."""

    def __init__(self, calculator):
        """Creates a new Specificity instance."""
        super().__init__(display_name='Specificity', column_name='specificity', calculator=calculator)
        self._min_chunk_size = 300

    def __str__(self):
        return "specificity"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], reference_data)
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.calculator.y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type {type(self.calculator.y_pred_proba)}\n"
                f"multiclass use cases require 'y_pred_proba' to "
                "be a dictionary mapping classes to columns."
            )

        _list_missing([self.calculator.y_true, self.calculator.y_pred], data)

        labels = sorted(list(self.calculator.y_pred_proba.keys()))
        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            MCM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
            tn_sum = MCM[:, 0, 0]
            fp_sum = MCM[:, 0, 1]
            class_wise_specificity = tn_sum / (tn_sum + fp_sum)
            return np.mean(class_wise_specificity)


@MetricFactory.register(metric='accuracy', use_case=UseCase.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAccuracy(Metric):
    """Accuracy metric."""

    def __init__(self, calculator):
        """Creates a new Accuracy instance."""
        super().__init__(display_name='Accuracy', column_name='accuracy', calculator=calculator)
        self._min_chunk_size = 300

    def __str__(self):
        return "accuracy"

    def _fit(self, reference_data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], reference_data)
        self._reference_stability = 0  # TODO: Jakub

    def _calculate(self, data: pd.DataFrame):
        _list_missing([self.calculator.y_true, self.calculator.y_pred], data)

        y_true = data[self.calculator.y_true]
        y_pred = data[self.calculator.y_pred]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric '{self.display_name}': " "prediction column contains no data"
            )

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return accuracy_score(y_true, y_pred)
