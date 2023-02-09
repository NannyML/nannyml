import abc
import logging
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
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
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import CalculatorException, InvalidArgumentsException
from nannyml.sampling_error import SAMPLING_ERROR_RANGE


class Metric(abc.ABC):
    """A performance metric used to calculate realized model performance."""

    def __init__(
        self,
        name: str,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        components: List[Tuple[str, str]],
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new Metric instance.

        Parameters
        ----------
        name: str
            The name used to indicate the metric in columns of a DataFrame.
        """
        self.name = name

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name
        self.chunker = chunker

        self.upper_threshold: Optional[float] = None
        self.lower_threshold: Optional[float] = None
        self.confidence_deviation: Optional[float] = None

        self.uncalibrated_y_pred_proba = f'uncalibrated_{self.y_pred_proba}'

        self.confidence_upper_bound = 1
        self.confidence_lower_bound = 0

        # A list of (display_name, column_name) tuples
        self.components: List[Tuple[str, str]] = components

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def column_name(self) -> str:
        return self.components[0][0]

    @property
    def display_names(self):
        return [c[0] for c in self.components]

    @property
    def column_names(self):
        return [c[1] for c in self.components]

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.column_name

    def fit(self, reference_data: pd.DataFrame):
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
        self.lower_threshold, self.upper_threshold = self._alert_thresholds(reference_chunks)

        # Calculate confidence bands
        self.confidence_deviation = self._confidence_deviation(reference_chunks)

        # Delegate to subclass
        self._fit(reference_data)

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

    def _confidence_deviation(self, reference_chunks: List[Chunk]):
        return np.std([self._estimate(chunk.data) for chunk in reference_chunks])

    def _alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        realized_chunk_performance = [self._realized_performance(chunk.data) for chunk in reference_chunks]
        deviation = np.std(realized_chunk_performance) * std_num
        mean_realised_performance = np.mean(realized_chunk_performance)
        lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
        upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

        return lower_threshold, upper_threshold

    @abc.abstractmethod
    def _realized_performance(self, data: pd.DataFrame) -> float:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' is a subclass of Metric and it must implement the realized_performance method"
        )

    def __eq__(self, other):
        return self.components == other.components

    def _common_cleaning(
        self, data: pd.DataFrame, y_pred_proba_column_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if y_pred_proba_column_name is None:
            if not isinstance(self.y_pred_proba, str):
                raise InvalidArgumentsException(
                    f"'y_pred_proba' is of type '{type(self.y_pred_proba)}'. "
                    f"Binary use cases require 'y_pred_proba' to be a string."
                )
            y_pred_proba_column_name = self.y_pred_proba

        clean_targets = self.y_true in data.columns and not data[self.y_true].isna().all()

        y_pred_proba = data[y_pred_proba_column_name]
        y_pred = data[self.y_pred]

        y_pred_proba.dropna(inplace=True)

        if clean_targets:
            y_true = data[self.y_true]
            y_true = y_true[~y_pred_proba.isna()]
            y_pred_proba = y_pred_proba[~y_true.isna()]
            y_pred = y_pred[~y_true.isna()]
            y_true.dropna(inplace=True)
        else:
            y_true = None

        return y_pred_proba, y_pred, y_true

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
        if len(self.components) > 1:
            raise NotImplementedError(
                "cannot use default 'get_chunk_record' implementation when a metric has multiple components."
            )

        column_name = self.components[0][1]

        chunk_record = {}

        estimated_metric_value = self._estimate(chunk_data)

        metric_estimate_sampling_error = self._sampling_error(chunk_data)

        chunk_record[f'estimated_{column_name}'] = estimated_metric_value

        chunk_record[f'sampling_error_{column_name}'] = metric_estimate_sampling_error

        chunk_record[f'realized_{column_name}'] = self._realized_performance(chunk_data)

        chunk_record[f'upper_confidence_boundary_{column_name}'] = min(
            self.confidence_upper_bound, estimated_metric_value + SAMPLING_ERROR_RANGE * metric_estimate_sampling_error
        )

        chunk_record[f'lower_confidence_boundary_{column_name}'] = max(
            self.confidence_lower_bound, estimated_metric_value - SAMPLING_ERROR_RANGE * metric_estimate_sampling_error
        )

        chunk_record[f'upper_threshold_{column_name}'] = self.upper_threshold
        chunk_record[f'lower_threshold_{column_name}'] = self.lower_threshold

        chunk_record[f'alert_{column_name}'] = (
            estimated_metric_value > self.upper_threshold or estimated_metric_value < self.lower_threshold
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
        if kwargs is None:
            kwargs = {}

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
        return metric_class(**kwargs)

    @classmethod
    def register(cls, metric: str, use_case: ProblemType) -> Callable:
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
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='roc_auc',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('ROC AUC', 'roc_auc')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.auroc_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_proba_reference=reference_data[self.y_pred_proba],
        )

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.y_pred_proba]

        return estimate_roc_auc(y_pred_proba)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        y_pred_proba, _, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        return roc_auc_score(y_true, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.auroc_sampling_error(self._sampling_error_components, data)


def estimate_roc_auc(y_pred_proba: pd.Series) -> float:
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


@MetricFactory.register('f1', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationF1(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='f1',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('F1', 'f1')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.f1_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.y_pred_proba]
        y_pred = data[self.y_pred]

        return estimate_f1(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.f1_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        return f1_score(y_true=y_true, y_pred=y_pred)


def estimate_f1(y_pred: pd.DataFrame, y_pred_proba: pd.DataFrame) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    fn = np.where(y_pred == 0, y_pred_proba, 0)
    TP, FP, FN = np.sum(tp), np.sum(fp), np.sum(fn)
    metric = TP / (TP + 0.5 * (FP + FN))
    return metric


@MetricFactory.register('precision', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationPrecision(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('Precision', 'precision')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.precision_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )
        pass

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.y_pred_proba]
        y_pred = data[self.y_pred]

        return estimate_precision(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        return precision_score(y_true=y_true, y_pred=y_pred)


def estimate_precision(y_pred: pd.DataFrame, y_pred_proba: pd.DataFrame) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    TP, FP = np.sum(tp), np.sum(fp)
    metric = TP / (TP + FP)
    return metric


@MetricFactory.register('recall', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationRecall(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='recall',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('Recall', 'recall')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.recall_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.y_pred_proba]
        y_pred = data[self.y_pred]

        return estimate_recall(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.recall_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        return recall_score(y_true=y_true, y_pred=y_pred)


def estimate_recall(y_pred: pd.DataFrame, y_pred_proba: pd.DataFrame) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fn = np.where(y_pred == 0, y_pred_proba, 0)
    TP, FN = np.sum(tp), np.sum(fn)
    metric = TP / (TP + FN)
    return metric


@MetricFactory.register('specificity', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationSpecificity(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='specificity',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('Specificity', 'specificity')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.specificity_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.y_pred_proba]
        y_pred = data[self.y_pred]

        return estimate_specificity(y_pred, y_pred_proba)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.specificity_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        return conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])


def estimate_specificity(y_pred: pd.DataFrame, y_pred_proba: pd.DataFrame) -> float:
    tn = np.where(y_pred == 0, 1 - y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    TN, FP = np.sum(tn), np.sum(fp)
    metric = TN / (TN + FP)
    return metric


@MetricFactory.register('accuracy', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationAccuracy(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='accuracy',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('Accuracy', 'accuracy')],
        )

        # sampling error
        self._sampling_error_components: Tuple = ()

    def _fit(self, reference_data: pd.DataFrame):
        self._sampling_error_components = bse.accuracy_sampling_error_components(
            y_true_reference=reference_data[self.y_true],
            y_pred_reference=reference_data[self.y_pred],
        )

    def _estimate(self, data: pd.DataFrame):
        y_pred_proba = data[self.y_pred_proba]
        y_pred = data[self.y_pred]

        tp = np.where(y_pred == 1, y_pred_proba, 0)
        tn = np.where(y_pred == 0, 1 - y_pred_proba, 0)
        TP, TN = np.sum(tp), np.sum(tn)
        metric = (TP + TN) / len(y_pred)
        return metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return bse.accuracy_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        return accuracy_score(y_true=y_true, y_pred=y_pred)


@MetricFactory.register('confusion_matrix', ProblemType.CLASSIFICATION_BINARY)
class BinaryClassificationConfusionMatrix(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        normalize_confusion_matrix: Optional[str] = None,
    ):
        super().__init__(
            name='confusion_matrix',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[
                ('True Positive', 'true_positive'),
                ('True Negative', 'true_negative'),
                ('False Positive', 'false_positive'),
                ('False Negative', 'false_negative'),
            ],
        )

        self.normalize_confusion_matrix: Optional[str] = normalize_confusion_matrix

        self.true_positive_lower_threshold: float = 0
        self.true_positive_upper_threshold: float = 1
        self.true_negative_lower_threshold: float = 0
        self.true_negative_upper_threshold: float = 1

        self.true_positive_confidence_deviation: float = 0
        self.true_negative_confidence_deviation: float = 0
        self.false_positive_confidence_deviation: float = 0
        self.false_negative_confidence_deviation: float = 0

        # self.components = ["true_positive", "true_negative", "false_positive", "false_negative"]

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
            reference_chunks, std_num=3, lower_limit=0, upper_limit=1
        )
        self.true_negative_lower_threshold, self.true_negative_upper_threshold = self._true_negative_alert_thresholds(
            reference_chunks, std_num=3, lower_limit=0, upper_limit=1
        )
        (
            self.false_positive_lower_threshold,
            self.false_positive_upper_threshold,
        ) = self._false_positive_alert_thresholds(reference_chunks, std_num=3, lower_limit=0, upper_limit=1)
        (
            self.false_negative_lower_threshold,
            self.false_negative_upper_threshold,
        ) = self._false_negative_alert_thresholds(reference_chunks, std_num=3, lower_limit=0, upper_limit=1)

        # Calculate confidence bands
        self.true_positive_confidence_deviation = self._true_positive_confidence_deviation(reference_chunks)
        self.true_negative_confidence_deviation = self._true_negative_confidence_deviation(reference_chunks)
        self.false_positive_confidence_deviation = self._false_positive_confidence_deviation(reference_chunks)
        self.false_negative_confidence_deviation = self._false_negative_confidence_deviation(reference_chunks)

        # Delegate to confusion matrix subclass
        self._fit(reference_data)  # could probably put _fit functionality here since overide fit method

        return

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

    def _true_positive_alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        true_positive_realized_chunk_performance = [
            self._true_negative_realized_performance(chunk.data) for chunk in reference_chunks
        ]
        deviation = np.std(true_positive_realized_chunk_performance) * std_num
        true_positive_mean_realized_performance = np.mean(true_positive_realized_chunk_performance)

        if self.normalize_confusion_matrix is None:
            true_positive_upper_threshold = true_positive_mean_realized_performance + deviation
        else:
            true_positive_upper_threshold = np.maximum(upper_limit, true_positive_mean_realized_performance + deviation)

        true_positive_lower_threshold = np.minimum(lower_limit, true_positive_mean_realized_performance - deviation)

        return true_positive_lower_threshold, true_positive_upper_threshold

    def _true_negative_alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        true_negative_realized_chunk_performance = [
            self._true_negative_realized_performance(chunk.data) for chunk in reference_chunks
        ]
        deviation = np.std(true_negative_realized_chunk_performance) * std_num
        true_negative_mean_realized_performance = np.mean(true_negative_realized_chunk_performance)

        if self.normalize_confusion_matrix is None:
            true_negative_upper_threshold = true_negative_mean_realized_performance + deviation
        else:
            true_negative_upper_threshold = np.maximum(upper_limit, true_negative_mean_realized_performance + deviation)

        true_negative_lower_threshold = np.minimum(lower_limit, true_negative_mean_realized_performance - deviation)

        return true_negative_lower_threshold, true_negative_upper_threshold

    def _false_positive_alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        false_positive_realized_chunk_performance = [
            self._false_positive_realized_performance(chunk.data) for chunk in reference_chunks
        ]
        deviation = np.std(false_positive_realized_chunk_performance) * std_num
        false_positive_mean_realized_performance = np.mean(false_positive_realized_chunk_performance)

        if self.normalize_confusion_matrix is None:
            false_positive_upper_threshold = false_positive_mean_realized_performance + deviation
        else:
            false_positive_upper_threshold = np.maximum(
                upper_limit, false_positive_mean_realized_performance + deviation
            )

        false_positive_lower_threshold = np.minimum(lower_limit, false_positive_mean_realized_performance - deviation)

        return false_positive_lower_threshold, false_positive_upper_threshold

    def _false_negative_alert_thresholds(
        self, reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
    ) -> Tuple[float, float]:
        false_negative_realized_chunk_performance = [
            self._false_negative_realized_performance(chunk.data) for chunk in reference_chunks
        ]
        deviation = np.std(false_negative_realized_chunk_performance) * std_num
        false_negative_mean_realized_performance = np.mean(false_negative_realized_chunk_performance)

        if self.normalize_confusion_matrix is None:
            false_negative_upper_threshold = false_negative_mean_realized_performance + deviation
        else:
            false_negative_upper_threshold = np.maximum(
                upper_limit, false_negative_mean_realized_performance + deviation
            )

        false_negative_lower_threshold = np.minimum(lower_limit, false_negative_mean_realized_performance - deviation)

        return false_negative_lower_threshold, false_negative_upper_threshold

    def _true_positive_realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        num_tp = np.sum(np.logical_and(y_pred, y_true))
        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

        if self.normalize_confusion_matrix is None:
            return num_tp
        elif self.normalize_confusion_matrix == 'true':
            return num_tp / (num_tp + num_fn)
        elif self.normalize_confusion_matrix == 'pred':
            return num_tp / (num_tp + num_fp)
        else:  # normalization is 'all'
            return num_tp / len(y_true)

    def _true_negative_realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

        if self.normalize_confusion_matrix is None:
            return num_tn
        elif self.normalize_confusion_matrix == 'true':
            return num_tn / (num_tn + num_fp)
        elif self.normalize_confusion_matrix == 'pred':
            return num_tn / (num_tn + num_fn)
        else:
            return num_tn / len(y_true)

    def _false_positive_realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        num_tp = np.sum(np.logical_and(y_pred, y_true))
        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))

        if self.normalize_confusion_matrix is None:
            return num_fp
        elif self.normalize_confusion_matrix == 'true':
            return num_fp / (num_fp + num_tn)
        elif self.normalize_confusion_matrix == 'pred':
            return num_fp / (num_fp + num_tp)
        else:
            return num_fp / len(y_true)

    def _false_negative_realized_performance(self, data: pd.DataFrame) -> float:
        _, y_pred, y_true = self._common_cleaning(data, y_pred_proba_column_name=self.uncalibrated_y_pred_proba)

        if y_true is None:
            return np.NaN

        num_tp = np.sum(np.logical_and(y_pred, y_true))
        num_tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
        num_fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

        if self.normalize_confusion_matrix is None:
            return num_fn
        elif self.normalize_confusion_matrix == 'true':
            return num_fn / (num_fn + num_tp)
        elif self.normalize_confusion_matrix == 'pred':
            return num_fn / (num_fn + num_tn)
        else:
            return num_fn / len(y_true)

    def _true_positive_confidence_deviation(self, reference_chunks: List[Chunk]) -> float:
        return np.std([self.get_true_positive_estimate(chunk.data) for chunk in reference_chunks])

    def _true_negative_confidence_deviation(self, reference_chunks: List[Chunk]) -> float:
        return np.std([self.get_true_negative_estimate(chunk.data) for chunk in reference_chunks])

    def _false_positive_confidence_deviation(self, reference_chunks: List[Chunk]) -> float:
        return np.std([self.get_false_positive_estimate(chunk.data) for chunk in reference_chunks])

    def _false_negative_confidence_deviation(self, reference_chunks: List[Chunk]) -> float:
        return np.std([self.get_false_negative_estimate(chunk.data) for chunk in reference_chunks])

    def get_true_positive_estimate(self, chunk_data: pd.DataFrame) -> float:
        y_pred_proba = chunk_data[self.y_pred_proba]
        y_pred = chunk_data[self.y_pred]

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

        return normalized_est_tp_ratio

    def get_true_negative_estimate(self, chunk_data: pd.DataFrame) -> float:
        y_pred_proba = chunk_data[self.y_pred_proba]
        y_pred = chunk_data[self.y_pred]

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

        return normalized_est_tn_ratio

    def get_false_positive_estimate(self, chunk_data: pd.DataFrame) -> float:
        y_pred_proba = chunk_data[self.y_pred_proba]
        y_pred = chunk_data[self.y_pred]

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

        return normalized_est_fp_ratio

    def get_false_negative_estimate(self, chunk_data: pd.DataFrame) -> float:
        y_pred_proba = chunk_data[self.y_pred_proba]
        y_pred = chunk_data[self.y_pred]

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

        return normalized_est_fn_ratio

    def get_true_pos_info(self, chunk_data: pd.DataFrame) -> Dict:
        true_pos_info = {}

        estimated_true_positives = self.get_true_positive_estimate(chunk_data)

        sampling_error_true_positives = bse.true_positive_sampling_error(
            self._true_positive_sampling_error_components, chunk_data
        )

        true_pos_info['estimated_true_positive'] = estimated_true_positives
        true_pos_info['sampling_error_true_positive'] = sampling_error_true_positives
        true_pos_info['realized_true_positive'] = self._true_positive_realized_performance(chunk_data)

        if self.normalize_confusion_matrix is None:
            true_pos_info['upper_confidence_boundary_true_positive'] = (
                estimated_true_positives + SAMPLING_ERROR_RANGE * sampling_error_true_positives
            )
        else:
            true_pos_info['upper_confidence_boundary_true_positive'] = min(
                self.confidence_upper_bound,
                estimated_true_positives + SAMPLING_ERROR_RANGE * sampling_error_true_positives,
            )

        true_pos_info['lower_confidence_boundary_true_positive'] = max(
            self.confidence_lower_bound, estimated_true_positives - SAMPLING_ERROR_RANGE * sampling_error_true_positives
        )

        true_pos_info['upper_threshold_true_positive'] = self.true_positive_upper_threshold
        true_pos_info['lower_threshold_true_positive'] = self.true_positive_lower_threshold

        true_pos_info['alert_true_positive'] = (
            estimated_true_positives > self.true_positive_upper_threshold
            or estimated_true_positives < self.true_positive_lower_threshold
        )

        return true_pos_info

    def get_true_neg_info(self, chunk_data: pd.DataFrame) -> Dict:
        true_neg_info = {}

        estimated_true_negatives = self.get_true_negative_estimate(chunk_data)

        sampling_error_true_negatives = bse.true_negative_sampling_error(
            self._true_negative_sampling_error_components, chunk_data
        )

        true_neg_info['estimated_true_negative'] = estimated_true_negatives
        true_neg_info['sampling_error_true_negative'] = sampling_error_true_negatives
        true_neg_info['realized_true_negative'] = self._true_negative_realized_performance(chunk_data)

        if self.normalize_confusion_matrix is None:
            true_neg_info['upper_confidence_boundary_true_negative'] = (
                estimated_true_negatives + SAMPLING_ERROR_RANGE * sampling_error_true_negatives
            )
        else:
            true_neg_info['upper_confidence_boundary_true_negative'] = min(
                self.confidence_upper_bound,
                estimated_true_negatives + SAMPLING_ERROR_RANGE * sampling_error_true_negatives,
            )

        true_neg_info['lower_confidence_boundary_true_negative'] = max(
            self.confidence_lower_bound, estimated_true_negatives - SAMPLING_ERROR_RANGE * sampling_error_true_negatives
        )

        true_neg_info['upper_threshold_true_negative'] = self.true_negative_upper_threshold
        true_neg_info['lower_threshold_true_negative'] = self.true_negative_lower_threshold

        true_neg_info['alert_true_negative'] = (
            estimated_true_negatives > self.true_negative_upper_threshold
            or estimated_true_negatives < self.true_negative_lower_threshold
        )

        return true_neg_info

    def get_false_pos_info(self, chunk_data: pd.DataFrame) -> Dict:
        false_pos_info = {}

        estimated_false_positives = self.get_false_positive_estimate(chunk_data)

        sampling_error_false_positives = bse.false_positive_sampling_error(
            self._false_positive_sampling_error_components, chunk_data
        )

        false_pos_info['estimated_false_positive'] = estimated_false_positives
        false_pos_info['sampling_error_false_positive'] = sampling_error_false_positives
        false_pos_info['realized_false_positive'] = self._false_positive_realized_performance(chunk_data)

        if self.normalize_confusion_matrix is None:
            false_pos_info['upper_confidence_boundary_false_positive'] = (
                estimated_false_positives + SAMPLING_ERROR_RANGE * sampling_error_false_positives
            )
        else:
            false_pos_info['upper_confidence_boundary_false_positive'] = min(
                self.confidence_upper_bound,
                estimated_false_positives + SAMPLING_ERROR_RANGE * sampling_error_false_positives,
            )

        false_pos_info['lower_confidence_boundary_false_positive'] = max(
            self.confidence_lower_bound,
            estimated_false_positives - SAMPLING_ERROR_RANGE * sampling_error_false_positives,
        )

        false_pos_info['upper_threshold_false_positive'] = self.false_positive_upper_threshold
        false_pos_info['lower_threshold_false_positive'] = self.false_positive_lower_threshold

        false_pos_info['alert_false_positive'] = (
            estimated_false_positives > self.false_positive_upper_threshold
            or estimated_false_positives < self.false_positive_lower_threshold
        )

        return false_pos_info

    def get_false_neg_info(self, chunk_data: pd.DataFrame) -> Dict:
        false_neg_info = {}

        estimated_false_negatives = self.get_false_negative_estimate(chunk_data)

        sampling_error_false_negatives = bse.false_negative_sampling_error(
            self._false_negative_sampling_error_components, chunk_data
        )

        false_neg_info['estimated_false_negative'] = estimated_false_negatives
        false_neg_info['sampling_error_false_negative'] = sampling_error_false_negatives
        false_neg_info['realized_false_negative'] = self._false_negative_realized_performance(chunk_data)

        if self.normalize_confusion_matrix is None:
            false_neg_info['upper_confidence_boundary_false_negative'] = (
                estimated_false_negatives + SAMPLING_ERROR_RANGE * sampling_error_false_negatives
            )
        else:
            false_neg_info['upper_confidence_boundary_false_negative'] = min(
                self.confidence_upper_bound,
                estimated_false_negatives + SAMPLING_ERROR_RANGE * sampling_error_false_negatives,
            )

        false_neg_info['lower_confidence_boundary_false_negative'] = max(
            self.confidence_lower_bound,
            estimated_false_negatives - SAMPLING_ERROR_RANGE * sampling_error_false_negatives,
        )

        false_neg_info['upper_threshold_false_negative'] = self.false_negative_upper_threshold
        false_neg_info['lower_threshold_false_negative'] = self.false_negative_lower_threshold

        false_neg_info['alert_false_negative'] = (
            estimated_false_negatives > self.false_negative_upper_threshold
            or estimated_false_negatives < self.false_negative_lower_threshold
        )

        return false_neg_info

    def get_chunk_record(self, chunk_data: pd.DataFrame) -> Dict:
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
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='roc_auc',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
            components=[('ROC AUC', 'roc_auc')],
        )
        # FIXME: Should we check the y_pred_proba argument here to ensure it's a dict?
        self.y_pred_proba: Dict[str, str]

        # sampling error
        self._sampling_error_components: List[Tuple] = []

    def _fit(self, reference_data: pd.DataFrame):
        classes = class_labels(self.y_pred_proba)
        binarized_y_true = list(label_binarize(reference_data[self.y_true], classes=classes).T)
        y_pred_proba = [reference_data[self.y_pred_proba[clazz]].T for clazz in classes]

        self._sampling_error_components = mse.auroc_sampling_error_components(
            y_true_reference=binarized_y_true, y_pred_proba_reference=y_pred_proba
        )

    def _estimate(self, data: pd.DataFrame):
        _, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred_proba_class in y_pred_probas:
            ovr_estimates.append(estimate_roc_auc(y_pred_proba_class))
        multiclass_roc_auc = np.mean(ovr_estimates)
        return multiclass_roc_auc

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.auroc_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        if self.y_true not in data.columns or data[self.y_true].isna().all():
            return np.NaN

        y_true = data[self.y_true]
        _, y_pred_probas, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return roc_auc_score(y_true, y_pred_probas, multi_class='ovr', average='macro', labels=labels)


@MetricFactory.register('f1', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationF1(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='f1',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
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

    def _estimate(self, data: pd.DataFrame):
        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_f1(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.f1_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        if self.y_true not in data.columns or data[self.y_true].isna().all():
            return np.NaN

        y_true = data[self.y_true]
        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('precision', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationPrecision(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='precision',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
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

    def _estimate(self, data: pd.DataFrame):
        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_precision(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.precision_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        if self.y_true not in data.columns or data[self.y_true].isna().all():
            return np.NaN

        y_true = data[self.y_true]
        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('recall', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationRecall(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='recall',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
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

    def _estimate(self, data: pd.DataFrame):
        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_recall(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.recall_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        if self.y_true not in data.columns or data[self.y_true].isna().all():
            return np.NaN

        y_true = data[self.y_true]
        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


@MetricFactory.register('specificity', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationSpecificity(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='specificity',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
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

    def _estimate(self, data: pd.DataFrame):
        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        ovr_estimates = []
        for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
            ovr_estimates.append(estimate_specificity(y_pred, y_pred_proba))
        multiclass_metric = np.mean(ovr_estimates)

        return multiclass_metric

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.specificity_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        if self.y_true not in data.columns or data[self.y_true].isna().all():
            return np.NaN

        y_true = data[self.y_true]
        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        tn_sum = mcm[:, 0, 0]
        fp_sum = mcm[:, 0, 1]
        class_wise_specificity = tn_sum / (tn_sum + fp_sum)
        return np.mean(class_wise_specificity)


@MetricFactory.register('accuracy', ProblemType.CLASSIFICATION_MULTICLASS)
class MulticlassClassificationAccuracy(Metric):
    def __init__(
        self,
        y_pred_proba: ModelOutputsType,
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        timestamp_column_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            name='accuracy',
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            chunker=chunker,
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

    def _estimate(self, data: pd.DataFrame):
        y_preds, y_pred_probas, _ = _get_binarized_multiclass_predictions(data, self.y_pred, self.y_pred_proba)
        y_preds_array = np.asarray(y_preds).T
        y_pred_probas_array = np.asarray(y_pred_probas).T
        probability_of_predicted = np.max(y_preds_array * y_pred_probas_array, axis=1)
        return np.mean(probability_of_predicted)

    def _sampling_error(self, data: pd.DataFrame) -> float:
        return mse.accuracy_sampling_error(self._sampling_error_components, data)

    def _realized_performance(self, data: pd.DataFrame) -> float:
        if self.y_true not in data.columns or data[self.y_true].isna().all():
            return np.NaN

        y_true = data[self.y_true]
        y_pred, _, labels = _get_multiclass_uncalibrated_predictions(data, self.y_pred, self.y_pred_proba)

        return accuracy_score(y_true, y_pred)
