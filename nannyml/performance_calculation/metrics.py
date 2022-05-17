#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing metric utilities and implementations."""
import abc
from typing import Type  # noqa: TYP001
from typing import Dict, List, Tuple

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

from nannyml import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.metadata.base import (
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_REFERENCE_PARTITION_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
    ModelMetadata,
)
from nannyml.metadata.binary_classification import (
    NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    NML_METADATA_PREDICTION_COLUMN_NAME,
    BinaryClassificationMetadata,
)
from nannyml.metadata.multiclass_classification import MulticlassClassificationMetadata


class Metric(abc.ABC):
    """Represents a performance metric."""

    def __init__(
        self,
        display_name: str,
        column_name: str,
        metadata: ModelMetadata,
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
        metadata: ModelMetadata
            Metadata describing the model being monitored.
        upper_threshold : float, default=None
            An optional upper threshold for the performance metric.
        lower_threshold : float, default=None
            An optional lower threshold for the performance metric.
        """
        self.display_name = display_name
        self.column_name = column_name
        self.metadata = metadata
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        self._minimum_chunk_size: int = 300

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
            reference_chunks = chunker.split(reference_data, minimum_chunk_size=self.minimum_chunk_size())
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
        if NML_METADATA_TARGET_COLUMN_NAME not in data.columns:
            raise RuntimeError('data does not contain target column')

        if (
            NML_METADATA_PREDICTION_COLUMN_NAME not in data.columns
            and NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME not in data.columns
        ):
            raise RuntimeError('data does contains neither prediction column or predicted probabilities column')

        return self._calculate(data)

    def _calculate(self, data: pd.DataFrame):
        raise NotImplementedError

    def minimum_chunk_size(self) -> int:
        """Determines the minimum number of observations a chunk should ideally for this metric to be trustworthy."""
        try:
            return self._minimum_chunk_size
        except Exception:
            # TODO: log failure
            return 300

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


def _floor_chunk_size(calculated_min_chunk_size: float, lower_limit_on_chunk_size: int = 300) -> int:
    return int(np.maximum(calculated_min_chunk_size, lower_limit_on_chunk_size))


def _minimum_chunk_size_roc_auc(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    predicted_probability_column_name: str = NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    required_std: float = 0.02,
) -> int:
    """Estimation of minimum sample size to get required standard deviation of AUROC.

    Estimation takes advantage of Standard Error of the Mean formula and expressing AUROC as Mann-Whitney U statistic.
    """
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred_proba = data.loc[
        data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, predicted_probability_column_name
    ]

    y_true, y_pred_proba = np.asarray(y_true), np.asarray(y_pred_proba)
    if np.mean(y_true) > 0.5:
        y_true = abs(np.asarray(y_true) - 1)
        y_pred_proba = 1 - y_pred_proba

    sorted_idx = np.argsort(y_pred_proba)
    y_pred_proba = y_pred_proba[sorted_idx]
    y_true = y_true[sorted_idx]
    rank_order = np.asarray(range(len(y_pred_proba)))
    positive_ranks = y_true * rank_order
    indexes = np.unique(positive_ranks)[1:]
    ser = []

    for i, index in enumerate(indexes):
        ser.append(index - i)

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    ser_divided = ser / (n_pos * n_neg)
    ser_multi = ser_divided * n_pos

    pos_targets = y_true
    # neg_targets = abs(y_true - 1)

    n_pos_targets = np.sum(pos_targets)

    fraction = n_pos_targets / len(y_true)
    sample_size = (np.std(ser_multi)) ** 2 / ((required_std**2) * fraction)
    sample_size = np.minimum(sample_size, len(y_true))
    sample_size = np.round(sample_size, -2)

    return _floor_chunk_size(sample_size)


def _minimum_chunk_size_f1(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    prediction_column_name: str = NML_METADATA_PREDICTION_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    required_std: float = 0.02,
):
    """Estimation of minimum sample size to get required standard deviation of F1.

    Estimation takes advantage of Standard Error of the Mean formula.
    """
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, prediction_column_name]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    TP = np.where((y_true == y_pred) & (y_pred == 1), 1, np.nan)
    FP = np.where((y_true != y_pred) & (y_pred == 1), 0, np.nan)
    FN = np.where((y_true != y_pred) & (y_pred == 0), 0, np.nan)

    TP = TP[~np.isnan(TP)]
    FN = FN[~np.isnan(FN)]
    FP = FP[~np.isnan(FP)]

    tp_fp_fn = np.concatenate([TP, FN, FP])

    correcting_factor = len(tp_fp_fn) / ((len(FN) + len(FP)) * 0.5 + len(TP))
    obs_level_f1 = tp_fp_fn * correcting_factor
    fraction_of_relevant = len(tp_fp_fn) / len(y_pred)
    sample_size = ((np.std(obs_level_f1)) ** 2) / ((required_std**2) * fraction_of_relevant)
    sample_size = np.minimum(sample_size, len(y_true))
    sample_size = np.round(sample_size, -2)

    return _floor_chunk_size(sample_size)


def _minimum_chunk_size_precision(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    prediction_column_name: str = NML_METADATA_PREDICTION_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    required_std: float = 0.02,
):
    """Estimation of minimum sample size to get required standard deviation of Precision.

    Estimation takes advantage of Standard Error of the Mean formula.
    """
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, prediction_column_name]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    TP = np.where((y_true == y_pred) & (y_pred == 1), 1, np.nan)
    FP = np.where((y_true != y_pred) & (y_pred == 1), 0, np.nan)

    TP = TP[~np.isnan(TP)]
    FP = FP[~np.isnan(FP)]
    obs_level_precision = np.concatenate([TP, FP])
    amount_positive_pred = np.sum(y_pred)
    fraction_of_pos_pred = amount_positive_pred / len(y_pred)
    sample_size = ((np.std(obs_level_precision)) ** 2) / ((required_std**2) * fraction_of_pos_pred)
    sample_size = np.minimum(sample_size, len(y_true))
    sample_size = np.round(sample_size, -2)

    return _floor_chunk_size(sample_size)


def _minimum_chunk_size_recall(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    prediction_column_name: str = NML_METADATA_PREDICTION_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    required_std: float = 0.02,
):
    """Estimation of minimum sample size to get required standard deviation of Recall.

    Estimation takes advantage of Standard Error of the Mean formula.
    """
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, prediction_column_name]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    TP = np.where((y_true == y_pred) & (y_pred == 1), 1, np.nan)
    FN = np.where((y_true != y_pred) & (y_pred == 0), 0, np.nan)
    TP = TP[~np.isnan(TP)]
    FN = FN[~np.isnan(FN)]

    obs_level_recall = np.concatenate([TP, FN])
    fraction_of_relevant = sum(obs_level_recall) / len(y_pred)

    sample_size = ((np.std(obs_level_recall)) ** 2) / ((required_std**2) * fraction_of_relevant)
    sample_size = np.minimum(sample_size, len(y_true))
    sample_size = np.round(sample_size, -2)

    return _floor_chunk_size(sample_size)


def _minimum_chunk_size_specificity(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    prediction_column_name: str = NML_METADATA_PREDICTION_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    required_std: float = 0.02,
):
    """Estimation of minimum sample size to get required standard deviation of Specificity.

    Estimation takes advantage of Standard Error of the Mean formula.
    """
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, prediction_column_name]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    TN = np.where((y_true == y_pred) & (y_pred == 0), 1, np.nan)
    FP = np.where((y_true != y_pred) & (y_pred == 1), 0, np.nan)
    TN = TN[~np.isnan(TN)]
    FP = FP[~np.isnan(FP)]

    obs_level_specificity = np.concatenate([TN, FP])
    fraction_of_relevant = len(obs_level_specificity) / len(y_pred)
    sample_size = ((np.std(obs_level_specificity)) ** 2) / ((required_std**2) * fraction_of_relevant)
    sample_size = np.minimum(sample_size, len(y_true))
    sample_size = np.round(sample_size, -2)

    return _floor_chunk_size(sample_size)


def _minimum_chunk_size_accuracy(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    prediction_column_name: str = NML_METADATA_PREDICTION_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
    required_std: float = 0.02,
):
    """Estimation of minimum sample size to get required standard deviation of Accuracy.

    Estimation takes advantage of Standard Error of the Mean formula.
    """
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, prediction_column_name]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true = np.asarray(y_true).astype(int)

    y_pred = np.asarray(y_pred).astype(int)
    correct_table = (y_true == y_pred).astype(int)
    sample_size = (np.std(correct_table) ** 2) / (required_std**2)
    sample_size = np.minimum(sample_size, len(y_true))
    sample_size = np.round(sample_size, -2)

    return _floor_chunk_size(sample_size)


class BinaryClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new AUROC instance."""
        super().__init__(display_name='ROC AUC', column_name='roc_auc', metadata=metadata)

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_chunk_size_roc_auc(reference_data)

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]  # TODO: this should be predicted_probabilities

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred)


class BinaryClassificationF1(Metric):
    """F1 score metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new F1 instance."""
        super().__init__(display_name='F1', column_name='f1', metadata=metadata)

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_chunk_size_f1(reference_data)

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred)


class BinaryClassificationPrecision(Metric):
    """Precision metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Precision instance."""
        super().__init__(display_name='Precision', column_name='precision', metadata=metadata)

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_chunk_size_precision(reference_data)

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred)


class BinaryClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Recall instance."""
        super().__init__(display_name='Recall', column_name='recall', metadata=metadata)

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_chunk_size_recall(reference_data)

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred)


class BinaryClassificationSpecificity(Metric):
    """Specificity metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new F1 instance."""
        super().__init__(display_name='Specificity', column_name='specificity', metadata=metadata)

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_chunk_size_specificity(reference_data)

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

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


class BinaryClassificationAccuracy(Metric):
    """Accuracy metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Accuracy instance."""
        super().__init__(display_name='Accuracy', column_name='accuracy', metadata=metadata)

    def _fit(self, reference_data: pd.DataFrame):
        self._min_chunk_size = _minimum_chunk_size_accuracy(reference_data)

    def _calculate(self, data: pd.DataFrame):
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

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


class MulticlassClassificationAUROC(Metric):
    """Area under Receiver Operating Curve metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new AUROC instance."""
        super().__init__(display_name='ROC AUC', column_name='roc_auc', metadata=metadata)
        self._min_chunk_size = 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        """Redefine to handle NaNs and edge cases."""
        if not isinstance(self.metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        labels, class_probability_columns = [], []
        for label in sorted(list(self.metadata.predicted_class_probability_metadata_columns())):
            labels.append(label)
            class_probability_columns.append(self.metadata.predicted_class_probability_metadata_columns()[label])

        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[class_probability_columns]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        # y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if y_true.nunique() <= 1:
            return np.nan
        else:
            return roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro', labels=labels)


class MulticlassClassificationF1(Metric):
    """F1 score metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new F1 instance."""
        super().__init__(display_name='F1', column_name='f1', metadata=metadata)
        self._min_chunk_size = 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        labels = sorted(list(self.metadata.predicted_class_probability_metadata_columns().keys()))
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        # y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return f1_score(y_true, y_pred, average='macro', labels=labels)


class MulticlassClassificationPrecision(Metric):
    """Precision metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Precision instance."""
        super().__init__(display_name='Precision', column_name='precision', metadata=metadata)
        self._min_chunk_size = 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        labels = sorted(list(self.metadata.predicted_class_probability_metadata_columns().keys()))
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        # y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return precision_score(y_true, y_pred, average='macro', labels=labels)


class MulticlassClassificationRecall(Metric):
    """Recall metric, also known as 'sensitivity'."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Recall instance."""
        super().__init__(display_name='Recall', column_name='recall', metadata=metadata)
        self._min_chunk_size = 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        labels = sorted(list(self.metadata.predicted_class_probability_metadata_columns().keys()))
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        # y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return recall_score(y_true, y_pred, average='macro', labels=labels)


class MulticlassClassificationSpecificity(Metric):
    """Specificity metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Specificity instance."""
        super().__init__(display_name='Specificity', column_name='specificity', metadata=metadata)
        self._min_chunk_size = 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        labels = sorted(list(self.metadata.predicted_class_probability_metadata_columns().keys()))
        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric {self.display_name}: " "prediction column contains no data"
            )

        # y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            MCM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
            tn_sum = MCM[:, 0, 0]
            fp_sum = MCM[:, 0, 1]
            class_wise_specificity = tn_sum / (tn_sum + fp_sum)
            return np.mean(class_wise_specificity)


class MulticlassClassificationAccuracy(Metric):
    """Accuracy metric."""

    def __init__(self, metadata: ModelMetadata):
        """Creates a new Accuracy instance."""
        super().__init__(display_name='Accuracy', column_name='accuracy', metadata=metadata)
        self._min_chunk_size = 300

    def _fit(self, reference_data: pd.DataFrame):
        pass

    def _calculate(self, data: pd.DataFrame):
        if not isinstance(self.metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        y_true = data[NML_METADATA_TARGET_COLUMN_NAME]
        y_pred = data[NML_METADATA_PREDICTION_COLUMN_NAME]

        if y_pred.isna().all().any():
            raise InvalidArgumentsException(
                f"could not calculate metric '{self.display_name}': " "prediction column contains no data"
            )

        # y_true, y_pred = _common_data_cleaning(y_true, y_pred)

        if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
            return np.nan
        else:
            return accuracy_score(y_true, y_pred)


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


class MetricFactory:
    """A factory class that produces Metric instances based on a given magic string or a metric specification."""

    _metrics: Dict[str, Dict[str, Type[Metric]]] = {
        'roc_auc': {
            BinaryClassificationMetadata.__name__: BinaryClassificationAUROC,
            MulticlassClassificationMetadata.__name__: MulticlassClassificationAUROC,
        },
        'f1': {
            BinaryClassificationMetadata.__name__: BinaryClassificationF1,
            MulticlassClassificationMetadata.__name__: MulticlassClassificationF1,
        },
        'precision': {
            BinaryClassificationMetadata.__name__: BinaryClassificationPrecision,
            MulticlassClassificationMetadata.__name__: MulticlassClassificationPrecision,
        },
        'recall': {
            BinaryClassificationMetadata.__name__: BinaryClassificationRecall,
            MulticlassClassificationMetadata.__name__: MulticlassClassificationRecall,
        },
        'specificity': {
            BinaryClassificationMetadata.__name__: BinaryClassificationSpecificity,
            MulticlassClassificationMetadata.__name__: MulticlassClassificationSpecificity,
        },
        'accuracy': {
            BinaryClassificationMetadata.__name__: BinaryClassificationAccuracy,
            MulticlassClassificationMetadata.__name__: MulticlassClassificationAccuracy,
        },
    }

    @classmethod
    def create(cls, key: str, metadata: ModelMetadata) -> Metric:
        """Returns a Metric instance for a given key."""
        if not isinstance(key, str):
            raise InvalidArgumentsException(
                f"cannot create metric given a '{type(key)}'" "Please provide a string, function or Metric"
            )

        if key not in cls._metrics:
            raise InvalidArgumentsException(
                f"unknown metric key '{key}' given. "
                "Should be one of ['roc_auc', 'f1', 'precision', 'recall', 'specificity', "
                "'accuracy']."
            )

        metadata_class_name = type(metadata).__name__
        if metadata_class_name not in cls._metrics[key]:
            raise RuntimeError(
                f"metric '{key}' is currently not supported for model type {metadata_class_name}. "
                "Please specify another metric or use one of these supported model types for this metric: "
                f"{[md for md in cls._metrics[key]]}"
            )
        metric_class = cls._metrics[key][metadata_class_name]
        return metric_class(metadata=metadata)  # type: ignore
