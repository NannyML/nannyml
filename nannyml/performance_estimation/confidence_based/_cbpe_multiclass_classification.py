#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from nannyml._typing import ModelOutputsType, model_output_column_names
from nannyml.base import _list_missing
from nannyml.calibration import Calibrator, NoopCalibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based._cbpe_binary_classification import (
    _estimate_f1 as estimate_binary_f1,
)
from nannyml.performance_estimation.confidence_based._cbpe_binary_classification import (
    _estimate_precision as estimate_binary_precision,
)
from nannyml.performance_estimation.confidence_based._cbpe_binary_classification import (
    _estimate_recall as estimate_binary_recall,
)
from nannyml.performance_estimation.confidence_based._cbpe_binary_classification import (
    _estimate_roc_auc as estimate_binary_roc_auc,
)
from nannyml.performance_estimation.confidence_based._cbpe_binary_classification import (
    _estimate_specificity as estimate_binary_specificity,
)
from nannyml.performance_estimation.confidence_based.results import (
    SUPPORTED_METRIC_VALUES,
    CBPEPerformanceEstimatorResult,
)


class _MulticlassClassificationCBPE(CBPE):
    def __init__(
        self,
        metrics: List[str],
        y_pred: str,
        y_pred_proba: ModelOutputsType,
        y_true: str,
        timestamp_column_name: str,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        calibration: str = None,
        calibrator: Calibrator = None,
    ):
        """Creates a new CBPE performance estimator."""
        super().__init__(
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            y_true=y_true,
            timestamp_column_name=timestamp_column_name,
            metrics=metrics,
            chunk_size=chunk_size,
            chunk_number=chunk_number,
            chunk_period=chunk_period,
            chunker=chunker,
            calibration=calibration,
            calibrator=calibrator,
        )

        if not isinstance(y_pred_proba, Dict):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type '{type(y_pred_proba)}'. "
                "Binary use cases require 'y_pred_proba' to be a dictionary mapping "
                "class labels to column names."
            )
        self.y_pred_proba: Dict[str, str] = y_pred_proba

        self._calibrators: Dict[str, Calibrator] = {}
        self.confidence_upper_bound = 1
        self.confidence_lower_bound = 0

        self.previous_reference_results: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> CBPE:
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_true, self.y_pred] + model_output_column_names(self.y_pred_proba), reference_data)

        reference_chunks = self.chunker.split(
            reference_data, minimum_chunk_size=300, timestamp_column_name=self.timestamp_column_name
        )

        self._alert_thresholds = _calculate_alert_thresholds(
            reference_chunks,
            self.metrics,
            self.y_true,
            self.y_pred,
            self.y_pred_proba,
        )

        self._confidence_deviations = _calculate_confidence_deviations(
            reference_chunks, self.y_pred, self.y_pred_proba, metrics=self.metrics
        )

        self.minimum_chunk_size = 300

        self._calibrators = _fit_calibrators(reference_data, self.y_true, self.y_pred_proba, self.calibrator)

        self.previous_reference_results = self._estimate(reference_data).data
        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> CBPEPerformanceEstimatorResult:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred] + model_output_column_names(self.y_pred_proba), data)

        data = _calibrate_predicted_probabilities(data, self.y_true, self.y_pred_proba, self._calibrators)

        chunks = self.chunker.split(data, minimum_chunk_size=300, timestamp_column_name=self.timestamp_column_name)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    **self.__estimate(chunk, self.y_true, self.y_pred, self.y_pred_proba),
                }
                for chunk in chunks
            ]
        )

        res = res.reset_index(drop=True)
        return CBPEPerformanceEstimatorResult(results_data=res, estimator=self)

    def __estimate(self, chunk: Chunk, y_true: str, y_pred: str, y_pred_proba: Dict[str, str]) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = _estimate_metric(
                data=chunk.data, y_pred=y_pred, y_pred_proba=y_pred_proba, metric=metric
            )
            estimates[f'realized_{metric}'] = _calculate_realized_performance(
                chunk, y_true, y_pred, y_pred_proba, metric
            )
            estimates[f'estimated_{metric}'] = estimated_metric
            estimates[f'upper_confidence_{metric}'] = min(
                self.confidence_upper_bound, estimated_metric + self._confidence_deviations[metric]
            )
            estimates[f'lower_confidence_{metric}'] = max(
                self.confidence_lower_bound, estimated_metric - self._confidence_deviations[metric]
            )
            estimates[f'upper_threshold_{metric}'] = self._alert_thresholds[metric][0]
            estimates[f'lower_threshold_{metric}'] = self._alert_thresholds[metric][1]
            estimates[f'alert_{metric}'] = (
                estimated_metric > self._alert_thresholds[metric][1]
                or estimated_metric < self._alert_thresholds[metric][0]
            )
        return estimates


def _get_predictions(data: pd.DataFrame, y_pred: str, y_pred_proba: Dict[str, str]):
    classes = sorted(y_pred_proba.keys())
    y_preds = list(label_binarize(data[y_pred], classes=classes).T)

    y_pred_probas = [data[y_pred_proba[clazz]] for clazz in classes]
    return y_preds, y_pred_probas


def _estimate_metric(data: pd.DataFrame, y_pred: str, y_pred_proba: Dict[str, str], metric: str) -> float:
    if metric == 'roc_auc':
        return _estimate_roc_auc(data[[v for k, v in y_pred_proba.items()]])
    elif metric == 'f1':
        y_preds, y_pred_probas = _get_predictions(data, y_pred, y_pred_proba)
        return _estimate_f1(y_preds, y_pred_probas)
    elif metric == 'precision':
        y_preds, y_pred_probas = _get_predictions(data, y_pred, y_pred_proba)
        return _estimate_precision(y_preds, y_pred_probas)
    elif metric == 'recall':
        y_preds, y_pred_probas = _get_predictions(data, y_pred, y_pred_proba)
        return _estimate_recall(y_preds, y_pred_probas)
    elif metric == 'specificity':
        y_preds, y_pred_probas = _get_predictions(data, y_pred, y_pred_proba)
        return _estimate_specificity(y_preds, y_pred_probas)
    elif metric == 'accuracy':
        y_preds, y_pred_probas = _get_predictions(data, y_pred, y_pred_proba)
        return _estimate_accuracy(y_preds, y_pred_probas)
    else:
        raise InvalidArgumentsException(
            f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
        )


def _estimate_roc_auc(y_pred_probas: pd.DataFrame) -> float:
    ovr_roc_auc_estimates = []
    for y_pred_proba_class in [y_pred_probas[col] for col in y_pred_probas]:
        ovr_roc_auc_estimates.append(estimate_binary_roc_auc(y_pred_proba_class))
    multiclass_roc_auc = np.mean(ovr_roc_auc_estimates)

    return multiclass_roc_auc


def _estimate_f1(y_preds: List[np.ndarray], y_pred_probas: List[np.ndarray]) -> float:
    ovr_f1_estimates = []
    for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
        ovr_f1_estimates.append(estimate_binary_f1(y_pred, y_pred_proba))
    multiclass_metric = np.mean(ovr_f1_estimates)

    return multiclass_metric


def _estimate_precision(y_preds: List[np.ndarray], y_pred_probas: List[np.ndarray]) -> float:
    ovr_precision_estimates = []
    for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
        ovr_precision_estimates.append(estimate_binary_precision(y_pred, y_pred_proba))
    multiclass_metric = np.mean(ovr_precision_estimates)

    return multiclass_metric


def _estimate_recall(y_preds: List[np.ndarray], y_pred_probas: List[np.ndarray]) -> float:
    ovr_recall_estimates = []
    for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
        ovr_recall_estimates.append(estimate_binary_recall(y_pred, y_pred_proba))

    multiclass_metric = np.mean(ovr_recall_estimates)

    return multiclass_metric


def _estimate_specificity(y_preds: List[np.ndarray], y_pred_probas: List[np.ndarray]) -> float:
    ovr_specificity_estimates = []
    for y_pred, y_pred_proba in zip(y_preds, y_pred_probas):
        ovr_specificity_estimates.append(estimate_binary_specificity(y_pred, y_pred_proba))

    multiclass_metric = np.mean(ovr_specificity_estimates)

    return multiclass_metric


def _estimate_accuracy(y_preds: List[np.ndarray], y_pred_probas: List[np.ndarray]) -> float:
    y_preds_array = np.asarray(y_preds).T
    y_pred_probas_array = np.asarray(y_pred_probas).T
    probability_of_predicted = np.max(y_preds_array * y_pred_probas_array, axis=1)
    metric = np.mean(probability_of_predicted)
    return metric


def _calculate_alert_thresholds(
    reference_chunks: List[Chunk],
    metrics: List[str],
    y_true: str,
    y_pred: str,
    y_pred_proba: Dict[str, str],
    std_num: int = 3,
    lower_limit: int = 0,
    upper_limit: int = 1,
) -> Dict[str, Tuple[float, float]]:

    alert_thresholds = {}
    for metric in metrics:
        realised_performance_chunks = [
            _calculate_realized_performance(chunk, y_true, y_pred, y_pred_proba, metric) for chunk in reference_chunks
        ]
        deviation = np.std(realised_performance_chunks) * std_num
        mean_realised_performance = np.mean(realised_performance_chunks)
        lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
        upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

        alert_thresholds[metric] = (lower_threshold, upper_threshold)
    return alert_thresholds


def _calculate_realized_performance(chunk: Chunk, y_true: str, y_pred: str, y_pred_proba: Dict[str, str], metric: str):
    if y_true not in chunk.data.columns or chunk.data[y_true].isna().all():
        return np.NaN

    # Make sure labels and class_probability_columns have the same ordering
    labels, class_probability_columns = [], []
    for label in sorted(y_pred_proba.keys()):
        labels.append(label)
        class_probability_columns.append(y_pred_proba[label])

    y_true = chunk.data[y_true]
    y_pred_probas = chunk.data[class_probability_columns]
    y_pred = chunk.data[y_pred]

    if metric == 'roc_auc':
        return roc_auc_score(y_true, y_pred_probas, multi_class='ovr', average='macro', labels=labels)
    elif metric == 'f1':
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)
    elif metric == 'precision':
        return precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)
    elif metric == 'recall':
        return recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)
    elif metric == 'specificity':
        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        tn_sum = mcm[:, 0, 0]
        fp_sum = mcm[:, 0, 1]
        class_wise_specificity = tn_sum / (tn_sum + fp_sum)
        return np.mean(class_wise_specificity)
    elif metric == 'accuracy':
        return accuracy_score(y_true=y_true, y_pred=y_pred)
    else:
        raise InvalidArgumentsException(
            f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
        )


def _calculate_confidence_deviations(
    reference_chunks: List[Chunk], y_pred: str, y_pred_proba: Dict[str, str], metrics: List[str]
):
    return {
        metric: np.std([_estimate_metric(chunk.data, y_pred, y_pred_proba, metric) for chunk in reference_chunks])
        for metric in metrics
    }


def _get_class_splits(
    data: pd.DataFrame, y_true: str, y_pred_proba: Dict[str, str], include_targets: bool = True
) -> List[Tuple]:
    classes = sorted(y_pred_proba.keys())
    y_trues: List[np.ndarray] = []

    if include_targets:
        y_trues = list(label_binarize(data[y_true], classes=classes).T)

    y_pred_probas = [data[y_pred_proba[clazz]] for clazz in classes]

    return [
        (classes[idx], y_trues[idx] if include_targets else None, y_pred_probas[idx]) for idx in range(len(classes))
    ]


def _fit_calibrators(
    reference_data: pd.DataFrame, y_true_col: str, y_pred_proba_col: Dict[str, str], calibrator: Calibrator
) -> Dict[str, Calibrator]:
    fitted_calibrators = {}
    noop_calibrator = NoopCalibrator()

    for clazz, y_true, y_pred_proba in _get_class_splits(reference_data, y_true_col, y_pred_proba_col):
        if not needs_calibration(np.asarray(y_true), np.asarray(y_pred_proba), calibrator):
            calibrator = noop_calibrator

        calibrator.fit(y_pred_proba, y_true)
        fitted_calibrators[clazz] = deepcopy(calibrator)

    return fitted_calibrators


def _calibrate_predicted_probabilities(
    data: pd.DataFrame, y_true: str, y_pred_proba: Dict[str, str], calibrators: Dict[str, Calibrator]
) -> pd.DataFrame:
    class_splits = _get_class_splits(data, y_true, y_pred_proba, include_targets=False)
    number_of_observations = len(data)
    number_of_classes = len(class_splits)

    calibrated_probas = np.zeros((number_of_observations, number_of_classes))

    for idx, split in enumerate(class_splits):
        clazz, _, y_pred_proba_zz = split
        calibrated_probas[:, idx] = calibrators[clazz].calibrate(y_pred_proba_zz)

    denominator = np.sum(calibrated_probas, axis=1)[:, np.newaxis]
    uniform_proba = np.full_like(calibrated_probas, 1 / number_of_classes)

    calibrated_probas = np.divide(calibrated_probas, denominator, out=uniform_proba, where=denominator != 0)

    calibrated_data = data.copy(deep=True)
    predicted_class_proba_column_names = sorted([v for k, v in y_pred_proba.items()])
    for idx in range(number_of_classes):
        calibrated_data[predicted_class_proba_column_names[idx]] = calibrated_probas[:, idx]

    return calibrated_data
