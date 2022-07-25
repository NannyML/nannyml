#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import PolynomialFeatures

from nannyml._typing import ModelOutputsType
from nannyml.calibration import Calibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics import _list_missing
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based.results import (
    SUPPORTED_METRIC_VALUES,
    CBPEPerformanceEstimatorResult,
)


class _BinaryClassificationCBPE(CBPE):
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

        if not isinstance(y_pred_proba, str):
            raise InvalidArgumentsException(
                f"'y_pred_proba' is of type '{type(y_pred_proba)}'. "
                f"Binary use cases require 'y_pred_proba' to be a string."
            )
        self.y_pred_proba: str = y_pred_proba  # redeclare as str type to ease mypy

        self.confidence_upper_bound = 1
        self.confidence_lower_bound = 0

        self.previous_reference_results: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> CBPE:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_true, self.y_pred_proba, self.y_pred], list(reference_data.columns))

        reference_chunks = self.chunker.split(
            reference_data, minimum_chunk_size=300, timestamp_column_name=self.timestamp_column_name
        )

        self._alert_thresholds = self._calculate_alert_thresholds(reference_chunks, metrics=self.metrics)

        self._confidence_deviations = _calculate_confidence_deviations(
            reference_chunks, self.y_pred, self.y_pred_proba, metrics=self.metrics
        )

        self.minimum_chunk_size = _minimum_chunk_size(reference_data, self.y_pred, self.y_true)

        # Fit calibrator if calibration is needed
        aligned_reference_data = reference_data.reset_index(drop=True)  # fix mismatch between data and shuffle split
        self.needs_calibration = needs_calibration(
            y_true=aligned_reference_data[self.y_true],
            y_pred_proba=aligned_reference_data[self.y_pred_proba],
            calibrator=self.calibrator,
        )

        if self.needs_calibration:
            self.calibrator.fit(
                aligned_reference_data[self.y_pred_proba],
                aligned_reference_data[self.y_true],
            )

        self.previous_reference_results = self._estimate(reference_data).data

        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> CBPEPerformanceEstimatorResult:
        """Calculates the data reconstruction drift for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred_proba, self.y_pred], list(data.columns))

        if self.needs_calibration:
            data[self.y_pred_proba] = self.calibrator.calibrate(data[self.y_pred_proba])

        chunks = self.chunker.split(data, minimum_chunk_size=300, timestamp_column_name=self.timestamp_column_name)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    **self.__estimate(chunk),
                }
                for chunk in chunks
            ]
        )

        res = res.reset_index(drop=True)
        return CBPEPerformanceEstimatorResult(results_data=res, estimator=self)

    def __estimate(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = _estimate_metric(
                data=chunk.data, y_pred=self.y_pred, y_pred_proba=self.y_pred_proba, metric=metric
            )
            estimates[f'realized_{metric}'] = _calculate_realized_performance(
                chunk, self.y_true, self.y_pred, self.y_pred_proba, metric
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

    def _calculate_alert_thresholds(
        self,
        reference_chunks: List[Chunk],
        metrics: List[str],
        std_num: int = 3,
        lower_limit: int = 0,
        upper_limit: int = 1,
    ) -> Dict[str, Tuple[float, float]]:
        alert_thresholds = {}
        for metric in metrics:
            realised_performance_chunks = [
                _calculate_realized_performance(chunk, self.y_true, self.y_pred, self.y_pred_proba, metric)
                for chunk in reference_chunks
            ]
            deviation = np.std(realised_performance_chunks) * std_num
            mean_realised_performance = np.mean(realised_performance_chunks)
            lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
            upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

            alert_thresholds[metric] = (lower_threshold, upper_threshold)
        return alert_thresholds


def _estimate_metric(data: pd.DataFrame, y_pred: str, y_pred_proba: str, metric: str) -> float:
    if metric == 'roc_auc':
        return _estimate_roc_auc(data[y_pred_proba])
    elif metric == 'f1':
        return _estimate_f1(
            y_pred=data[y_pred],
            y_pred_proba=data[y_pred_proba],
        )
    elif metric == 'precision':
        return _estimate_precision(
            y_pred=data[y_pred],
            y_pred_proba=data[y_pred_proba],
        )
    elif metric == 'recall':
        return _estimate_recall(
            y_pred=data[y_pred],
            y_pred_proba=data[y_pred_proba],
        )
    elif metric == 'specificity':
        return _estimate_specificity(
            y_pred=data[y_pred],
            y_pred_proba=data[y_pred_proba],
        )
    elif metric == 'accuracy':
        return _estimate_accuracy(
            y_pred=data[y_pred],
            y_pred_proba=data[y_pred_proba],
        )
    else:
        raise InvalidArgumentsException(
            f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
        )


def _estimate_roc_auc(y_pred_proba: pd.Series) -> float:
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


def _estimate_f1(y_pred: np.ndarray, y_pred_proba: np.ndarray) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    fn = np.where(y_pred == 0, y_pred_proba, 0)
    TP, FP, FN = np.sum(tp), np.sum(fp), np.sum(fn)
    metric = TP / (TP + 0.5 * (FP + FN))
    return metric


def _estimate_precision(y_pred: np.ndarray, y_pred_proba: np.ndarray) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    TP, FP = np.sum(tp), np.sum(fp)
    metric = TP / (TP + FP)
    return metric


def _estimate_recall(y_pred: np.ndarray, y_pred_proba: np.ndarray) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    fn = np.where(y_pred == 0, y_pred_proba, 0)
    TP, FN = np.sum(tp), np.sum(fn)
    metric = TP / (TP + FN)
    return metric


def _estimate_specificity(y_pred: np.ndarray, y_pred_proba: np.ndarray) -> float:
    tn = np.where(y_pred == 0, 1 - y_pred_proba, 0)
    fp = np.where(y_pred == 1, 1 - y_pred_proba, 0)
    TN, FP = np.sum(tn), np.sum(fp)
    metric = TN / (TN + FP)
    return metric


def _estimate_accuracy(y_pred: np.ndarray, y_pred_proba: np.ndarray) -> float:
    tp = np.where(y_pred == 1, y_pred_proba, 0)
    tn = np.where(y_pred == 0, 1 - y_pred_proba, 0)
    TP, TN = np.sum(tp), np.sum(tn)
    metric = (TP + TN) / len(y_pred)
    return metric


def _calculate_confidence_deviations(reference_chunks: List[Chunk], y_pred: str, y_pred_proba: str, metrics: List[str]):
    return {
        metric: np.std([_estimate_metric(chunk.data, y_pred, y_pred_proba, metric) for chunk in reference_chunks])
        for metric in metrics
    }


def _calculate_realized_performance(chunk: Chunk, y_true_col: str, y_pred_col: str, y_pred_proba_col: str, metric: str):
    if y_true_col not in chunk.data.columns or chunk.data[y_true_col].isna().all():
        return np.NaN

    y_true = chunk.data[y_true_col]
    y_pred_proba = chunk.data[y_pred_proba_col]
    y_pred = chunk.data[y_pred_col]

    y_true = y_true[~y_pred_proba.isna()]
    y_pred_proba.dropna(inplace=True)

    y_pred_proba = y_pred_proba[~y_true.isna()]
    y_pred = y_pred[~y_true.isna()]
    y_true.dropna(inplace=True)

    if metric == 'roc_auc':
        return roc_auc_score(y_true, y_pred_proba)
    elif metric == 'f1':
        return f1_score(y_true=y_true, y_pred=y_pred)
    elif metric == 'precision':
        return precision_score(y_true=y_true, y_pred=y_pred)
    elif metric == 'recall':
        return recall_score(y_true=y_true, y_pred=y_pred)
    elif metric == 'specificity':
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        return conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    elif metric == 'accuracy':
        return accuracy_score(y_true=y_true, y_pred=y_pred)
    else:
        raise InvalidArgumentsException(
            f"unknown 'metric' value: '{metric}'. " f"Supported values are {SUPPORTED_METRIC_VALUES}."
        )


def _minimum_chunk_size(
    data: pd.DataFrame,
    prediction_column_name: str,
    target_column_name: str,
    lower_threshold: int = 300,
) -> int:
    def get_prediction(X):
        # model data
        h_coefs = [
            0.00000000e00,
            -3.46098897e04,
            2.65871679e04,
            3.46098897e04,
            2.29602791e04,
            -4.96886646e04,
            -1.12777343e-10,
            -2.29602791e04,
            3.13775672e-10,
            2.48718826e04,
        ]
        h_intercept = 1421.9522967076875
        transformation = PolynomialFeatures(3)
        #

        inputs = np.asarray(X)
        transformed_inputs = transformation.fit_transform(inputs)
        prediction = np.dot(transformed_inputs, h_coefs)[0] + h_intercept

        return prediction

    class_balance = np.mean(data[target_column_name])

    # Clean up NaN values
    y_true = data[target_column_name]
    y_pred = data[prediction_column_name]

    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    y_pred = y_pred[~y_true.isna()]
    y_true.dropna(inplace=True)

    auc = roc_auc_score(y_true=y_true, y_score=y_pred)

    chunk_size = get_prediction([[class_balance, auc]])
    chunk_size = np.maximum(lower_threshold, chunk_size)
    chunk_size = np.round(chunk_size, -2)
    minimum_chunk_size = int(chunk_size)

    return minimum_chunk_size
