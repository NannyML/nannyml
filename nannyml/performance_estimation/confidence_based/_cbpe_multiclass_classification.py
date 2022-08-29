#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from nannyml._typing import ModelOutputsType, model_output_column_names
from nannyml.base import _list_missing
from nannyml.calibration import Calibrator, NoopCalibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based.results import CBPEPerformanceEstimatorResult
from nannyml.sampling_error import SAMPLING_ERROR_RANGE


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

        for metric in self.metrics:
            metric.fit(reference_data)

        self._calibrators = _fit_calibrators(reference_data, self.y_true, self.y_pred_proba, self.calibrator)

        self.previous_reference_results = self._estimate(reference_data).data
        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> CBPEPerformanceEstimatorResult:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred] + model_output_column_names(self.y_pred_proba), data)

        data = _calibrate_predicted_probabilities(data, self.y_true, self.y_pred_proba, self._calibrators)

        chunks = self.chunker.split(data, timestamp_column_name=self.timestamp_column_name)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    **self._estimate_for_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        res = res.reset_index(drop=True)
        return CBPEPerformanceEstimatorResult(results_data=res, estimator=self)

    def _estimate_for_chunk(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = metric.estimate(chunk.data)
            sampling_error = metric.sampling_error(chunk.data)
            estimates[f'realized_{metric.column_name}'] = metric.realized_performance(chunk.data)
            estimates[f'estimated_{metric.column_name}'] = estimated_metric
            estimates[f'sampling_error_{metric.column_name}'] = sampling_error
            estimates[f'upper_confidence_{metric.column_name}'] = min(
                self.confidence_upper_bound, estimated_metric + SAMPLING_ERROR_RANGE * sampling_error
            )
            estimates[f'lower_confidence_{metric.column_name}'] = max(
                self.confidence_lower_bound, estimated_metric - SAMPLING_ERROR_RANGE * sampling_error
            )
            estimates[f'upper_threshold_{metric.column_name}'] = metric.upper_threshold
            estimates[f'lower_threshold_{metric.column_name}'] = metric.lower_threshold
            estimates[f'alert_{metric.column_name}'] = (
                estimated_metric > metric.upper_threshold or estimated_metric < metric.lower_threshold
            )
        return estimates


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
