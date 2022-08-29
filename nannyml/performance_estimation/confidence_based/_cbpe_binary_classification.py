#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import Any, Dict, List, Optional

import pandas as pd

from nannyml._typing import ModelOutputsType
from nannyml.calibration import Calibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_calculation.metrics import _list_missing
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based.results import CBPEPerformanceEstimatorResult
from nannyml.sampling_error import SAMPLING_ERROR_RANGE


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

        for metric in self.metrics:
            metric.fit(reference_data)

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

        chunks = self.chunker.split(data, timestamp_column_name=self.timestamp_column_name)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    **self._estimate_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        res = res.reset_index(drop=True)
        return CBPEPerformanceEstimatorResult(results_data=res, estimator=self)

    def _estimate_chunk(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = metric.estimate(chunk.data)
            sampling_error = metric.sampling_error(chunk.data)
            estimates[f'realized_{metric.column_name}'] = metric.realized_performance(chunk.data)
            estimates[f'estimated_{metric.column_name}'] = estimated_metric
            estimates[f'upper_confidence_{metric.column_name}'] = min(
                self.confidence_upper_bound, estimated_metric + SAMPLING_ERROR_RANGE * sampling_error
            )
            estimates[f'lower_confidence_{metric.column_name}'] = max(
                self.confidence_lower_bound, estimated_metric - SAMPLING_ERROR_RANGE * sampling_error
            )
            estimates[f'sampling_error_{metric.column_name}'] = sampling_error
            estimates[f'upper_threshold_{metric.column_name}'] = metric.upper_threshold
            estimates[f'lower_threshold_{metric.column_name}'] = metric.lower_threshold
            estimates[f'alert_{metric.column_name}'] = (
                estimated_metric > metric.upper_threshold or estimated_metric < metric.lower_threshold
            )
        return estimates
