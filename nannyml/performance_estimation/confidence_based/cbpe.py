#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the CBPE estimator."""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

from nannyml import Calibrator, Chunk, Chunker, ModelMetadata
from nannyml.calibration import CalibratorFactory, needs_calibration
from nannyml.exceptions import NotFittedException
from nannyml.metadata import (
    NML_METADATA_COLUMNS,
    NML_METADATA_PARTITION_COLUMN_NAME,
    NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    NML_METADATA_REFERENCE_PARTITION_NAME,
    NML_METADATA_TARGET_COLUMN_NAME,
)
from nannyml.performance_estimation.base import BasePerformanceEstimator, PerformanceEstimatorResult
from nannyml.performance_estimation.confidence_based.results import CBPEPerformanceEstimatorResult


class CBPE(BasePerformanceEstimator):
    """Performance estimator using the Confidence Based Performance Estimation (CBPE) technique."""

    def __init__(
        self,
        model_metadata: ModelMetadata,
        features: List[str] = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        calibration: str = None,
        calibrator: Calibrator = None,
    ):
        """Creates a new CBPE performance estimator.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata telling the DriftCalculator what columns are required for drift calculation.
        features: List[str], default=None
            An optional list of feature column names. When set only these columns will be included in the
            drift calculation. If not set all feature columns will be used.
        chunk_size: int, default=None
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int, default=None
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str, default=None
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker, default=None
            The `Chunker` used to split the data sets into a lists of chunks.
        calibration: str, default='isotonic'
            Determines which calibration will be applied to the model predictions. Defaults to ``isotonic``, currently
            the only supported value.
        calibrator: Calibrator, default=None
            A specific instance of a Calibrator to be applied to the model predictions.
            If not set NannyML will use the value of the ``calibration`` variable instead.

        """
        super().__init__(model_metadata, features, chunk_size, chunk_number, chunk_period, chunker)

        self._upper_alert_threshold: float
        self._lower_alert_threshold: float

        if calibrator is None:
            calibrator = CalibratorFactory.create(calibration)
        self.calibrator = calibrator

        self.minimum_chunk_size: int = None  # type: ignore

    def _fit(self, reference_data: pd.DataFrame):
        if self.chunker is None:
            raise NotFittedException()

        # y_true = y_true[~y_pred_proba.isna()]
        # y_pred_proba.dropna(inplace=True)
        #
        # y_pred_proba = y_pred_proba[~y_true.isna()]
        # y_true.dropna(inplace=True)

        reference_chunks = self.chunker.split(reference_data, minimum_chunk_size=300)

        self._lower_alert_threshold, self._upper_alert_threshold = _calculate_alert_thresholds(reference_chunks)

        self._confidence_deviation = _calculate_confidence_deviation(reference_chunks)

        self.minimum_chunk_size = _minimum_chunk_size(reference_data)

        # Fit calibrator if calibration is needed
        self.needs_calibration = needs_calibration(
            y_true=reference_data[NML_METADATA_TARGET_COLUMN_NAME],
            y_pred_proba=reference_data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
            calibrator=self.calibrator,
        )

        if self.needs_calibration:
            self.calibrator.fit(
                reference_data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
                reference_data[NML_METADATA_TARGET_COLUMN_NAME],
            )

    def _estimate(self, data: pd.DataFrame) -> PerformanceEstimatorResult:
        features_and_metadata = NML_METADATA_COLUMNS + self.selected_features
        chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=self.minimum_chunk_size)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'partition': 'analysis' if chunk.is_transition else chunk.partition,
                    'realized_roc_auc': _calculate_realized_performance(chunk),
                    'estimated_roc_auc': _calculate_cbpe(
                        self.calibrator.calibrate(chunk.data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME])
                        if self.needs_calibration
                        else chunk.data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]
                    ),
                }
                for chunk in chunks
            ]
        )

        res['confidence'] = self._confidence_deviation
        res['upper_threshold'] = [self._upper_alert_threshold] * len(res)
        res['lower_threshold'] = [self._lower_alert_threshold] * len(res)
        res['alert'] = _add_alert_flag(res, self._upper_alert_threshold, self._lower_alert_threshold)
        res = res.reset_index(drop=True)
        return CBPEPerformanceEstimatorResult(estimated_data=res, model_metadata=self.model_metadata)


def _calculate_alert_thresholds(
    reference_chunks: List[Chunk], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
) -> Tuple[float, float]:

    realised_performance_chunks = [_calculate_realized_performance(chunk) for chunk in reference_chunks]

    deviation = np.std(realised_performance_chunks) * std_num
    mean_realised_performance = np.mean(realised_performance_chunks)
    lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
    upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

    return lower_threshold, upper_threshold


def _calculate_confidence_deviation(reference_chunks: List[Chunk]):
    estimated_reference_performance_chunks = [
        _calculate_cbpe(chunk.data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]) for chunk in reference_chunks
    ]
    deviation = np.std(estimated_reference_performance_chunks)
    return deviation


def _calculate_realized_performance(chunk: Chunk):
    if (
        NML_METADATA_TARGET_COLUMN_NAME not in chunk.data.columns
        or chunk.data[NML_METADATA_TARGET_COLUMN_NAME].isna().all()
    ):
        return np.NaN

    y_true = chunk.data[NML_METADATA_TARGET_COLUMN_NAME]
    y_pred_proba = chunk.data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]

    y_true = y_true[~y_pred_proba.isna()]
    y_pred_proba.dropna(inplace=True)

    y_pred_proba = y_pred_proba[~y_true.isna()]
    y_true.dropna(inplace=True)

    return roc_auc_score(y_true, y_pred_proba)


def _calculate_cbpe(data: pd.Series) -> float:
    thresholds = np.sort(data)
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


def _add_alert_flag(estimated_performance: pd.DataFrame, upper_threshold: float, lower_threshold: float) -> pd.Series:
    alert = estimated_performance.apply(
        lambda row: True
        if (row['estimated_roc_auc'] > upper_threshold or row['estimated_roc_auc'] < lower_threshold)
        and row['partition'] == 'analysis'
        else False,
        axis=1,
    )

    return alert


def _minimum_chunk_size(
    data: pd.DataFrame,
    partition_column_name: str = NML_METADATA_PARTITION_COLUMN_NAME,
    prediction_column_name: str = NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME,
    target_column_name: str = NML_METADATA_TARGET_COLUMN_NAME,
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
    y_true = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, target_column_name]
    y_pred = data.loc[data[partition_column_name] == NML_METADATA_REFERENCE_PARTITION_NAME, prediction_column_name]

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
