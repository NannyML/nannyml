#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import Any, Dict, List, Tuple, cast

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

from nannyml.calibration import Calibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
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
from nannyml.performance_estimation.base import PerformanceEstimator
from nannyml.performance_estimation.confidence_based import CBPE
from nannyml.performance_estimation.confidence_based.results import (
    SUPPORTED_METRIC_VALUES,
    CBPEPerformanceEstimatorResult,
)
from nannyml.preprocessing import preprocess


class _BinaryClassificationCBPE(CBPE):
    def __init__(
        self,
        model_metadata: BinaryClassificationMetadata,
        metrics: List[str],
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
        metrics: List[str]
            A list of metrics to calculate.
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

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df)
        >>> # create a new estimator, chunking by week
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W')

        """
        super().__init__(
            model_metadata=model_metadata,
            features=features,
            metrics=metrics,
            chunk_size=chunk_size,
            chunk_number=chunk_number,
            chunk_period=chunk_period,
            chunker=chunker,
            calibration=calibration,
            calibrator=calibrator,
        )

        self.model_metadata = model_metadata  # seems to be required for typing to kick in

    def fit(self, reference_data: pd.DataFrame) -> PerformanceEstimator:
        """Fits the drift calculator using a set of reference data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A reference data set containing predictions (labels and/or probabilities) and target values.

        Returns
        -------
        estimator: PerformanceEstimator
            The fitted estimator.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # create a new estimator and fit it on reference data
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W').fit(ref_df)

        """
        reference_data = preprocess(
            data=reference_data, metadata=cast(BinaryClassificationMetadata, self.model_metadata), reference=True
        )

        _validate_data_requirements_for_metrics(reference_data, self.model_metadata, self.metrics)

        reference_chunks = self.chunker.split(reference_data, minimum_chunk_size=300)

        self._alert_thresholds = _calculate_alert_thresholds(reference_chunks, metrics=self.metrics)

        self._confidence_deviations = _calculate_confidence_deviations(reference_chunks, metrics=self.metrics)

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

        return self

    def estimate(self, data: pd.DataFrame) -> CBPEPerformanceEstimatorResult:
        """Calculates the data reconstruction drift for a given data set.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to calculate the reconstruction drift for.

        Returns
        -------
        estimates: PerformanceEstimatorResult
            A :class:`result<nannyml.performance_estimation.confidence_based.results.CBPEPerformanceEstimatorResult>`
            object where each row represents a :class:`~nannyml.chunk.Chunk`,
            containing :class:`~nannyml.chunk.Chunk` properties and the estimated metrics
            for that :class:`~nannyml.chunk.Chunk`.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # create a new estimator and fit it on reference data
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W').fit(ref_df)
        >>> estimates = estimator.estimate(data)
        """
        data = preprocess(data=data, metadata=self.model_metadata)

        _validate_data_requirements_for_metrics(data, self.model_metadata, self.metrics)

        if self.needs_calibration:
            data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME] = self.calibrator.calibrate(
                data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]
            )

        features_and_metadata = self.model_metadata.metadata_columns + self.selected_features
        chunks = self.chunker.split(data, columns=features_and_metadata, minimum_chunk_size=300)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'partition': 'analysis' if chunk.is_transition else chunk.partition,
                    **self._estimate(chunk),
                }
                for chunk in chunks
            ]
        )

        res = res.reset_index(drop=True)
        return CBPEPerformanceEstimatorResult(estimated_data=res, model_metadata=self.model_metadata)

    def _estimate(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = _estimate_metric(data=chunk.data, metric=metric)
            estimates[f'confidence_{metric}'] = self._confidence_deviations[metric]
            estimates[f'realized_{metric}'] = _calculate_realized_performance(chunk, metric)
            estimates[f'estimated_{metric}'] = estimated_metric
            estimates[f'upper_threshold_{metric}'] = self._alert_thresholds[metric][0]
            estimates[f'lower_threshold_{metric}'] = self._alert_thresholds[metric][1]
            estimates[f'alert_{metric}'] = (
                estimated_metric > self._alert_thresholds[metric][1]
                or estimated_metric < self._alert_thresholds[metric][0]
            ) and chunk.partition == 'analysis'
        return estimates


def _validate_data_requirements_for_metrics(data: pd.DataFrame, metadata: ModelMetadata, metrics: List[str]):
    if not isinstance(metadata, BinaryClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of BinaryClassificationMetadata')

    if 'roc_auc' in metrics:
        if metadata.predicted_probability_column_name is None:
            raise MissingMetadataException(
                "missing value for 'predicted_probability_column_name'. Please ensure predicted "
                "label values are specified and present in the sample."
            )

    if metadata.prediction_column_name is None:
        raise MissingMetadataException(
            "missing value for 'prediction_column_name'. Please ensure predicted "
            "label values are specified and present in the sample."
        )


def _estimate_metric(data: pd.DataFrame, metric: str) -> float:
    if metric == 'roc_auc':
        return _estimate_roc_auc(data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME])
    elif metric == 'f1':
        return _estimate_f1(
            y_pred=data[NML_METADATA_PREDICTION_COLUMN_NAME],
            y_pred_proba=data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
        )
    elif metric == 'precision':
        return _estimate_precision(
            y_pred=data[NML_METADATA_PREDICTION_COLUMN_NAME],
            y_pred_proba=data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
        )
    elif metric == 'recall':
        return _estimate_recall(
            y_pred=data[NML_METADATA_PREDICTION_COLUMN_NAME],
            y_pred_proba=data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
        )
    elif metric == 'specificity':
        return _estimate_specificity(
            y_pred=data[NML_METADATA_PREDICTION_COLUMN_NAME],
            y_pred_proba=data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
        )
    elif metric == 'accuracy':
        return _estimate_accuracy(
            y_pred=data[NML_METADATA_PREDICTION_COLUMN_NAME],
            y_pred_proba=data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME],
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


def _calculate_confidence_deviations(reference_chunks: List[Chunk], metrics: List[str]):
    return {metric: np.std([_estimate_metric(chunk.data, metric) for chunk in reference_chunks]) for metric in metrics}


def _calculate_realized_performance(chunk: Chunk, metric: str):
    if (
        NML_METADATA_TARGET_COLUMN_NAME not in chunk.data.columns
        or chunk.data[NML_METADATA_TARGET_COLUMN_NAME].isna().all()
    ):
        return np.NaN

    y_true = chunk.data[NML_METADATA_TARGET_COLUMN_NAME]
    y_pred_proba = chunk.data[NML_METADATA_PREDICTED_PROBABILITY_COLUMN_NAME]
    y_pred = chunk.data[NML_METADATA_PREDICTION_COLUMN_NAME]

    y_true = y_true[~y_pred_proba.isna()]
    y_pred_proba.dropna(inplace=True)

    y_pred_proba = y_pred_proba[~y_true.isna()]
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


def _calculate_alert_thresholds(
    reference_chunks: List[Chunk], metrics: List[str], std_num: int = 3, lower_limit: int = 0, upper_limit: int = 1
) -> Dict[str, Tuple[float, float]]:
    alert_thresholds = {}
    for metric in metrics:
        realised_performance_chunks = [_calculate_realized_performance(chunk, metric) for chunk in reference_chunks]
        deviation = np.std(realised_performance_chunks) * std_num
        mean_realised_performance = np.mean(realised_performance_chunks)
        lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
        upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

        alert_thresholds[metric] = (lower_threshold, upper_threshold)
    return alert_thresholds


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
