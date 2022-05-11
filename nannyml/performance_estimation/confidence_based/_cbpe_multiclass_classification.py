#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from copy import deepcopy
from typing import Any, Dict, List, Tuple

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

from nannyml.calibration import Calibrator, NoopCalibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException, MissingMetadataException
from nannyml.metadata.base import NML_METADATA_TARGET_COLUMN_NAME, ModelMetadata
from nannyml.metadata.multiclass_classification import (
    NML_METADATA_PREDICTION_COLUMN_NAME,
    MulticlassClassificationMetadata,
)
from nannyml.performance_estimation.base import PerformanceEstimator
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
from nannyml.preprocessing import preprocess


class _MulticlassClassificationCBPE(CBPE):
    def __init__(
        self,
        model_metadata: MulticlassClassificationMetadata,
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
        self._calibrators: Dict[str, Calibrator] = {}

    def fit(self, reference_data: pd.DataFrame) -> PerformanceEstimator:
        if not isinstance(self.model_metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        reference_data = preprocess(data=reference_data, metadata=self.model_metadata, reference=True)

        _validate_data_requirements_for_metrics(data=reference_data, metadata=self.model_metadata, metrics=self.metrics)

        reference_chunks = self.chunker.split(reference_data, minimum_chunk_size=300)

        self._alert_thresholds = _calculate_alert_thresholds(
            reference_chunks, metadata=self.model_metadata, metrics=self.metrics
        )

        self._confidence_deviations = _calculate_confidence_deviations(
            reference_chunks, metadata=self.model_metadata, metrics=self.metrics
        )

        # self.minimum_chunk_size = _minimum_chunk_size(reference_data)
        self.minimum_chunk_size = 300

        # Fit calibrator if calibration is needed
        # This is just a flag, might just want to skip this
        # self.needs_calibration = needs_calibration(
        #     y_true=reference_data[NML_METADATA_TARGET_COLUMN_NAME],
        #     y_pred_proba=reference_data[self.model_metadata.predicted_class_probability_metadata_columns()],
        #     calibrator=self.calibrator,
        # )

        self._calibrators = _fit_calibrators(reference_data, self.model_metadata, self.calibrator)

        return self

    def estimate(self, data: pd.DataFrame) -> CBPEPerformanceEstimatorResult:
        data = preprocess(data=data, metadata=self.model_metadata)

        _validate_data_requirements_for_metrics(data, self.model_metadata, self.metrics)

        data = _calibrate_predicted_probabilities(data, self.model_metadata, self._calibrators)

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
        if not isinstance(self.model_metadata, MulticlassClassificationMetadata):
            raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = _estimate_metric(data=chunk.data, metadata=self.model_metadata, metric=metric)
            estimates[f'confidence_{metric}'] = self._confidence_deviations[metric]
            estimates[f'realized_{metric}'] = _calculate_realized_performance(chunk, self.model_metadata, metric)
            estimates[f'estimated_{metric}'] = estimated_metric
            estimates[f'upper_threshold_{metric}'] = self._alert_thresholds[metric][0]
            estimates[f'lower_threshold_{metric}'] = self._alert_thresholds[metric][1]
            estimates[f'alert_{metric}'] = (
                estimated_metric > self._alert_thresholds[metric][1]
                or estimated_metric < self._alert_thresholds[metric][0]
            ) and chunk.partition == 'analysis'
        return estimates


def _validate_data_requirements_for_metrics(data: pd.DataFrame, metadata: ModelMetadata, metrics: List[str]):
    if not isinstance(metadata, MulticlassClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

    if 'roc_auc' in metrics:
        if metadata.predicted_probabilities_column_names is None:
            raise MissingMetadataException(
                "missing value for 'predicted_probabilities_column_names'. Please ensure predicted "
                "class probabilities are specified and present in the sample."
            )

    if metadata.prediction_column_name is None:
        raise MissingMetadataException(
            "missing value for 'prediction_column_name'. Please ensure predicted "
            "label values are specified and present in the sample."
        )


def _get_predictions(data: pd.DataFrame, metadata: MulticlassClassificationMetadata):
    classes = sorted(list(metadata.predicted_probabilities_column_names.keys()))
    y_preds = list(label_binarize(data[NML_METADATA_PREDICTION_COLUMN_NAME], classes=classes).T)

    class_probability_column_names = metadata.predicted_class_probability_metadata_columns()
    y_pred_probas = [data[class_probability_column_names[clazz]] for clazz in classes]
    return y_preds, y_pred_probas


def _estimate_metric(data: pd.DataFrame, metadata: MulticlassClassificationMetadata, metric: str) -> float:
    if metric == 'roc_auc':
        return _estimate_roc_auc(data[list(metadata.predicted_class_probability_metadata_columns().values())])
    elif metric == 'f1':
        y_preds, y_pred_probas = _get_predictions(data, metadata)
        return _estimate_f1(y_preds, y_pred_probas)
    elif metric == 'precision':
        y_preds, y_pred_probas = _get_predictions(data, metadata)
        return _estimate_precision(y_preds, y_pred_probas)
    elif metric == 'recall':
        y_preds, y_pred_probas = _get_predictions(data, metadata)
        return _estimate_recall(y_preds, y_pred_probas)
    elif metric == 'specificity':
        y_preds, y_pred_probas = _get_predictions(data, metadata)
        return _estimate_specificity(y_preds, y_pred_probas)
    elif metric == 'accuracy':
        y_preds, y_pred_probas = _get_predictions(data, metadata)
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
    metadata: ModelMetadata,
    std_num: int = 3,
    lower_limit: int = 0,
    upper_limit: int = 1,
) -> Dict[str, Tuple[float, float]]:
    if not isinstance(metadata, MulticlassClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

    alert_thresholds = {}
    for metric in metrics:
        realised_performance_chunks = [
            _calculate_realized_performance(chunk, metadata, metric) for chunk in reference_chunks
        ]
        deviation = np.std(realised_performance_chunks) * std_num
        mean_realised_performance = np.mean(realised_performance_chunks)
        lower_threshold = np.maximum(mean_realised_performance - deviation, lower_limit)
        upper_threshold = np.minimum(mean_realised_performance + deviation, upper_limit)

        alert_thresholds[metric] = (lower_threshold, upper_threshold)
    return alert_thresholds


def _calculate_realized_performance(chunk: Chunk, metadata: ModelMetadata, metric: str):
    if not isinstance(metadata, MulticlassClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

    if (
        NML_METADATA_TARGET_COLUMN_NAME not in chunk.data.columns
        or chunk.data[NML_METADATA_TARGET_COLUMN_NAME].isna().all()
    ):
        return np.NaN

    # Make sure labels and class_probability_columns have the same ordering
    labels, class_probability_columns = [], []
    for label in sorted(list(metadata.predicted_class_probability_metadata_columns())):
        labels.append(label)
        class_probability_columns.append(metadata.predicted_class_probability_metadata_columns()[label])

    y_true = chunk.data[NML_METADATA_TARGET_COLUMN_NAME]
    y_pred_probas = chunk.data[class_probability_columns]
    y_pred = chunk.data[NML_METADATA_PREDICTION_COLUMN_NAME]

    # y_true = y_true[~y_pred_proba.isna()]
    # y_pred_proba.dropna(inplace=True)
    #
    # y_pred_proba = y_pred_proba[~y_true.isna()]
    # y_true.dropna(inplace=True)

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


def _calculate_confidence_deviations(reference_chunks: List[Chunk], metadata: ModelMetadata, metrics: List[str]):
    if not isinstance(metadata, MulticlassClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

    return {
        metric: np.std([_estimate_metric(chunk.data, metadata, metric) for chunk in reference_chunks])
        for metric in metrics
    }


def _get_class_splits(data: pd.DataFrame, metadata: ModelMetadata, include_targets: bool = True) -> List[Tuple]:
    if not isinstance(metadata, MulticlassClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

    classes = sorted(list(metadata.predicted_probabilities_column_names.keys()))
    y_trues: List[np.ndarray] = []

    if include_targets:
        y_trues = list(label_binarize(data[NML_METADATA_TARGET_COLUMN_NAME], classes=classes).T)

    class_probability_column_names = metadata.predicted_class_probability_metadata_columns()
    y_pred_probas = [data[class_probability_column_names[clazz]] for clazz in classes]

    return [
        (classes[idx], y_trues[idx] if include_targets else None, y_pred_probas[idx]) for idx in range(len(classes))
    ]


def _fit_calibrators(
    reference_data: pd.DataFrame, metadata: ModelMetadata, calibrator: Calibrator
) -> Dict[str, Calibrator]:
    fitted_calibrators = {}
    noop_calibrator = NoopCalibrator()

    for clazz, y_true, y_pred_proba in _get_class_splits(reference_data, metadata):
        if not needs_calibration(np.asarray(y_true), np.asarray(y_pred_proba), calibrator):
            calibrator = noop_calibrator

        calibrator.fit(y_pred_proba, y_true)
        fitted_calibrators[clazz] = deepcopy(calibrator)

    return fitted_calibrators


def _calibrate_predicted_probabilities(
    data: pd.DataFrame, metadata: ModelMetadata, calibrators: Dict[str, Calibrator]
) -> pd.DataFrame:
    if not isinstance(metadata, MulticlassClassificationMetadata):
        raise InvalidArgumentsException('metadata was not an instance of MulticlassClassificationMetadata')

    class_splits = _get_class_splits(data, metadata, include_targets=False)
    number_of_observations = len(data)
    number_of_classes = len(class_splits)

    calibrated_probas = np.zeros((number_of_observations, number_of_classes))

    for idx, split in enumerate(class_splits):
        clazz, _, y_pred_proba = split
        calibrated_probas[:, idx] = calibrators[clazz].calibrate(y_pred_proba)

    denominator = np.sum(calibrated_probas, axis=1)[:, np.newaxis]
    uniform_proba = np.full_like(calibrated_probas, 1 / number_of_classes)

    calibrated_probas = np.divide(calibrated_probas, denominator, out=uniform_proba, where=denominator != 0)

    calibrated_data = data.copy(deep=True)
    predicted_class_proba_column_names = sorted(metadata.predicted_class_probability_metadata_columns())
    for idx in range(number_of_classes):
        calibrated_data[predicted_class_proba_column_names[idx]] = calibrated_probas[:, idx]

    return calibrated_data
