#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""A module with the implementation of the CBPE estimator.

The estimator manages a list of :class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances,
constructed using the :class:`~nannyml.performance_estimation.confidence_based.metrics.MetricFactory`.

The estimator is then responsible for delegating the `fit` and `estimate` method calls to each of the managed
:class:`~nannyml.performance_estimation.confidence_based.metrics.Metric` instances and building a
:class:`~nannyml.performance_estimation.confidence_based.results.Result` object.

For more information, check out the `tutorial`_ and the `deep dive`_.

.. _tutorial:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/binary_performance_estimation.html

.. _deep dive:
    https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#confidence-based-performance-estimation-cbpe
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import MultiIndex
from sklearn.preprocessing import label_binarize

from nannyml._typing import ModelOutputsType, ProblemType, model_output_column_names
from nannyml.base import AbstractEstimator, _list_missing
from nannyml.calibration import Calibrator, CalibratorFactory, NoopCalibrator, needs_calibration
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.confidence_based import SUPPORTED_METRIC_VALUES
from nannyml.performance_estimation.confidence_based.metrics import MetricFactory
from nannyml.performance_estimation.confidence_based.results import Result
from nannyml.thresholds import StandardDeviationThreshold, Threshold
from nannyml.usage_logging import UsageEvent, log_usage

DEFAULT_THRESHOLDS: Dict[str, Threshold] = {
    'roc_auc': StandardDeviationThreshold(),
    'f1': StandardDeviationThreshold(),
    'precision': StandardDeviationThreshold(),
    'recall': StandardDeviationThreshold(),
    'specificity': StandardDeviationThreshold(),
    'accuracy': StandardDeviationThreshold(),
    'confusion_matrix': StandardDeviationThreshold(),
    'business_value': StandardDeviationThreshold(),
    'average_precision': StandardDeviationThreshold(),
}


class CBPE(AbstractEstimator):
    """Performance estimator using the Confidence Based Performance Estimation (CBPE) technique.

    CBPE leverages the confidence score of the model predictions. It is used to estimate the performance of
    classification models as they return predictions with an associated confidence score.

    For more information, check out the `tutorial for binary classification`_,
    the `tutorial for multiclass classification`_ or the `deep dive`_.

    .. _tutorial for binary classification:
        https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/binary_performance_estimation.html

    .. _tutorial for multiclass classification:
        https://nannyml.readthedocs.io/en/stable/tutorials/performance_estimation/multiclass_performance_estimation.html

    .. _deep dive:
        https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html#confidence-based-performance-estimation-cbpe
    """

    def __init__(
        self,
        metrics: Union[str, List[str]],
        y_pred_proba: ModelOutputsType,
        y_true: str,
        problem_type: Union[str, ProblemType],
        y_pred: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        calibration: str = 'isotonic',
        calibrator: Optional[Calibrator] = None,
        thresholds: Optional[Dict[str, Threshold]] = None,
        normalize_confusion_matrix: Optional[str] = None,
        business_value_matrix: Optional[Union[List, np.ndarray]] = None,
        normalize_business_value: Optional[str] = None,
    ):
        """Initializes a new CBPE performance estimator.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred_proba: Union[str, Dict[str, str]]
            Name(s) of the column(s) containing your model output.

                - For binary classification, pass a single string refering to the model output column.
                - For multiclass classification, pass a dictionary that maps a class string to the column name
                  model outputs for that class.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        metrics: Union[str, List[str]]
            A metric or list of metrics to calculate.

            Supported metrics by CBPE:

                - `roc_auc`
                - `f1`
                - `precision`
                - `recall`
                - `specificity`
                - `accuracy`
                - `confusion_matrix` - only for binary classification tasks
                - `business_value` - only for binary classification tasks
        y_pred: str
            The name of the column containing your model predictions.
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
            Determines which calibration will be applied to the model predictions. Defaults to 'isotonic', currently
            the only supported value.
        calibrator: Calibrator, default=None
            A specific instance of a Calibrator to be applied to the model predictions.
            If not set NannyML will use the value of the ``calibration`` variable instead.
        thresholds: dict
            The default values are::

                {
                    'roc_auc': StandardDeviationThreshold(),
                    'f1': StandardDeviationThreshold(),
                    'precision': StandardDeviationThreshold(),
                    'recall': StandardDeviationThreshold(),
                    'specificity': StandardDeviationThreshold(),
                    'accuracy': StandardDeviationThreshold(),
                    'confusion_matrix': StandardDeviationThreshold(),
                    'business_value': StandardDeviationThreshold(),
                }

            A dictionary allowing users to set a custom threshold for each method. It links a `Threshold` subclass
            to a method name. This dictionary is optional.
            When a dictionary is given its values will override the default values. If no dictionary is given a default
            will be applied.
        problem_type: Union[str, ProblemType]
            Determines which CBPE implementation to use. Allowed problem type values are 'classification_binary' and
            'classification_multiclass'.
        normalize_confusion_matrix: str, default=None
            Determines how the confusion matrix will be normalized. Allowed values are None, 'all', 'true' and
            'predicted'.

                - None - the confusion matrix will not be normalized and the counts for each cell of the matrix \
                will be returned.
                - 'all' - the confusion matrix will be normalized by the total number of observations.
                - 'true' - the confusion matrix will be normalized by the total number of observations for each true  \
                class.
                - 'predicted' - the confusion matrix will be normalized by the total number of observations for each \
                predicted class.
        business_value_matrix: Optional[Union[List, np.ndarray]], default=None
            A nxn matrix that specifies the value of each cell in the confusion matrix.
            The format of the business value matrix must be specified so that each element represents the business
            value of it's respective confusion matrix element. Hence the element on the i-th row and j-column of the
            business value matrix tells us the value of the i-th target while we predicted the j-th value.
            It can be provided as a list of lists or a numpy array.
        normalize_business_value: str, default=None
            Determines how the business value will be normalized. Allowed values are None and
            'per_prediction'.

            - None - the business value will not be normalized and the value returned will be the total value per chunk.
            - 'per_prediction' - the value will be normalized by the number of predictions in the chunk.

        Examples
        --------
        Using CBPE to estimate the perfomance of a model for a binary classification problem.

        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df = nml.load_synthetic_car_loan_dataset()[0]
        >>> analysis_df = nml.load_synthetic_car_loan_dataset()[1]
        >>> display(reference_df.head(3))
        >>> estimator = nml.CBPE(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='repaid',
        ...     timestamp_column_name='timestamp',
        ...     metrics=['roc_auc', 'accuracy', 'f1'],
        ...     chunk_size=5000,
        ...     problem_type='classification_binary',
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)
        >>> display(results.filter(period='analysis').to_df())
        >>> metric_fig = results.plot()
        >>> metric_fig.show()


        Using CBPE to estimate the perfomance of a model for a multiclass classification problem.

        >>> import nannyml as nml
        >>> reference_df, analysis_df, _ = nml.load_synthetic_multiclass_classification_dataset()
        >>> estimator = nml.CBPE(
        ...     y_pred_proba={
        ...         'prepaid_card': 'y_pred_proba_prepaid_card',
        ...         'highstreet_card': 'y_pred_proba_highstreet_card',
        ...         'upmarket_card': 'y_pred_proba_upmarket_card'},
        ...     y_pred='y_pred',
        ...     y_true='y_true',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_multiclass',
        ...     metrics=['roc_auc', 'f1'],
        ...     chunk_size=6000,
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)
        >>> metric_fig = results.plot()
        >>> metric_fig.show()
        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name)

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        if metrics is None or len(metrics) == 0:
            raise InvalidArgumentsException(
                "no metrics provided. Please provide a non-empty list of metrics."
                f"Supported values are {SUPPORTED_METRIC_VALUES}."
            )

        valid_normalizations = [None, 'all', 'pred', 'true']
        if normalize_confusion_matrix not in valid_normalizations:
            raise InvalidArgumentsException(
                f"'normalize_confusion_matrix' given was '{normalize_confusion_matrix}'. "
                f"Binary use cases require 'normalize_confusion_matrix' to be one of {valid_normalizations}."
            )

        if normalize_business_value not in [None, "per_prediction"]:
            raise InvalidArgumentsException(
                f"normalize_business_value must be None or 'per_prediction', but got '{normalize_business_value}'"
            )

        if isinstance(problem_type, str):
            self.problem_type = ProblemType.parse(problem_type)
        else:
            self.problem_type = problem_type

        if self.problem_type is not ProblemType.CLASSIFICATION_BINARY and y_pred is None:
            raise InvalidArgumentsException(f"'y_pred' can not be 'None' for problem type {self.problem_type.value}")

        if self.problem_type == ProblemType.CLASSIFICATION_BINARY:
            if not isinstance(self.y_pred_proba, str):
                raise InvalidArgumentsException("y_pred_proba must be a string for binary classification")
        elif self.problem_type == ProblemType.CLASSIFICATION_MULTICLASS:
            if not isinstance(self.y_pred_proba, dict):
                raise InvalidArgumentsException("y_pred_proba must be a dictionary for multiclass classification")

        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(**thresholds)

        if isinstance(metrics, str):
            metrics = [metrics]

        raise_if_metrics_require_y_pred(metrics, y_pred)

        self.metrics = []
        for metric in metrics:
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(
                    f"unknown metric key '{metric}' given. " f"Should be one of {SUPPORTED_METRIC_VALUES}."
                )
            self.metrics.append(
                MetricFactory.create(
                    metric,
                    self.problem_type,
                    y_pred_proba=self.y_pred_proba,
                    y_pred=self.y_pred,
                    y_true=self.y_true,
                    timestamp_column_name=self.timestamp_column_name,
                    chunker=self.chunker,
                    threshold=self.thresholds[metric],
                    normalize_confusion_matrix=normalize_confusion_matrix,
                    business_value_matrix=business_value_matrix,
                    normalize_business_value=normalize_business_value,
                )
            )

        self.needs_calibration: bool = False

        if calibrator is None:
            calibrator = CalibratorFactory.create(calibration)

        # Used in binary cases
        # TODO: unify this with multiclass case (or remove from public interface)
        self.calibrator = calibrator

        # Used in multiclass cases
        self._calibrators: Dict[str, Calibrator] = {}

        self.result: Optional[Result] = None

    @log_usage(UsageEvent.CBPE_ESTIMATOR_FIT, metadata_from_self=['metrics', 'problem_type'])
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> CBPE:
        """Fits the drift calculator using a set of reference data.

        Parameters
        ----------
        reference_data : pd.DataFrame
            A reference data set containing predictions (labels and/or probabilities) and target values.

        Returns
        -------
        estimator: PerformanceEstimator
            The fitted estimator.
        """
        reference_data = reference_data.copy(deep=True)

        if self.problem_type == ProblemType.CLASSIFICATION_BINARY:
            return self._fit_binary(reference_data)
        elif self.problem_type == ProblemType.CLASSIFICATION_MULTICLASS:
            return self._fit_multiclass(reference_data)
        else:
            raise InvalidArgumentsException('CBPE can only be used for binary or multiclass classification problems.')

    @log_usage(UsageEvent.CBPE_ESTIMATOR_RUN, metadata_from_self=['metrics', 'problem_type'])
    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates the data reconstruction drift for a given data set.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to calculate the reconstruction drift for.

        Returns
        -------
        estimates: PerformanceEstimatorResult
            A :class:`result<nannyml.performance_estimation.confidence_based.results.Result>`
            object where each row represents a :class:`~nannyml.chunk.Chunk`,
            containing :class:`~nannyml.chunk.Chunk` properties and the estimated metrics
            for that :class:`~nannyml.chunk.Chunk`.
        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        data = data.copy(deep=True)

        if self.problem_type == ProblemType.CLASSIFICATION_BINARY:
            assert isinstance(self.y_pred_proba, str)
            required_cols = [self.y_pred_proba]
            if self.y_pred is not None:
                required_cols.append(self.y_pred)
            _list_missing(required_cols, list(data.columns))

            # We need uncalibrated data to calculate the realized performance on.
            # https://github.com/NannyML/nannyml/issues/98
            data[f'uncalibrated_{self.y_pred_proba}'] = data[self.y_pred_proba]

            if self.needs_calibration:
                data[self.y_pred_proba] = self.calibrator.calibrate(data[self.y_pred_proba])
        else:
            assert isinstance(self.y_pred_proba, Dict)
            _list_missing([self.y_pred] + model_output_column_names(self.y_pred_proba), data)

            # We need uncalibrated data to calculate the realized performance on.
            # https://github.com/NannyML/nannyml/issues/98
            for class_proba in model_output_column_names(self.y_pred_proba):
                data[f'uncalibrated_{class_proba}'] = data[class_proba]

            data = _calibrate_predicted_probabilities(data, self.y_true, self.y_pred_proba, self._calibrators)

        chunks = self.chunker.split(data)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'period': 'analysis',
                    **self._estimate_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        metric_column_names = [name for metric in self.metrics for name in metric.column_names]
        multilevel_index = _create_multilevel_index(metric_names=metric_column_names)

        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self.result = Result(
                results_data=res,
                y_pred_proba=self.y_pred_proba,
                y_pred=self.y_pred,
                y_true=self.y_true,
                timestamp_column_name=self.timestamp_column_name,
                metrics=self.metrics,
                chunker=self.chunker,
                problem_type=self.problem_type,
            )
        else:
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, res], ignore_index=True)

        return self.result

    def _estimate_chunk(self, chunk: Chunk) -> Dict:
        chunk_records: Dict[str, Any] = {}
        for metric in self.metrics:
            chunk_record = metric.get_chunk_record(chunk.data)
            # add the chunk record to the chunk_records dict
            chunk_records.update(chunk_record)
        return chunk_records

    def _fit_binary(self, reference_data: pd.DataFrame) -> CBPE:
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        required_cols = [self.y_true, self.y_pred_proba]
        if self.y_pred is not None:
            required_cols.append(self.y_pred)
        _list_missing(required_cols, list(reference_data.columns))

        # We need uncalibrated data to calculate the realized performance on.
        # We need realized performance in threshold calculations.
        # https://github.com/NannyML/nannyml/issues/98
        reference_data[f'uncalibrated_{self.y_pred_proba}'] = reference_data[self.y_pred_proba]

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

        self.result = self._estimate(reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    def _fit_multiclass(self, reference_data: pd.DataFrame) -> CBPE:
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_true, self.y_pred] + model_output_column_names(self.y_pred_proba), reference_data)

        # We need uncalibrated data to calculate the realized performance on.
        # We need realized performance in threshold calculations.
        # https://github.com/NannyML/nannyml/issues/98
        for class_proba in model_output_column_names(self.y_pred_proba):
            reference_data[f'uncalibrated_{class_proba}'] = reference_data[class_proba]

        for metric in self.metrics:
            metric.fit(reference_data)

        assert isinstance(self.y_pred_proba, Dict)
        self._calibrators = _fit_calibrators(reference_data, self.y_true, self.y_pred_proba, self.calibrator)

        self.result = self._estimate(reference_data)
        self.result.data[('chunk', 'period')] = 'reference'
        return self


def _create_multilevel_index(metric_names: List[str]) -> MultiIndex:
    chunk_column_names = [
        'key',
        'chunk_index',
        'start_index',
        'end_index',
        'start_date',
        'end_date',
        'period',
    ]

    method_column_names = [
        'value',
        'sampling_error',
        'realized',
        'upper_confidence_boundary',
        'lower_confidence_boundary',
        'upper_threshold',
        'lower_threshold',
        'alert',
    ]

    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    reconstruction_tuples = [
        (metric_name, column_name) for metric_name in metric_names for column_name in method_column_names
    ]

    tuples = chunk_tuples + reconstruction_tuples

    return MultiIndex.from_tuples(tuples)


def _get_class_splits(
    data: pd.DataFrame, y_true: str, y_pred_proba: Dict[str, str], include_targets: bool = True
) -> List[Tuple]:
    classes = sorted(y_pred_proba.keys())
    y_trues: Dict[str, np.ndarray] = {}

    if include_targets:
        y_trues = {classes[idx]: (label_binarize(data[y_true], classes=classes).T[idx]) for idx in range(len(classes))}

    y_pred_probas = {clazz: data[y_pred_proba[clazz]] for clazz in classes}

    return [(cls, y_trues[cls] if include_targets else None, y_pred_probas[cls]) for cls in classes]


def _fit_calibrators(
    reference_data: pd.DataFrame, y_true_col: str, y_pred_proba_col: Dict[str, str], calibrator: Calibrator
) -> Dict[str, Calibrator]:
    fitted_calibrators = {}
    noop_calibrator = NoopCalibrator()

    for clazz, y_true, y_pred_proba in _get_class_splits(reference_data, y_true_col, y_pred_proba_col):
        _calibrator = copy.deepcopy(calibrator)
        if not needs_calibration(np.asarray(y_true), np.asarray(y_pred_proba), calibrator):
            _calibrator = noop_calibrator

        _calibrator.fit(y_pred_proba, y_true)
        fitted_calibrators[clazz] = copy.deepcopy(_calibrator)

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
    predicted_class_proba_column_names = [y_pred_proba[cls] for cls in sorted(y_pred_proba.keys())]
    for idx in range(number_of_classes):
        calibrated_data[predicted_class_proba_column_names[idx]] = calibrated_probas[:, idx]

    return calibrated_data


def raise_if_metrics_require_y_pred(metrics: List[str], y_pred: Optional[str]):
    """Raise an exception if metrics require y_pred and y_pred is not set.

    Current metrics that require 'y_pred' are:
    - roc_auc
    - average_precision
    """
    metrics_that_need_y_pred = [m for m in metrics if m not in ['roc_auc', 'average_precision']]

    if len(metrics_that_need_y_pred) > 0 and y_pred is None:
        raise InvalidArgumentsException(f"Metrics '{metrics_that_need_y_pred}' require 'y_pred' to be set.")
