#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Implementation of the CBPE estimator."""
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
from nannyml.performance_estimation.confidence_based.metrics import MetricFactory
from nannyml.performance_estimation.confidence_based.results import SUPPORTED_METRIC_VALUES, Result
from nannyml.sampling_error import SAMPLING_ERROR_RANGE
from nannyml.usage_logging import UsageEvent, log_usage


class CBPE(AbstractEstimator):
    """Performance estimator using the Confidence Based Performance Estimation (CBPE) technique."""

    # def __new__(cls, y_pred_proba: ModelOutputsType, problem_type: Union[str, ProblemType], *args, **kwargs):
    #     """Creates a new CBPE subclass instance based on the type of the provided ``model_metadata``."""
    #     from ._cbpe_binary_classification import _BinaryClassificationCBPE
    #     from ._cbpe_multiclass_classification import _MulticlassClassificationCBPE
    #
    #     if isinstance(problem_type, str):
    #         problem_type = ProblemType.parse(problem_type)
    #
    #     if problem_type is ProblemType.CLASSIFICATION_BINARY:
    #         return super(CBPE, cls).__new__(_BinaryClassificationCBPE)
    #     elif problem_type is ProblemType.CLASSIFICATION_MULTICLASS:
    #         return super(CBPE, cls).__new__(_MulticlassClassificationCBPE)
    #     else:
    #         raise NotImplementedError

    def __init__(
        self,
        metrics: Union[str, List[str]],
        y_pred: str,
        y_pred_proba: ModelOutputsType,
        y_true: str,
        problem_type: Union[str, ProblemType],
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        calibration: Optional[str] = None,
        calibrator: Optional[Calibrator] = None,
    ):
        """Initializes a new CBPE performance estimator.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values (that are provided in reference data during fitting).
        y_pred_proba: ModelOutputsType
            Name(s) of the column(s) containing your model output.
            Pass a single string when there is only a single model output column, e.g. in binary classification cases.
            Pass a dictionary when working with multiple output columns, e.g. in multiclass classification cases.
            The dictionary maps a class/label string to the column name containing model outputs for that class/label.
        y_pred: str
            The name of the column containing your model predictions.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
        metrics: Union[str, List[str]]
            A metric or list of metrics to calculate.
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
        problem_type: Union[str, ProblemType]
            Determines which CBPE implementation to use. Allowed problem type values are 'classification_binary' and
            'classification_multiclass'.

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
        >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
        >>> display(reference_df.head(3))
        >>> estimator = nml.CBPE(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='work_home_actual',
        ...     timestamp_column_name='timestamp',
        ...     metrics=['roc_auc', 'f1'],
        ...     chunk_size=5000,
        ...     problem_type='classification_binary',
        >>> )
        >>> estimator.fit(reference_df)
        >>> results = estimator.estimate(analysis_df)
        >>> display(results.data)
        >>> for metric in estimator.metrics:
        ...     metric_fig = results.plot(kind='performance', metric=metric)
        ...     metric_fig.show()
        >>> for metric in estimator.metrics:
        ...     metric_fig = results.plot(kind='performance', plot_reference=True, metric=metric)
        ...     metric_fig.show()
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

        if isinstance(problem_type, str):
            self.problem_type = ProblemType.parse(problem_type)
        else:
            self.problem_type = problem_type

        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = [
            MetricFactory.create(
                metric,
                self.problem_type,
                y_pred_proba=self.y_pred_proba,
                y_pred=self.y_pred,
                y_true=self.y_true,
                timestamp_column_name=self.timestamp_column_name,
                chunker=self.chunker,
            )
            for metric in metrics
        ]

        self.confidence_upper_bound = 1
        self.confidence_lower_bound = 0
        self._alert_thresholds: Dict[str, Tuple[float, float]] = {}
        self.needs_calibration: bool = False

        if calibrator is None:
            calibrator = CalibratorFactory.create(calibration)

        # Used in binary cases
        # TODO: unify this with multiclass case (or remove from public interface)
        self.calibrator = calibrator

        # Used in multiclass cases
        self._calibrators: Dict[str, Calibrator] = {}

        self.result: Optional[Result] = None

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls, y_pred_proba=self.y_pred_proba, problem_type=self.problem_type)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

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

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # create a new estimator and fit it on reference data
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W').fit(ref_df)

        """
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

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> # create a new estimator and fit it on reference data
        >>> estimator = nml.CBPE(model_metadata=metadata, chunk_period='W').fit(ref_df)
        >>> estimates = estimator.estimate(data)
        """
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.problem_type == ProblemType.CLASSIFICATION_BINARY:
            _list_missing([self.y_pred, self.y_pred_proba], data)

            # We need uncalibrated data to calculate the realized performance on.
            # https://github.com/NannyML/nannyml/issues/98
            data[f'uncalibrated_{self.y_pred_proba}'] = data[self.y_pred_proba]

            assert isinstance(self.y_pred_proba, str)
            if self.needs_calibration:
                data[self.y_pred_proba] = self.calibrator.calibrate(data[self.y_pred_proba])
        else:
            _list_missing([self.y_pred] + model_output_column_names(self.y_pred_proba), data)

            # We need uncalibrated data to calculate the realized performance on.
            # https://github.com/NannyML/nannyml/issues/98
            for class_proba in model_output_column_names(self.y_pred_proba):
                data[f'uncalibrated_{class_proba}'] = data[class_proba]

            assert isinstance(self.y_pred_proba, Dict)
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

        multilevel_index = _create_multilevel_index(metric_names=[m.column_name for m in self.metrics])
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
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)

        return self.result

    def _estimate_chunk(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = metric.estimate(chunk.data)
            sampling_error = metric.sampling_error(chunk.data)
            estimates[f'sampling_error_{metric.column_name}'] = sampling_error
            estimates[f'realized_{metric.column_name}'] = metric.realized_performance(chunk.data)
            estimates[f'estimated_{metric.column_name}'] = estimated_metric
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

    def _fit_binary(self, reference_data: pd.DataFrame) -> CBPE:
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_true, self.y_pred_proba, self.y_pred], list(reference_data.columns))

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
        assert self.result
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
        assert self.result
        self.result.data[('chunk', 'period')] = 'reference'
        return self


def _create_multilevel_index(metric_names: List[str]):
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
        'sampling_error',
        'realized',
        'value',
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
        fitted_calibrators[clazz] = copy.deepcopy(calibrator)

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
