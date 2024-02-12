#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates realized performance metrics when target data is available.

The performance calculator manages a list of :class:`~nannyml.performance_calculation.metrics.base.Metric` instances,
constructed using the :class:`~nannyml.performance_calculation.metrics.base.MetricFactory`.
The estimator is then responsible for delegating the `fit` and `estimate` method calls to each of the managed
:class:`~nannyml.performance_calculation.metrics.base.Metric` instances and building a
:class:`~nannyml.performance_calculation.result.Result` object.

For more information, check out the `tutorials`_.

.. _tutorials:
    https://nannyml.readthedocs.io/en/stable/tutorials/performance_calculation.html

Examples
--------
>>> import nannyml as nml
>>> from IPython.display import display
>>> reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
>>> analysis_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)
>>> display(reference_df.head(3))
>>> calc = nml.PerformanceCalculator(
...     y_pred_proba='y_pred_proba',
...     y_pred='y_pred',
...     y_true='repaid',
...     timestamp_column_name='timestamp',
...     problem_type='classification_binary',
...     metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
...     chunk_size=5000)
>>> calc.fit(reference_df)
>>> results = calc.calculate(analysis_df)
>>> display(results.filter(period='analysis').to_df())
>>> display(results.filter(period='reference').to_df())
>>> figure = results.plot()
>>> figure.show()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import MultiIndex

from nannyml._typing import ModelOutputsType, ProblemType
from nannyml.base import AbstractCalculator
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import CalculatorNotFittedException, InvalidArgumentsException
from nannyml.performance_calculation import SUPPORTED_METRIC_VALUES
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory
from nannyml.performance_calculation.result import Result
from nannyml.thresholds import StandardDeviationThreshold, Threshold
from nannyml.usage_logging import UsageEvent, log_usage

TARGET_COMPLETENESS_RATE_COLUMN_NAME = 'NML_TARGET_INCOMPLETE'

DEFAULT_THRESHOLDS: Dict[str, Threshold] = {
    'roc_auc': StandardDeviationThreshold(),
    'f1': StandardDeviationThreshold(),
    'precision': StandardDeviationThreshold(),
    'recall': StandardDeviationThreshold(),
    'specificity': StandardDeviationThreshold(),
    'accuracy': StandardDeviationThreshold(),
    'confusion_matrix': StandardDeviationThreshold(),
    'business_value': StandardDeviationThreshold(),
    'mae': StandardDeviationThreshold(),
    'mape': StandardDeviationThreshold(),
    'mse': StandardDeviationThreshold(),
    'msle': StandardDeviationThreshold(),
    'rmse': StandardDeviationThreshold(),
    'rmsle': StandardDeviationThreshold(),
}


class PerformanceCalculator(AbstractCalculator):
    """Calculates realized performance metrics when target data is available."""

    def __init__(
        self,
        metrics: Union[str, List[str]],
        y_true: str,
        y_pred: str,
        problem_type: Union[str, ProblemType],
        y_pred_proba: Optional[ModelOutputsType] = None,
        timestamp_column_name: Optional[str] = None,
        thresholds: Optional[Dict[str, Threshold]] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        normalize_confusion_matrix: Optional[str] = None,
        business_value_matrix: Optional[Union[List, np.ndarray]] = None,
        normalize_business_value: Optional[str] = None,
    ):
        """Creates a new performance calculator.

        Parameters
        ----------
        metrics: Union[str, List[str]]
            A metric or list of metrics to calculate.
        y_true: str
            The name of the column containing target values.
        y_pred: str
            The name of the column containing your model predictions.
        problem_type: Union[str, ProblemType]
            Determines which method to use. Allowed values are:

                - 'regression'
                - 'classification_binary'
                - 'classification_multiclass'

        y_pred_proba: ModelOutputsType, default=None
            Name(s) of the column(s) containing your model output.
            Pass a single string when there is only a single model output column, e.g. in binary classification cases.
            Pass a dictionary when working with multiple output columns, e.g. in multiclass classification cases.
            The dictionary maps a class/label string to the column name containing model outputs for that class/label.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
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
                    'mae': StandardDeviationThreshold(),
                    'mape': StandardDeviationThreshold(),
                    'mse': StandardDeviationThreshold(),
                    'msle': StandardDeviationThreshold(),
                    'rmse': StandardDeviationThreshold(),
                    'rmsle': StandardDeviationThreshold(),
                }

            A dictionary allowing users to set a custom threshold for each method. It links a `Threshold` subclass
            to a method name. This dictionary is optional.
            When a dictionary is given its values will override the default values. If no dictionary is given a default
            will be applied.
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
        normalize_confusion_matrix: str, default=None
            Determines how the confusion matrix will be normalized. Allowed values are None, 'all', 'true' and
            'predicted'. If None, the confusion matrix will not be normalized and the counts for each cell of
            the matrix will be returned. If 'all', the confusion matrix will be normalized by the total number
            of observations. If 'true', the confusion matrix will be normalized by the total number of
            observations for each true class. If 'predicted', the confusion matrix will be normalized by the
            total number of observations for each predicted class.
        business_value_matrix: Optional[Union[List, np.ndarray]], default=None
            A matrix containing the business costs for each combination of true and predicted class.
            The i-th row and j-th column entry of the matrix contains the business cost for predicting the
            i-th class as the j-th class. The matrix must have the same number of rows and columns as the number
            of classes in the problem.
        normalize_business_value: str, default=None
            Determines how the business value will be normalized. Allowed values are None and
            'per_prediction'. If None, the business value will not be normalized and the value
            returned will be the total value per chunk. If 'per_prediction', the value will be normalized
            by the number of predictions in the chunk.

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df, analysis_df, analysis_targets_df = nml.load_synthetic_car_loan_dataset()
        >>> analysis_df = analysis_df.merge(analysis_targets_df, left_index=True, right_index=True)
        >>> display(reference_df.head(3))
        >>> calc = nml.PerformanceCalculator(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='repaid',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_binary',
        ...     metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
        ...     chunk_size=5000)
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> display(results.filter(period='analysis').to_df())
        >>> display(results.filter(period='reference').to_df())
        >>> figure = results.plot()
        >>> figure.show()
        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name)

        self.y_true = y_true
        self.y_pred = y_pred

        self.y_pred_proba = y_pred_proba

        if isinstance(problem_type, str):
            self.problem_type = ProblemType.parse(problem_type)
        else:
            self.problem_type = problem_type

        if self.problem_type is not ProblemType.REGRESSION and y_pred_proba is None:
            raise InvalidArgumentsException(f"'y_pred_proba' can not be 'None' for problem type {ProblemType.value}")

        self.thresholds = DEFAULT_THRESHOLDS
        if thresholds:
            self.thresholds.update(**thresholds)

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

        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            if metric not in SUPPORTED_METRIC_VALUES:
                raise InvalidArgumentsException(f"Metric '{metric}' is not supported.")

        self.metrics: List[Metric] = [
            MetricFactory.create(
                m,
                self.problem_type,
                y_true=y_true,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                threshold=self.thresholds[m],
                normalize_confusion_matrix=normalize_confusion_matrix,
                business_value_matrix=business_value_matrix,
                normalize_business_value=normalize_business_value,
            )
            for m in metrics
        ]

        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None

        self.result: Optional[Result] = None

    def __str__(self):  # noqa: D105
        return f"PerformanceCalculator[metrics={str(self.metrics)}]"

    @log_usage(UsageEvent.PERFORMANCE_CALC_FIT, metadata_from_self=['metrics', 'problem_type'])
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> PerformanceCalculator:
        """Fits the calculator on the reference data, calibrating it for further use on the full dataset."""
        if reference_data.empty:
            raise InvalidArgumentsException('reference data contains no rows. Provide a valid reference data set.')

        if self.y_true not in reference_data.columns:
            raise InvalidArgumentsException(
                f"target data column '{self.y_true}' not found in data columns: {reference_data.columns}."
            )

        reference_data = reference_data.copy()

        # data validation is performed during the _fit for each metric

        for metric in self.metrics:
            try:
                metric.fit(reference_data=reference_data, chunker=self.chunker)
            except Exception as exc:
                self._logger.error(
                    f"an unexpected error occurred when calculating metric '{metric.display_name}': {exc}"
                )
                continue
        self.previous_reference_data = reference_data

        self.result = self._calculate(reference_data)
        self.result.data[('chunk', 'period')] = 'reference'
        self.result.reference_data = reference_data.copy()

        return self

    @log_usage(UsageEvent.PERFORMANCE_CALC_RUN, metadata_from_self=['metrics', 'problem_type'])
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates performance on the analysis data, using the metrics specified on calculator creation."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        if self.y_true not in data.columns:
            raise InvalidArgumentsException(f"data does not contain target data column '{self.y_true}'.")

        data = data.copy(deep=True)

        # Setup for target completeness rate
        data['NML_TARGET_INCOMPLETE'] = data[self.y_true].isna().astype(np.int16)

        # Generate chunks
        if self.chunker is None:
            raise CalculatorNotFittedException(
                'chunker has not been set. '
                'Please ensure you run ``calculator.fit()`` '
                'before running ``calculator.calculate()``'
            )
        chunks = self.chunker.split(data)

        # Construct result frame
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
                    'targets_missing_rate': chunk.data[TARGET_COMPLETENESS_RATE_COLUMN_NAME].sum()
                    / chunk.data[TARGET_COMPLETENESS_RATE_COLUMN_NAME].count(),
                    **self._calculate_metrics_for_chunk(chunk),
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
                metrics=self.metrics,
                y_true=self.y_true,
                y_pred=self.y_pred,
                y_pred_proba=self.y_pred_proba,
                timestamp_column_name=self.timestamp_column_name,
                problem_type=self.problem_type,
            )
        else:
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)

        return self.result

    def _calculate_metrics_for_chunk(self, chunk: Chunk) -> Dict:
        chunk_records: Dict[str, Any] = {}
        for metric in self.metrics:
            chunk_record = metric.get_chunk_record(chunk.data)
            chunk_records.update(chunk_record)
        return chunk_records


def _create_multilevel_index(metric_names: List[str]):
    chunk_column_names = [
        'key',
        'chunk_index',
        'start_index',
        'end_index',
        'start_date',
        'end_date',
        'period',
        'targets_missing_rate',
    ]
    method_column_names = [
        'sampling_error',
        'value',
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
