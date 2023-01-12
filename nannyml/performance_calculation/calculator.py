#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates realized performance metrics when target data is available."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import MultiIndex

from nannyml._typing import ModelOutputsType, ProblemType
from nannyml.base import AbstractCalculator
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import CalculatorNotFittedException, InvalidArgumentsException
from nannyml.performance_calculation.metrics.base import Metric, MetricFactory
from nannyml.performance_calculation.result import Result
from nannyml.usage_logging import UsageEvent, log_usage

TARGET_COMPLETENESS_RATE_COLUMN_NAME = 'NML_TARGET_INCOMPLETE'

SUPPORTED_METRICS = list(MetricFactory.registry.keys())


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
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
    ):
        """Creates a new performance calculator.

        Parameters
        ----------
        y_true: str
            The name of the column containing target values.
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

        Examples
        --------
        >>> import nannyml as nml
        >>> from IPython.display import display
        >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
        >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
        >>> analysis_target_df = nml.load_synthetic_binary_classification_dataset()[2]
        >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')
        >>> display(reference_df.head(3))
        >>> calc = nml.PerformanceCalculator(
        ...     y_pred_proba='y_pred_proba',
        ...     y_pred='y_pred',
        ...     y_true='work_home_actual',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='classification_binary',
        ...     metrics=['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'],
        ...     chunk_size=5000)
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> display(results.data)
        >>> display(results.calculator.previous_reference_results)
        >>> for metric in calc.metrics:
        ...     figure = results.plot(kind='performance', plot_reference=True, metric=metric)
        ...     figure.show()
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

        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics: List[Metric] = [
            MetricFactory.create(m, self.problem_type, y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
            for m in metrics  # type: ignore
        ]

        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None

        self.result: Optional[Result] = None

    def __str__(self):
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
            metric.fit(reference_data=reference_data, chunker=self.chunker)

        self.previous_reference_data = reference_data
        self.result = self._calculate(reference_data)

        assert self.result is not None

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

        data = data.copy()

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

        multilevel_index = _create_multilevel_index(metric_names=[metric.column_name for metric in self.metrics])
        res.columns = multilevel_index

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
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)
            self.result.analysis_data = data.copy()

        return self.result

    def _calculate_metrics_for_chunk(self, chunk: Chunk) -> Dict:
        metrics_results = {}
        for metric in self.metrics:
            chunk_metric = metric.calculate(chunk.data)
            metrics_results[f'{metric.column_name}_sampling_error'] = metric.sampling_error(chunk.data)
            metrics_results[metric.column_name] = chunk_metric
            metrics_results[f'{metric.column_name}_upper_threshold'] = metric.upper_threshold
            metrics_results[f'{metric.column_name}_lower_threshold'] = metric.lower_threshold
            metrics_results[f'{metric.column_name}_alert'] = (
                metric.lower_threshold > chunk_metric if metric.lower_threshold else False
            ) or (chunk_metric > metric.upper_threshold if metric.upper_threshold else False)

        return metrics_results


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
