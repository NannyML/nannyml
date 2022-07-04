#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for performance calculation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from nannyml._typing import ModelOutputsType, derive_use_case
from nannyml.base import AbstractCalculator
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import CalculatorNotFittedException, InvalidArgumentsException
from nannyml.performance_calculation.metrics import Metric, MetricFactory
from nannyml.performance_calculation.result import PerformanceCalculatorResult

TARGET_COMPLETENESS_RATE_COLUMN_NAME = 'NML_TARGET_INCOMPLETE'


class PerformanceCalculator(AbstractCalculator):
    """Base class for performance metric calculation."""

    def __init__(
        self,
        timestamp_column_name: str,
        metrics: List[str],
        y_true: str,
        y_pred_proba: Optional[ModelOutputsType],
        y_pred: Optional[str],
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
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
        timestamp_column_name: str
            The name of the column containing the timestamp of the model prediction.
        metrics: List[str]
            A list of metrics to calculate.
        chunk_size: int
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df)
        >>> # create a new calculator, chunking by week
        >>> calculator = nml.PerformanceCalculator(model_metadata=metadata, chunk_period='W')

        """
        super().__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.timestamp_column_name = timestamp_column_name
        self.metrics: List[Metric] = [
            MetricFactory.create(m, derive_use_case(self.y_pred_proba), {'calculator': self})  # type: ignore
            for m in metrics
        ]

        self._minimum_chunk_size = None
        self.previous_reference_data: Optional[pd.DataFrame] = None
        self.previous_reference_results: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> PerformanceCalculator:
        """Fits the calculator on the reference data, calibrating it for further use on the full dataset.

        Parameters
        ----------
        reference_data: pd.DataFrame
            Reference data for the model, i.e. model inputs and predictions enriched with target data.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df)
        >>> calculator = nml.PerformanceCalculator(model_metadata=metadata, chunk_period='W')
        >>> # fit the calculator on reference data
        >>> calculator.fit(ref_df)
        """
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

        self._minimum_chunk_size = np.max([metric.minimum_chunk_size() for metric in self.metrics])

        self.previous_reference_data = reference_data
        self.previous_reference_results = self._calculate(reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> PerformanceCalculatorResult:
        """Calculates performance on the analysis data, using the metrics specified on calculator creation.

        Parameters
        ----------
        analysis_data: pd.DataFrame
            Analysis data for the model, i.e. model inputs and predictions.

        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df)
        >>> calculator = nml.PerformanceCalculator(model_metadata=metadata, chunk_period='W')
        >>> calculator.fit(ref_df)
        >>> # calculate realized performance on analysis data
        >>> realized_performance = calculator.calculate(ana_df)
        """
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
        chunks = self.chunker.split(
            data,
            minimum_chunk_size=self._minimum_chunk_size,
            timestamp_column_name=self.timestamp_column_name,
        )

        # Construct result frame
        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'period': 'analysis' if chunk.is_transition else chunk.period,
                    'targets_missing_rate': chunk.data[TARGET_COMPLETENESS_RATE_COLUMN_NAME].sum()
                    / chunk.data[TARGET_COMPLETENESS_RATE_COLUMN_NAME].count(),
                    **self._calculate_metrics_for_chunk(chunk),
                }
                for chunk in chunks
            ]
        )

        return PerformanceCalculatorResult(results_data=res, calculator=self)

    def _calculate_metrics_for_chunk(self, chunk: Chunk) -> Dict:
        metrics_results = {}
        for metric in self.metrics:
            chunk_metric = metric.calculate(chunk.data)
            metrics_results[metric.column_name] = chunk_metric
            metrics_results[f'{metric.column_name}_lower_threshold'] = metric.lower_threshold
            metrics_results[f'{metric.column_name}_upper_threshold'] = metric.upper_threshold
            metrics_results[f'{metric.column_name}_alert'] = (
                metric.lower_threshold > chunk_metric or chunk_metric > metric.upper_threshold
            )

        return metrics_results
