#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, List

import pandas as pd

from nannyml._typing import UseCase
from nannyml.base import AbstractEstimator, AbstractEstimatorResult, _list_missing
from nannyml.chunk import Chunk, Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_error_estimation import DEFAULT_METRICS
from nannyml.performance_estimation.direct_error_estimation.metrics import Metric, MetricFactory
from nannyml.performance_estimation.direct_error_estimation.result import Result


class DEE(AbstractEstimator):
    def __init__(
        self,
        feature_column_names: List[str],
        y_pred: str,
        y_true: str,
        timestamp_column_name: str,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        metrics: List[str] = None,
        tune_hyperparameters: bool = False,
        hyperparameter_tuning_config: Dict[str, Any] = None,
        hyperparameters: Dict[str, Any] = None,
    ):
        super().__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name

        if metrics is None:
            metrics = DEFAULT_METRICS
        self.metrics: List[Metric] = [
            MetricFactory.create(metric, UseCase.REGRESSION, kwargs={'estimator': self}) for metric in metrics
        ]

        if hyperparameter_tuning_config is None:
            hyperparameter_tuning_config = {
                "time_budget": 15,  # total running time in seconds
                "metric": "mse",
                "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
                "eval_method": "cv",  # resampling strategy
                "hpo_method": "cfo",  # hyperparameter optimization method, cfo is default.
                "n_splits": 5,  # Default Value is 5
                "task": 'regression',  # task type
                "seed": 1,  # random seed
                "verbose": 0,
            }
        self.hyperparameter_tuning_config = hyperparameter_tuning_config
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameters = hyperparameters

    def __str__(self):
        return (
            f"{self.__class__.__name__}[tune_hyperparameters={self.tune_hyperparameters}, "
            f"metrics={[str(m) for m in self.metrics]}]"
        )

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_true, self.y_pred], list(reference_data.columns))

        for metric in self.metrics:
            metric.fit(reference_data)

        self.previous_reference_results = self._estimate(reference_data).data

        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> AbstractEstimatorResult:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing([self.y_pred], list(data.columns))

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
        return Result(results_data=res, estimator=self)

    def _estimate_chunk(self, chunk: Chunk) -> Dict:
        estimates: Dict[str, Any] = {}
        for metric in self.metrics:
            estimated_metric = metric.estimate(chunk.data)
            sampling_error = metric.sampling_error(chunk.data)
            estimates[f'realized_{metric.column_name}'] = metric.realized_performance(chunk.data)
            estimates[f'estimated_{metric.column_name}'] = estimated_metric
            estimates[f'upper_confidence_{metric.column_name}'] = estimated_metric + 3 * sampling_error
            estimates[f'lower_confidence_{metric.column_name}'] = estimated_metric - 3 * sampling_error
            estimates[f'sampling_error_{metric.column_name}'] = sampling_error
            estimates[f'upper_threshold_{metric.column_name}'] = metric.upper_threshold
            estimates[f'lower_threshold_{metric.column_name}'] = metric.lower_threshold
            estimates[f'alert_{metric.column_name}'] = (
                estimated_metric > metric.upper_threshold or estimated_metric < metric.lower_threshold
            )
        return estimates
