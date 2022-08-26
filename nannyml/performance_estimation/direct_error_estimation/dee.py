#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from base import AbstractEstimatorResult, _raise_exception_for_negative_values
from lightgbm import LGBMRegressor

from nannyml.base import AbstractEstimator
from nannyml.chunk import Chunker

DEFAULT_METRICS = ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']


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
        tune_hyperparameters: bool = True,
        hypertuning_config: Dict[str, Any] = None,
        hyperparameters: Dict[str, Any] = None,
    ):
        super().__init__(chunk_size, chunk_number, chunk_period, chunker)

        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true

        if metrics is None:
            metrics = DEFAULT_METRICS
        # TODO: validate metrics input
        self.metrics = metrics

        if hypertuning_config is None:
            hypertuning_config = {
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
        self.hypertuning_config = hypertuning_config
        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameters = hyperparameters

        self._metric_models: Dict[str, LGBMRegressor] = {}

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> AbstractEstimator:
        for metric in self.metrics:
            observation_level_metric = _calculate_observation_level_metric(
                reference_data[self.y_true], reference_data[self.y_pred], metric
            )

            self._metric_models[metric] = _train_direct_error_estimation_model(
                X_train=reference_data[self.feature_column_names + [self.y_true]],
                y_train=observation_level_metric,
                tune_hyperparameters=self.tune_hyperparameters,
                hypertuning_config=self.hypertuning_config,
                hyperparameters=self.hyperparameters,
            )

        return self

    def _estimate(self, data: pd.DataFrame, *args, **kwargs) -> AbstractEstimatorResult:
        pass


def _calculate_observation_level_metric(y_true: pd.Series, y_pred: pd.Series, metric: str) -> pd.Series:
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    if metric in ['rmse', 'mse']:
        return np.square(y_true_arr - y_pred_arr)
    elif metric in ['mae']:
        return abs(y_true_arr - y_pred_arr)
    elif metric in ['mape']:
        epsilon = np.finfo(np.float64).eps
        return abs(y_true_arr - y_pred_arr) / (np.maximum(epsilon, abs(y_true_arr)))
    elif metric in ['rmsle', 'rmse']:
        _raise_exception_for_negative_values(y_true)
        _raise_exception_for_negative_values(y_pred)
        return np.square(np.log1p(y_true_arr) - np.log1p(y_pred_arr))


def _train_direct_error_estimation_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool,
    hypertuning_config: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]],
) -> LGBMRegressor:
    pass
