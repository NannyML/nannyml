import copy
from typing import Any, Dict, List, Optional

import pandas as pd
from plotly.graph_objects import Figure

from nannyml._typing import Key
from nannyml._typing import Result as ResultType
from nannyml.base import Abstract1DResult
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation.metrics import Metric
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.usage_logging import UsageEvent, log_usage


class Result(Abstract1DResult, ResultCompareMixin):
    def __init__(
        self,
        results_data: pd.DataFrame,
        metrics: List[Metric],
        feature_column_names: List[str],
        y_pred: str,
        y_true: str,
        chunker: Chunker,
        tune_hyperparameters: bool,
        hyperparameter_tuning_config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]],
        timestamp_column_name: Optional[str] = None,
    ):
        super().__init__(results_data)

        self.metrics = metrics
        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name

        self.chunker = chunker

        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_tuning_config = (hyperparameter_tuning_config,)
        self.hyperparameters = hyperparameters

    def _filter(self, period: str, metrics: Optional[List[str]] = None, *args, **kwargs) -> ResultType:
        if metrics is None:
            metrics = [metric.column_name for metric in self.metrics]

        data = pd.concat([self.data.loc[:, (['chunk'])], self.data.loc[:, (metrics,)]], axis=1)

        if period != 'all':
            data = self.data.loc[self.data.loc[:, ('chunk', 'period')] == period, :]

        data = data.reset_index(drop=True)
        res = copy.deepcopy(self)
        res.data = data
        res.metrics = [m for m in self.metrics if m.column_name in metrics]

        return res

    def keys(self) -> List[Key]:
        return [
            Key(properties=(metric.column_name,), display_names=(f'estimated {metric.display_name}',))
            for metric in self.metrics
        ]

    @log_usage(UsageEvent.DLE_PLOT, metadata_from_kwargs=['kind'])
    def plot(
        self,
        kind: str = 'performance',
        *args,
        **kwargs,
    ) -> Figure:
        if kind == 'performance':
            return plot_metrics(
                self,
                title='Estimated performance <b>(DLE)</b>',
                subplot_title_format='Estimated <b>{display_names}</b>',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")
