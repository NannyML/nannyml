from typing import Any, Dict, List, Optional

import pandas as pd
from plotly.graph_objects import Figure

from nannyml._typing import Key
from nannyml.base import Abstract1DResult
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation.metrics import Metric
from nannyml.plots import Colors
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.usage_logging import UsageEvent, log_usage


class Result(Abstract1DResult[Metric], ResultCompareMixin):
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
        super().__init__(results_data, metrics)

        self.feature_column_names = feature_column_names
        self.y_pred = y_pred
        self.y_true = y_true
        self.timestamp_column_name = timestamp_column_name

        self.chunker = chunker

        self.tune_hyperparameters = tune_hyperparameters
        self.hyperparameter_tuning_config = (hyperparameter_tuning_config,)
        self.hyperparameters = hyperparameters

    def keys(self) -> List[Key]:
        return [
            Key(
                properties=(metric.column_name,),
                display_names=(f'estimated {metric.display_name}', metric.display_name),
            )
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
                subplot_title_format='Estimated <b>{display_names[1]}</b>',
                subplot_y_axis_title_format='{display_names[1]}',
                color=Colors.INDIGO_PERSIAN,
                line_dash='dash',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide on of: ['performance'].")
