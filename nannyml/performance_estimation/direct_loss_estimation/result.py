"""Module containing CBPE estimation results and plotting implementations."""

from typing import Any, Dict, List, Optional

import pandas as pd
from plotly.graph_objects import Figure

from nannyml._typing import Key
from nannyml.base import PerMetricResult
from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.performance_estimation.direct_loss_estimation.metrics import Metric
from nannyml.plots import Colors
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.usage_logging import UsageEvent, log_usage


class Result(PerMetricResult[Metric], ResultCompareMixin):
    """Contains results for CBPE estimation and adds filtering and plotting functionality."""

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
        """DLE Result Class.

        Parameters
        ----------
        results_data: pd.DataFrame
            Results data returned by a DLE estimator.
        metrics: List[nannyml.performance_estimation.direct_loss_estimation.metrics.Metric]
            List of metrics to evaluate.
        feature_column_names: List[str]
            A list of column names indicating which columns contain feature values.
        y_pred: str
            The name of the column containing your model predictions.
        y_true: str
            The name of the column containing target values (that are provided in reference data during fitting).
        chunker: Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        tune_hyperparameters: bool,
            A boolean controlling whether hypertuning should be performed on the internal regressor models
            whilst fitting on reference data.
            Tuning hyperparameters takes some time and does not guarantee better results, hence it defaults to `False`.
        hyperparameter_tuning_config: Dict[str, Any],
            A dictionary that allows you to provide a custom hyperparameter tuning configuration when
            `tune_hyperparameters` has been set to `True`.
            The following dictionary is the default tuning configuration. It can be used as a template to modify::

                {
                    "time_budget": 15,
                    "metric": "mse",
                    "estimator_list": ['lgbm'],
                    "eval_method": "cv",
                    "hpo_method": "cfo",
                    "n_splits": 5,
                    "task": 'regression',
                    "seed": 1,
                    "verbose": 0,
                }

            For an overview of possible parameters for the tuning process check out the
            `FLAML documentation <https://microsoft.github.io/FLAML/docs/reference/automl#automl-objects>`_.
        hyperparameters: Dict[str, Any],
            A dictionary used to provide your own custom hyperparameters when `tune_hyperparameters` has
            been set to `True`.
            Check out the available hyperparameter options in the
            `LightGBM documentation <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html>`_.
        timestamp_column_name: str, default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        """
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
        """Creates a list of keys where each Key is a `namedtuple('Key', 'properties display_names')`."""
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
        """Render plots based on DLE estimation results.

        This function will return a :class:`plotly.graph_objects.Figure` object.
        The following kinds of plots are available:

        Parameters
        ----------
        kind: str, default='performance'
            What kind of plot to create, currently only performance is supported.

        Raises
        ------
        InvalidArgumentsException: when an unknown plot ``kind`` is provided.

        Returns
        -------
        fig: :class:`plotly.graph_objs._figure.Figure`
            A :class:`~plotly.graph_objs._figure.Figure` object containing the requested drift plot.

            Can be saved to disk using the :meth:`~plotly.graph_objs._figure.Figure.write_image` method
            or shown rendered on screen using the :meth:`~plotly.graph_objs._figure.Figure.show` method.

        Examples
        --------
        >>> import nannyml as nml
        >>>
        >>> reference_df, analysis_df, _ = nml.load_synthetic_car_price_dataset()
        >>>
        >>> estimator = nml.DLE(
        ...     feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count',
        ...                           'door_count', 'fuel', 'transmission'],
        ...     y_pred='y_pred',
        ...     y_true='y_true',
        ...     timestamp_column_name='timestamp',
        ...     metrics=['rmse', 'rmsle'],
        ...     chunk_size=6000,
        >>> )
        >>>
        >>> estimator.fit(reference_df)
        >>>
        >>> results = estimator.estimate(analysis_df)
        >>> results.plot().show()
        """
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
