#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the data reconstruction drift calculation and provides plotting functionality."""
from __future__ import annotations

from collections import namedtuple
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from nannyml._typing import Key
from nannyml.base import PerMetricResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.metrics import plot_metric
from nannyml.usage_logging import UsageEvent, log_usage

Metric = namedtuple("Metric", "display_name column_name")


class Result(PerMetricResult[Metric], ResultCompareMixin):
    """Class wrapping the results of the data reconstruction drift calculator and providing plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        categorical_column_names: List[str],
        continuous_column_names: List[str],
        timestamp_column_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        results_data: pd.DataFrame
            Results data returned by a DataReconstructionDriftCalculator.
        column_names: List[str]
            A list of column names indicating which columns contain feature values.
        categorical_column_names : List[str]
            Subset of categorical features to be included in calculation.
        continuous_column_names : List[str]
            Subset of continuous features to be included in calculation.
        timestamp_column_name: Optional[str], default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        """
        metric = Metric(display_name='Reconstruction error', column_name='reconstruction_error')
        super().__init__(results_data, [metric])

        self.column_names = column_names
        self.categorical_column_names = categorical_column_names
        self.continuous_column_names = continuous_column_names
        self.timestamp_column_name = timestamp_column_name

    def keys(self) -> List[Key]:
        """
        Creates a list of keys where each Key is a `namedtuple('Key', 'properties display_names')`
        """
        return [Key(properties=('reconstruction_error',), display_names=('Reconstruction error',))]

    @log_usage(UsageEvent.MULTIVAR_DRIFT_PLOT, metadata_from_kwargs=['kind'])
    def plot(self, kind: str = 'drift', *args, **kwargs) -> go.Figure:
        """Renders plots for metrics returned by the multivariate data reconstruction calculator.

        Parameters
        ----------
        kind: str, default='drift'
            The kind of plot you want to have. Value can currently only be 'drift'.

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
        >>> # Load synthetic data
        >>> reference, analysis, _ = nml.load_synthetic_car_loan_dataset()
        >>> non_feature_columns = ['timestamp', 'y_pred_proba', 'y_pred', 'repaid']
        >>> feature_column_names = [
        ...     col for col in reference.columns
        ...     if col not in non_feature_columns
        >>> ]
        >>> calc = nml.DataReconstructionDriftCalculator(
        ...     column_names=feature_column_names,
        ...     timestamp_column_name='timestamp',
        ...     chunk_size=5000
        >>> )
        >>> calc.fit(reference)
        >>> results = calc.calculate(analysis)
        >>> figure = results.plot()
        >>> figure.show()
        """
        if kind == 'drift':
            return plot_metric(
                self,
                title='Multivariate Drift (PCA Reconstruction Error)',
                metric_display_name='Reconstruction Error',
                metric_column_name='reconstruction_error',
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['drift'].")
