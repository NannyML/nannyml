#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
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
from nannyml.plots.components import Hover
from nannyml.usage_logging import UsageEvent, log_usage

Metric = namedtuple("Metric", "display_name column_name")


class Result(PerMetricResult[Metric], ResultCompareMixin):
    """Class wrapping the results of the classifier for drift detection and providing plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        categorical_column_names: List[str],
        continuous_column_names: List[str],
        timestamp_column_name: Optional[str] = None,
    ):
        """Initialize a DomainClassifierCalculator results object.

        Parameters
        ----------
        results_data: pd.DataFrame
            Results data returned by a DomainClassifierCalculator.
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
        metric = Metric(display_name='Domain Classifier', column_name='domain_classifier_auroc')
        super().__init__(results_data, [metric])

        self.column_names = column_names
        self.categorical_column_names = categorical_column_names
        self.continuous_column_names = continuous_column_names
        self.timestamp_column_name = timestamp_column_name

    def keys(self) -> List[Key]:
        """Create a list of keys where each Key is a `namedtuple('Key', 'properties display_names')`."""
        return [Key(properties=('domain_classifier_auroc',), display_names=('Classifier AUROC ',))]

    @log_usage(UsageEvent.DC_RESULTS_PLOT, metadata_from_kwargs=['kind'])
    def plot(self, kind: str = 'drift', *args, **kwargs) -> go.Figure:
        """Render plots for metrics returned by the multivariate classifier for drift detection.

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
        >>> reference_df, analysis_df, _ = nml.load_synthetic_car_loan_dataset()
        >>> # Define feature columns
        >>> feature_column_names = [
        ...     col for col in reference_df.columns
        ...     if col not in non_feature_columns
        >>> ]
        >>> calc = nml.DomainClassifierCalculator(
        ...     feature_column_names=feature_column_names,
        ...     timestamp_column_name='timestamp',
        ...     chunk_size=5000
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> figure = results.plot()
        >>> figure.show()
        """
        if kind == 'drift':
            return plot_metric(
                self,
                title='Multivariate Drift - Domain Classifier',
                metric_display_name='Domain Classifier AUROC ',
                metric_column_name='domain_classifier_auroc',
                hover=Hover(
                    template='%{period} &nbsp; &nbsp; %{alert} <br />'
                    'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
                    '%{metric_name}: <b>%{metric_value}</b><b r />',
                    show_extra=True,
                ),
            )
        else:
            raise InvalidArgumentsException(f"unknown plot kind '{kind}'. " f"Please provide one of: ['drift'].")
