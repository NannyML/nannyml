#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Contains the results of the univariate drift calculator and provides plotting functionality."""
from __future__ import annotations

import warnings
from typing import List, Optional, cast

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pandas as pd

import plotly.graph_objects as go

from nannyml._typing import Key
from nannyml.base import PerMetricPerColumnResult
from nannyml.chunk import Chunker, DefaultChunker
from nannyml.drift.univariate.methods import FeatureType, Method, MethodFactory
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots import Colors
from nannyml.plots.blueprints.comparisons import ResultCompareMixin
from nannyml.plots.blueprints.distributions import plot_distributions
from nannyml.plots.blueprints.metrics import plot_metrics
from nannyml.plots.components import Hover
from nannyml.thresholds import StandardDeviationThreshold
from nannyml.usage_logging import UsageEvent, log_usage


class Result(PerMetricPerColumnResult[Method], ResultCompareMixin):
    """Class wrapping the results of the univariate drift calculator and providing plotting functionality."""

    def __init__(
        self,
        results_data: pd.DataFrame,
        column_names: List[str],
        categorical_column_names: List[str],
        continuous_column_names: List[str],
        categorical_method_names: List[str],
        continuous_method_names: List[str],
        timestamp_column_name: Optional[str],
        chunker: Chunker,
        analysis_data: pd.DataFrame = None,
        reference_data: pd.DataFrame = None,
    ):
        """Initialize resuts class.

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
        categorical_method_names: List[str]
            A list of method names that will be performed on categorical columns.
            Supported methods for categorical variables:

                - `jensen_shannon`
                - `chi2`
                - `hellinger`
                - `l_infinity`
        continuous_method_names: List[str]
            A list of method names that will be performed on continuous columns.
            Supported methods for continuous variables:

                - `jensen_shannon`
                - `kolmogorov_smirnov`
                - `hellinger`
                - `wasserstein`
        timestamp_column_name: Optional[str], default=None
            The name of the column containing the timestamp of the model prediction.
            If not given, plots will not use a time-based x-axis but will use the index of the chunks instead.
        chunker: Chunker
            The `Chunker` used to split the data sets into a lists of chunks.
        analysis_data: pd.DataFrame, default= None
            Portion of data that NannyML will use to calculate the observed drift.
        reference_data: pd.DataFrame, default = None
            Portion of data that NannyML will use to fit its drift methods.
        """
        categorical_methods = [
            MethodFactory.create(
                m, FeatureType.CATEGORICAL, chunker=DefaultChunker(), threshold=StandardDeviationThreshold()
            )
            for m in categorical_method_names
        ]
        continuous_methods = [
            MethodFactory.create(
                m, FeatureType.CONTINUOUS, chunker=DefaultChunker(), threshold=StandardDeviationThreshold()
            )
            for m in continuous_method_names
        ]
        methods = continuous_methods + categorical_methods
        # Passing on methods as metrics to base class, as they're essentially a more specialised form of metrics
        # satisfying the same contract
        super().__init__(results_data, methods, column_names)

        self.continuous_column_names = continuous_column_names
        self.categorical_column_names = categorical_column_names
        self.timestamp_column_name = timestamp_column_name
        self.categorical_method_names = categorical_method_names
        self.categorical_methods = categorical_methods
        self.continuous_method_names = continuous_method_names
        self.continuous_methods = continuous_methods
        self.chunker = chunker

        self.analysis_data = analysis_data
        self.reference_data = reference_data

    @property
    def methods(self) -> List[Method]:
        """Methods used during calculation."""
        return cast(List[Method], self.metrics)

    def _filter(
        self,
        period: str,
        metrics: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Result:
        # TODO: Use TypeVar with generic self instead of cast
        result = cast(Result, super()._filter(period, methods, column_names, *args, **kwargs))
        method_names = [m.column_name for m in result.methods]

        # The `column_names` and `methods` filters can impact each other. Handled conditionally here
        if any(c in result.column_names for c in result.categorical_column_names):
            result.categorical_method_names = [m for m in self.categorical_method_names if m in method_names]
            result.categorical_methods = [m for m in self.categorical_methods if m.column_name in method_names]
        else:
            result.categorical_method_names = []
            result.categorical_methods = []
        if any(c in result.column_names for c in result.continuous_column_names):
            result.continuous_method_names = [m for m in self.continuous_method_names if m in method_names]
            result.continuous_methods = [m for m in self.continuous_methods if m.column_name in method_names]
        else:
            result.continuous_method_names = []
            result.continuous_methods = []
        if result.categorical_methods:
            result.categorical_column_names = [c for c in self.categorical_column_names if c in result.column_names]
        else:
            result.categorical_column_names = []
        if result.continuous_methods:
            result.continuous_column_names = [c for c in self.continuous_column_names if c in result.column_names]
        else:
            result.continuous_column_names = []
        result.metrics = result.continuous_methods + result.categorical_methods
        result.column_names = result.continuous_column_names + result.categorical_column_names

        return result

    def _get_result_property(self, property_name: str) -> List[pd.Series]:
        continuous_values = [
            self.data[(column, method.column_name, property_name)]
            for column in sorted(self.continuous_column_names)
            for method in sorted(self.continuous_methods, key=lambda m: m.column_name)
        ]
        categorical_values = [
            self.data[(column, method.column_name, property_name)]
            for column in sorted(self.categorical_column_names)
            for method in sorted(self.categorical_methods, key=lambda m: m.column_name)
        ]
        return continuous_values + categorical_values

    def keys(self) -> List[Key]:
        """Creates a list of keys for continuos and categorial columns.

        Each Key is a `namedtuple('Key', 'properties display_names')`
        """
        continuous_keys = [
            Key(properties=(column, method.column_name), display_names=(column, method.display_name))
            for column in sorted(self.continuous_column_names)
            for method in sorted(self.continuous_methods, key=lambda m: m.column_name)
        ]
        categorical_keys = [
            Key(properties=(column, method.column_name), display_names=(column, method.display_name))
            for column in sorted(self.categorical_column_names)
            for method in sorted(self.categorical_methods, key=lambda m: m.column_name)
        ]
        return continuous_keys + categorical_keys

    @log_usage(UsageEvent.UNIVAR_DRIFT_PLOT, metadata_from_kwargs=['kind'])
    def plot(
        self,
        kind: str = 'drift',
        *args,
        **kwargs,
    ) -> go.Figure:
        """Renders plots for metrics returned by the univariate distance drift calculator.

        For any feature you can render the statistic value or p-values as a step plot, or create a distribution plot.

        Parameters
        ----------
        kind: str, default='drift'
            The kind of plot you want to have. Allowed values are:

            - 'drift'
                    plots drift per :class:`~nannyml.chunk.Chunk` for a single feature of a chunked data set.
            - 'distribution'
                    plots feature distribution per :class:`~nannyml.chunk.Chunk`.
                    Joyplot for continuous features, stacked bar charts for categorical features.

        Returns
        -------
        fig: :class:`plotly.graph_objs._figure.Figure`
            A :class:`~plotly.graph_objs._figure.Figure` object containing the requested drift plot.

            Can be saved to disk using the :meth:`~plotly.graph_objs._figure.Figure.write_image` method
            or shown rendered on screen using the :meth:`~plotly.graph_objs._figure.Figure.show` method.

        Examples
        --------
        >>> import nannyml as nml
        >>> reference, analysis, _ = nml.load_synthetic_car_price_dataset()
        >>> column_names = [col for col in reference.columns if col not in ['timestamp', 'y_pred', 'y_true']]
        >>> calc = nml.UnivariateDriftCalculator(
        ...   column_names=column_names,
        ...   timestamp_column_name='timestamp',
        ...   continuous_methods=['kolmogorov_smirnov', 'jensen_shannon', 'wasserstein'],
        ...   categorical_methods=['chi2', 'jensen_shannon', 'l_infinity'],
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> res = res.filter(period='analysis')
        >>> for column_name in res.continuous_column_names:
        ...  for method in res.continuous_method_names:
        ...    res.plot(kind='drift', column_name=column_name, method=method).show()

        """
        if kind == 'drift':
            return plot_metrics(
                self,
                title='Univariate drift metrics',
                hover=Hover(
                    template='%{period} &nbsp; &nbsp; %{alert} <br />'
                    'Chunk: <b>%{chunk_key}</b> &nbsp; &nbsp; %{x_coordinate} <br />'
                    '%{metric_name}: <b>%{metric_value}</b><b r />',
                    show_extra=True,
                ),
                subplot_title_format='{display_names[1]} for <b>{display_names[0]}</b>',
                subplot_y_axis_title_format='{display_names[1]}',
                color=Colors.BLUE_SKY_CRAYOLA,
                metric_name='Method',
            )
        elif kind == 'distribution':
            return plot_distributions(
                self,
                reference_data=self.reference_data,
                analysis_data=self.analysis_data,
                chunker=self.chunker,
            )
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. " f"Please provide on of: ['drift', 'distribution']."
            )
