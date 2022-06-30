#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing univariate statistical drift calculation results and associated plotting implementations."""

import pandas as pd
import plotly.graph_objects as go

from nannyml.base import AbstractCalculatorResult
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.feature_drift import feature_drift


class UnivariateStatisticalDriftCalculatorResult(AbstractCalculatorResult):
    """Contains the univariate statistical drift calculation results and provides additional plotting functionality."""

    def __init__(self, results_data: pd.DataFrame, calculator):
        super().__init__(results_data)

        from nannyml.drift.model_inputs.univariate.statistical.calculator import UnivariateStatisticalDriftCalculator

        if not isinstance(calculator, UnivariateStatisticalDriftCalculator):
            raise InvalidArgumentsException(
                f"{calculator.__class__.__name__} is not an instance of type " f"UnivariateStatisticalDriftCalculator"
            )
        self.calculator = calculator

    @property
    def calculator_name(self) -> str:
        return "univariate_statistical_feature_drift"

    def plot(
        self,
        kind: str = 'feature',
        metric: str = 'statistic',
        feature_column_name: str = None,
        *args,
        **kwargs,
    ) -> go.Figure:
        """Renders a line plot for a chosen metric of statistical drift calculation results.

        Given either a feature label (check ``model_metadata.features``) or the actual feature column name
        and a metric (one of either ``statistic`` or ``p_value``) this function will render a line plot displaying
        the metric value for the selected feature per chunk.
        Chunks are set on a time-based X-axis by using the period containing their observations.
        Chunks of different periods (``reference`` and ``analysis``) are represented using different colors and
        a vertical separation if the drift results contain multiple periods.

        The different plot kinds that are available:

        - ``feature_drift``: plots drift per :class:`~nannyml.chunk.Chunk` for a single feature of a chunked data set.
        - ``feature_distribution``: plots feature distribution per :class:`~nannyml.chunk.Chunk`.
          Joyplot for continuous features, stacked bar charts for categorical features.

        Parameters
        ----------
        kind: str, default=`feature_drift`
            The kind of plot you want to have. Value must be one of ``feature_drift``, ``feature_distribution``.
        metric : str, default=``statistic``
            The metric to plot. Value must be one of ``statistic`` or ``p_value``
        feature_column_name : str
            Column name identifying a feature according to the preset model metadata. The function will raise an
            exception when no feature using that column name was found in the metadata.
            Either ``feature_column_name`` or ``feature_label`` should be specified.


        Returns
        -------
        fig: plotly.graph_objects.Figure
            A ``Figure`` object containing the requested drift plot. Can be saved to disk or shown rendered on screen
            using ``fig.show()``.


        Examples
        --------
        >>> import nannyml as nml
        >>> ref_df, ana_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>> metadata = nml.extract_metadata(ref_df, model_type=nml.ModelType.CLASSIFICATION_BINARY)
        >>> drift_calc = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='W')
        >>> drift_calc.fit(ref_df)
        >>> drifts = drift_calc.calculate(ana_df)
        >>> # loop over all features and plot the feature drift and feature distribution for each
        >>> for f in metadata.features:
        >>>     drifts.plot(kind='feature_drift', feature_label=f.label).show()
        >>>     drifts.plot(kind='feature_distribution', feature_label=f.label).show()

        """
        if kind == 'feature_drift':
            return feature_drift(self, feature_column_name, metric)
        # elif kind == 'feature_distribution':
        #     feature = _get_feature(self.metadata, feature_label, feature_column_name)
        #     return _plot_feature_distribution(
        #         data=self._analysis_data,
        #         drift_data=self.data,
        #         feature=feature,
        #     )
        else:
            raise InvalidArgumentsException(
                f"unknown plot kind '{kind}'. " f"Please provide on of: ['feature_drift', 'feature_distribution']."
            )
