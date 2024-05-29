from functools import partial
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from statsmodels import api as sm

from nannyml import Chunker
from nannyml._typing import Self
from nannyml.base import AbstractCalculator, _list_missing
from nannyml.distribution.continuous.result import Result
from nannyml.exceptions import InvalidArgumentsException


class ContinuousDistributionCalculator(AbstractCalculator):
    def __init__(
        self,
        column_names: Union[str, List[str]],
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        points_per_joy_plot: Optional[int] = None,
    ):
        super().__init__(
            chunk_size,
            chunk_number,
            chunk_period,
            chunker,
            timestamp_column_name,
        )

        self.column_names = column_names if isinstance(column_names, List) else [column_names]
        self.result: Optional[Result] = None
        self.points_per_joy_plot = points_per_joy_plot

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        self.result = self._calculate(reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, data)

        result_data = pd.DataFrame(columns=_create_multilevel_index(self.column_names))

        for column in self.column_names:
            column_distributions_per_chunk = calculate_chunk_distributions(
                data[column],
                self.chunker,
                data.get(self.timestamp_column_name, default=None),
                points_per_joy_plot=self.points_per_joy_plot,
            )
            column_distributions_per_chunk.drop(columns=['key', 'chunk_index'], inplace=True)
            for c in column_distributions_per_chunk.columns:
                result_data.loc[:, (column, c)] = column_distributions_per_chunk[c]

        chunks = self.chunker.split(data)
        result_data[('chunk', 'key')] = [c.key for c in chunks]
        result_data[('chunk', 'chunk_index')] = [c.chunk_index for c in chunks]
        result_data[('chunk', 'start_index')] = [c.start_index for c in chunks]
        result_data[('chunk', 'end_index')] = [c.end_index for c in chunks]
        result_data[('chunk', 'start_date')] = [c.start_datetime for c in chunks]
        result_data[('chunk', 'end_date')] = [c.end_datetime for c in chunks]
        result_data[('chunk', 'period')] = ['analysis' for _ in chunks]

        if self.result is None:
            self.result = Result(result_data, self.column_names, self.timestamp_column_name, self.chunker)
        else:
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, result_data], ignore_index=True)

        return self.result


def _get_kde(array, cut=3, clip=(-np.inf, np.inf)):
    try:  # pragma: no cover
        kde = sm.nonparametric.KDEUnivariate(array)
        kde.fit(cut=cut, clip=clip)

        # Calculation may return duplicate support values in edge cases. These results are not sensible. Treating it as
        # an error case and returning None
        if len(np.unique(kde.support)) < len(kde.support):
            return None

        return kde
    except Exception:
        return None


def _get_kde_support(kde, points_per_joy_plot: Optional[int] = None):
    if kde is not None:  # pragma: no cover
        return kde.support[:: (len(kde.support) // (points_per_joy_plot or 50))]
    else:
        return np.array([])


def _get_kde_density(kde, points_per_joy_plot: Optional[int] = None):
    if kde is not None:  # pragma: no cover
        return kde.density[:: (len(kde.support) // (points_per_joy_plot or 50))]
    else:
        return np.array([])


def _get_kde_cdf(kde_support, kde_density):
    if len(kde_support) > 0 and len(kde_density) > 0:
        cdf = cumulative_trapezoid(y=kde_density, x=kde_support, initial=0)
        return cdf
    else:
        return np.array([])


def _get_kde_quartiles(cdf, kde_support, kde_density):
    if len(cdf) > 0:
        quartiles = []
        for quartile in [0.25, 0.50, 0.75]:
            quartile_index = np.argmax(cdf >= quartile)
            quartiles.append((kde_support[quartile_index], kde_density[quartile_index], cdf[quartile_index]))
        return quartiles
    else:
        return []


def calculate_chunk_distributions(
    data: Union[np.ndarray, pd.Series],
    chunker: Chunker,
    timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
    data_periods: Optional[Union[np.ndarray, pd.Series]] = None,
    kde_cut=3,
    kde_clip=(-np.inf, np.inf),
    post_kde_clip=None,
    points_per_joy_plot: Optional[int] = None,
):
    if isinstance(data, np.ndarray):
        data = pd.Series(data, name='data')

    if isinstance(data_periods, np.ndarray):
        data_periods = pd.Series(data_periods, name='period')

    get_kde_partial_application = partial(_get_kde, cut=kde_cut, clip=kde_clip)

    data_with_chunk_keys = pd.concat(
        [
            chunk.data.assign(key=chunk.key, chunk_index=chunk.chunk_index)
            for chunk in chunker.split(pd.concat([data, timestamps], axis=1))
        ]
    )

    group_by_cols = ['chunk_index', 'key']
    if data_periods is not None:
        data_with_chunk_keys['period'] = data_periods
        group_by_cols += ['period']
    data = (
        #  group by period too, 'key' column can be there for both reference and analysis
        data_with_chunk_keys.groupby(group_by_cols)[data.name]
        .apply(get_kde_partial_application)
        .to_frame('kde')
        .reset_index()
    )

    data['kde_support'] = data['kde'].apply(lambda kde: _get_kde_support(kde, points_per_joy_plot))
    data['kde_density'] = data['kde'].apply(lambda kde: _get_kde_density(kde, points_per_joy_plot))
    data['kde_cdf'] = data[['kde_support', 'kde_density']].apply(
        lambda row: _get_kde_cdf(row['kde_support'], row['kde_density'] if len(row['kde_support']) > 0 else []),
        axis=1,
    )

    if post_kde_clip:
        # Clip the kde support to the clip values, adjust the density and cdf to the same length
        data['kde_support'] = data['kde_support'].apply(lambda x: x[x > post_kde_clip[0]])
        data['kde_support_len'] = data['kde_support'].apply(lambda x: len(x))
        data['kde_density'] = data.apply(lambda row: row['kde_density'][-row['kde_support_len'] :], axis=1)
        data['kde_cdf'] = data.apply(lambda row: row['kde_cdf'][-row['kde_support_len'] :], axis=1)
        data['kde_support'] = data['kde_support'].apply(lambda x: x[x < post_kde_clip[1]])
        data['kde_support_len'] = data['kde_support'].apply(lambda x: len(x))
        data['kde_density'] = data.apply(lambda row: row['kde_density'][: row['kde_support_len']], axis=1)
        data['kde_cdf'] = data.apply(lambda row: row['kde_cdf'][: row['kde_support_len']], axis=1)
        data['kde_support_len'] = data['kde_support'].apply(lambda x: len(x))

    data['kde_support_len'] = data['kde_support'].apply(lambda x: len(x))
    data['kde_quartiles'] = data[['kde_cdf', 'kde_support', 'kde_density']].apply(
        lambda row: _get_kde_quartiles(
            row['kde_cdf'], row['kde_support'], row['kde_density'] if len(row['kde_support']) > 0 else []
        ),
        axis=1,
    )
    data['kde_density_local_max'] = data['kde_density'].apply(lambda x: max(x) if len(x) > 0 else 0)
    data['kde_density_global_max'] = data.groupby('chunk_index')['kde_density_local_max'].max().max()
    data['kde_density_scaled'] = data[['kde_density', 'kde_density_local_max']].apply(
        lambda row: np.divide(np.array(row['kde_density']), row['kde_density_local_max']), axis=1
    )
    data['kde_quartiles_scaled'] = data[['kde_quartiles', 'kde_density_local_max']].apply(
        lambda row: [(q[0], q[1] / row['kde_density_local_max'], q[2]) for q in row['kde_quartiles']], axis=1
    )

    # TODO: Consider removing redundant columns to reduce fitted calculator memory usage
    # The kde calculator creates issues for pickling the calculator. We don't need it anymore, so removing it here
    del data['kde']

    return data


def _create_multilevel_index(column_names: List[str]):
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    distribution_column_names = [
        'kde',
        'kde_support',
        'kde_density',
        'kde_cdf',
        'kde_support_len',
        'kde_quartiles',
        'kde_density_local_max',
        'kde_density_global_max',
        'kde_density_scaled',
        'kde_quartiles_scaled',
    ]
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    continuous_column_tuples = [
        (column_name, distribution_column_name)
        for column_name in column_names
        for distribution_column_name in distribution_column_names
    ]

    tuples = chunk_tuples + continuous_column_tuples

    return pd.MultiIndex.from_tuples(tuples)
