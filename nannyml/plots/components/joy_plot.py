#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from functools import partial
from typing import Any, Dict, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
from statsmodels import api as sm

from nannyml.chunk import Chunker
from nannyml.plots.colors import Colors
from nannyml.plots.components.hover import Hover, render_x_coordinate
from nannyml.plots.util import ensure_numpy, is_time_based_x_axis


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


def _get_kde_support(kde):
    if kde is not None:  # pragma: no cover
        return kde.support[::5]
    else:
        return np.array([])


def _get_kde_density(kde):
    if kde is not None:  # pragma: no cover
        return kde.density[::5]
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
            quartile_index = np.abs(cdf - quartile).argmin()
            quartiles.append((kde_support[quartile_index], kde_density[quartile_index]))
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
):
    if isinstance(data, np.ndarray):
        data = pd.Series(data, name='data')

    if isinstance(data_periods, np.ndarray):
        data_periods = pd.Series(data_periods, name='period')

    get_kde_partial_application = partial(_get_kde, cut=kde_cut, clip=kde_clip)

    data_with_chunk_keys = pd.concat(
        [
            chunk.data.assign(chunk_key=chunk.key, chunk_index=chunk.chunk_index)
            for chunk in chunker.split(pd.concat([data, timestamps], axis=1))
        ]
    )

    group_by_cols = ['chunk_index', 'chunk_key']
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

    data['kde_support'] = data['kde'].apply(lambda kde: _get_kde_support(kde))
    data['kde_density'] = data['kde'].apply(lambda kde: _get_kde_density(kde))
    data['kde_cdf'] = data[['kde_support', 'kde_density']].apply(
        lambda row: _get_kde_cdf(row['kde_support'], row['kde_density'] if len(row['kde_support']) > 0 else []), axis=1
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
        lambda row: [(q[0], q[1] / row['kde_density_local_max']) for q in row['kde_quartiles']], axis=1
    )

    return data


def _create_joy_table(data_distributions: pd.DataFrame, result_data: pd.DataFrame):
    joy_table = pd.merge(result_data, data_distributions)

    is_time_based_x_axis = not result_data['chunk_end_date'].isnull().values.any()
    # Sort to make sure most current chunks are plotted in front of the others
    if is_time_based_x_axis:
        joy_table = joy_table.sort_values('chunk_end_date', ascending=True).reset_index(drop=True)
    else:
        joy_table = joy_table.sort_values('chunk_index', ascending=True).reset_index(drop=True)

    return joy_table


def joy(
    fig: go.Figure,
    data_distributions: pd.DataFrame,
    color: str,
    name: str,
    chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    subplot_args: Optional[Dict[str, Any]] = None,
    alpha=0.2,
    plot_quartiles: bool = True,
    **kwargs,
) -> go.Figure:
    chunk_keys, chunk_indices, chunk_start_dates, chunk_end_dates = ensure_numpy(
        chunk_keys, chunk_indices, chunk_start_dates, chunk_end_dates
    )
    joy_overlap = 1

    if subplot_args is None:
        subplot_args = {}
    else:
        fig.update_xaxes(
            linecolor=Colors.INDIGO_PERSIAN,
            showgrid=False,
            mirror=True,
            zeroline=False,
            matches='x',
            title=fig.layout.xaxis.title,
            row=subplot_args['row'],
            col=subplot_args['col'],
        )
        fig.update_yaxes(
            linecolor=Colors.INDIGO_PERSIAN,
            showgrid=False,
            range=fig.layout.yaxis.range,
            title=fig.layout.yaxis.title,
            mirror=True,
            zeroline=False,
            row=subplot_args['row'],
            col=subplot_args['col'],
        )

    for i, row in data_distributions.iterrows():
        if is_time_based_x_axis(chunk_start_dates, chunk_end_dates):
            y_date_position = chunk_start_dates[i]
            y_date_height_scaler = chunk_start_dates[i] - chunk_end_dates[i]
        else:
            y_date_position = chunk_indices[i]
            y_date_height_scaler = -1

        kde_support = row['kde_support']
        kde_density_scaled = row['kde_density_scaled'] * joy_overlap
        kde_quartiles = [(q[0], q[1] * joy_overlap) for q in row['kde_quartiles_scaled']]

        fig.add_trace(
            go.Scatter(
                name=name,
                x=y_date_position + kde_density_scaled * y_date_height_scaler,
                y=kde_support,
                mode='lines',
                line=dict(color=color, width=1),
                hoverinfo='skip',
                showlegend=False,
                **kwargs,
            ),
            **subplot_args,
        )

        fig.add_trace(
            go.Scatter(
                x=[y_date_position] * len(kde_density_scaled),
                y=kde_support,
                line=dict(color='rgba(0,0,0,0)', width=1),
                fill='tonexty',
                fillcolor='rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha)),
                hoverinfo='skip',
                showlegend=False,
                **kwargs,
            ),
            **subplot_args,
        )

        if plot_quartiles:
            for kde_quartile in kde_quartiles:
                hover = Hover(template='Chunk %{chunk_key}: %{x_coordinate}, <b>%{quartile}</b>')
                hover.add(chunk_keys[i] if chunk_keys is not None else row['chunk_key'], name='chunk_key')
                hover.add(
                    render_x_coordinate(chunk_indices, chunk_start_dates, chunk_end_dates)[i], name='x_coordinate'
                )
                hover.add(np.round(kde_quartile[0], 3), name='quartile')

                fig.add_trace(
                    go.Scatter(
                        name=name,
                        x=[y_date_position, y_date_position + kde_quartile[1] * y_date_height_scaler],
                        y=[kde_quartile[0], kde_quartile[0]],
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        hovertemplate=hover.get_template(),
                        customdata=hover.get_custom_data(),
                        hoverlabel=dict(bgcolor=color, font=dict(color='white')),
                        showlegend=False,
                        **kwargs,
                    ),
                    **subplot_args,
                )

    return fig


def alert(
    fig: go.Figure,
    data_distributions: pd.DataFrame,
    color: str,
    name: str,
    alerts: Union[np.ndarray, pd.Series],
    chunk_keys: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    subplot_args: Optional[Dict[str, Any]] = None,
    alpha=0.3,
    plot_quartiles: bool = True,
    **kwargs,
) -> go.Figure:
    data = pd.DataFrame(
        {
            'chunk_keys': chunk_keys,
            'chunk_indices': chunk_indices,
            'chunk_start_dates': chunk_start_dates,
            'chunk_end_dates': chunk_end_dates,
            'alerts': alerts,
        }
    )
    data = pd.concat([data, data_distributions], axis=1)
    alerts_data = data.loc[data['alerts']].reset_index(drop=True)
    return joy(
        fig,
        alerts_data[data_distributions.columns],
        color,
        name,
        alerts_data['chunk_keys'],
        alerts_data['chunk_start_dates'],
        alerts_data['chunk_end_dates'],
        alerts_data['chunk_indices'],
        subplot_args,
        alpha,
        plot_quartiles,
        **kwargs,
    )
