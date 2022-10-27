#  Author:   Wiljan Cools    <wiljan@nannyml.com>
#            Niels Nuyttens  <niels@nannyml.com>
#  License: Apache Software License 2.0

from functools import partial

import matplotlib
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy.integrate import cumulative_trapezoid
from statsmodels import api as sm

from nannyml.plots.colors import Colors


def _get_kde(array, cut=3, clip=(-np.inf, np.inf)):
    try:  # pragma: no cover
        kde = sm.nonparametric.KDEUnivariate(array)
        kde.fit(cut=cut, clip=clip)
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


def _create_kde_table(
    feature_table,
    feature_column_name,
    chunk_column_name,
    chunk_type_column_name,
    kde_cut=3,
    kde_clip=(-np.inf, np.inf),
    post_kde_clip=None,
):
    get_kde_partial_application = partial(_get_kde, cut=kde_cut, clip=kde_clip)
    group_by_cols = [chunk_column_name]
    if chunk_type_column_name in feature_table.columns:
        group_by_cols += [chunk_type_column_name]
    data = (
        #  group by period too, 'key' column can be there for both reference and analysis
        feature_table.groupby(group_by_cols)[feature_column_name]
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
    data['kde_density_global_max'] = data.groupby(chunk_column_name)['kde_density_local_max'].max().max()
    data['kde_density_scaled'] = data[['kde_density', 'kde_density_global_max']].apply(
        lambda row: np.divide(np.array(row['kde_density']), row['kde_density_global_max']), axis=1
    )
    data['kde_quartiles_scaled'] = data[['kde_quartiles', 'kde_density_global_max']].apply(
        lambda row: [(q[0], q[1] / row['kde_density_global_max']) for q in row['kde_quartiles']], axis=1
    )

    return data


def _create_joy_table(
    drift_table,
    kde_table,
    feature_column_name,
    chunk_column_name,
    chunk_type_column_name,
    end_date_column_name,
    chunk_index_column_name,
    drift_column_name,
    chunk_types,
):

    joy_table = pd.merge(drift_table, kde_table)

    joy_table['hue'] = 0
    if chunk_types and chunk_type_column_name and chunk_column_name in drift_table.columns:
        for i, chunk_type in enumerate(chunk_types):
            joy_table.loc[joy_table[chunk_type_column_name] == chunk_types[i], 'hue'] = i
    if drift_column_name and drift_column_name in joy_table.columns:
        joy_table.loc[joy_table[drift_column_name], 'hue'] = -1

    # Sort to make sure most current chunks are plotted in front of the others
    if end_date_column_name:
        joy_table = joy_table.sort_values(end_date_column_name, ascending=True).reset_index(drop=True)
    else:
        joy_table = joy_table.sort_values(chunk_index_column_name, ascending=True).reset_index(drop=True)

    return joy_table


def _create_joy_plot(
    joy_table,
    chunk_column_name,
    start_date_column_name,
    end_date_column_name,
    chunk_index_column_name,
    chunk_type_column_name,
    drift_column_name,
    chunk_types,
    hue_legend_labels,
    chunk_hover_label,
    hue_joy_hover_labels,
    quartiles_legend_label,
    date_label_hover_format,
    joy_hover_format,
    joy_overlap,
    figure,
    title,
    x_axis_title,
    x_axis_lim,
    post_kde_clip,
    y_axis_title,
    alpha,
    colors,
    style,
):
    if chunk_types is None:
        chunk_types = ['reference', 'analysis']

    if hue_legend_labels is None:
        hue_legend_labels = ['Reference period', 'Analysis period', 'Period with probable data drift']

    if hue_joy_hover_labels is None:
        hue_joy_hover_labels = ['Reference', 'No drift', 'Probable drift']

    if colors is None:
        colors = [Colors.BLUE_SKY_CRAYOLA, Colors.INDIGO_PERSIAN, Colors.GRAY_DARK, Colors.RED_IMPERIAL]

    is_time_based_x_axis = start_date_column_name and end_date_column_name
    offset = (
        joy_table.loc[joy_table[chunk_type_column_name] == 'reference', chunk_index_column_name].max() + 1
        if len(joy_table.loc[joy_table[chunk_type_column_name] == 'reference']) > 0
        else 0
    )
    joy_table['chunk_index_unified'] = [
        idx + offset if period == 'analysis' else idx
        for idx, period in zip(joy_table[chunk_index_column_name], joy_table[chunk_type_column_name])
    ]

    colors_transparent = [
        'rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha)) for color in colors
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_axis_title if style == 'horizontal' else y_axis_title,
            linecolor=colors[2],
            range=post_kde_clip if style == 'horizontal' else None,
            showgrid=False,
            mirror=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=y_axis_title if style == 'horizontal' else x_axis_title,
            linecolor=colors[2],
            range=None if style == 'horizontal' else post_kde_clip,
            showgrid=False,
            mirror=True,
            autorange="reversed" if style == 'horizontal' else None,
            zeroline=False,
        ),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        legend=dict(itemclick=False, itemdoubleclick=False),
    )

    if figure:
        fig = figure
        fig.update_layout(layout)
    else:
        fig = go.Figure(layout=layout)

    for i, row in joy_table.iterrows():
        start_date_label_hover, end_date_label_hover = '', ''
        if is_time_based_x_axis:
            y_date_position = row[start_date_column_name]
            y_date_height_scaler = row[start_date_column_name] - row[end_date_column_name]
            start_date_label_hover = row[start_date_column_name].strftime(date_label_hover_format)
            end_date_label_hover = row[end_date_column_name].strftime(date_label_hover_format)
        else:
            y_date_position = row['chunk_index_unified']
            y_date_height_scaler = -1

        kde_support = row['kde_support']
        kde_density_scaled = row['kde_density_scaled'] * joy_overlap
        kde_quartiles = [(q[0], q[1] * joy_overlap) for q in row['kde_quartiles_scaled']]
        color = colors[int(row[chunk_type_column_name] == chunk_types[1])]
        color_drift = colors[row['hue']]
        color_fill = colors_transparent[row['hue']]
        trace_name = hue_joy_hover_labels[row['hue']]

        # ____Plot elements___#
        fig.add_trace(
            go.Scatter(
                name=trace_name,
                x=kde_support if style == 'horizontal' else y_date_position + kde_density_scaled * y_date_height_scaler,
                y=y_date_position + kde_density_scaled * y_date_height_scaler if style == 'horizontal' else kde_support,
                mode='lines',
                line=dict(color=color, width=1),
                hoverinfo='skip',
                showlegend=False,
                hoverlabel=dict(bgcolor=color_drift, font=dict(color='white')),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=kde_support if style == 'horizontal' else [y_date_position] * len(kde_density_scaled),
                y=[y_date_position] * len(kde_density_scaled) if style == 'horizontal' else kde_support,
                line=dict(color='rgba(0,0,0,0)', width=1),
                fill='tonexty',
                fillcolor=color_fill,
                hoverinfo='skip',
                showlegend=False,
            )
        )

        if quartiles_legend_label:
            for kde_quartile in kde_quartiles:

                if is_time_based_x_axis:
                    hover_content = (
                        row[chunk_column_name],
                        start_date_label_hover,
                        end_date_label_hover,
                        np.round(kde_quartile[0], 3),
                    )
                    hover_template = (
                        chunk_hover_label
                        + ' %{customdata[0]}: %{customdata[1]} - %{customdata[2]}, <b>%{customdata[3]}</b>'
                    )
                else:
                    hover_content = (
                        row[chunk_column_name],
                        row['chunk_index_unified'],
                        np.round(kde_quartile[0], 3),
                    )
                    hover_template = (
                        chunk_hover_label
                        + ' %{customdata[0]}: chunk index <b>%{customdata[1]}</b>, <b>%{customdata[2]}</b>'
                    )

                hover_data = np.asarray([hover_content, hover_content])

                fig.add_trace(
                    go.Scatter(
                        name=trace_name,
                        x=[kde_quartile[0], kde_quartile[0]]
                        if style == 'horizontal'
                        else [y_date_position, y_date_position + kde_quartile[1] * y_date_height_scaler],
                        y=[y_date_position, y_date_position + kde_quartile[1] * y_date_height_scaler]
                        if style == 'horizontal'
                        else [kde_quartile[0], kde_quartile[0]],
                        mode='lines',
                        line=dict(color=color_drift, width=1, dash='dot'),
                        hovertemplate=hover_template,
                        customdata=hover_data,
                        showlegend=False,
                    )
                )

    # ____Add elements to legend___#
    if is_time_based_x_axis:
        x = [np.nan] * len(joy_table) if style == 'horizontal' else joy_table[end_date_column_name]
        y = joy_table[end_date_column_name] if style == 'horizontal' else [np.nan] * len(joy_table)
    else:
        x = [np.nan] * len(joy_table) if style == 'horizontal' else joy_table['chunk_index_unified']
        y = joy_table['chunk_index_unified'] if style == 'horizontal' else [np.nan] * len(joy_table)

    # Add joy coloring
    for i, hue_label in enumerate(hue_legend_labels):
        if i == len(chunk_types):
            i = -1
        fig.add_traces(
            [
                go.Scatter(x=x, y=y, mode='lines', showlegend=False),
                go.Scatter(
                    name=hue_legend_labels[i],
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(color=colors[i] if i != -1 else 'rgba(0,0,0,0)', width=1),
                    fill='tonexty',
                    fillcolor=colors_transparent[i],
                ),
            ]
        )
    # Add kde quartiles
    if quartiles_legend_label:
        fig.add_traces(
            [
                go.Scatter(
                    name=quartiles_legend_label,
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(color=colors[-2], width=1, dash='dot'),
                )
            ]
        )

    fig.update_layout(legend=dict(traceorder='normal', yanchor="top", y=-0.25, xanchor="left", x=0, orientation="h"))

    return fig


def _joy_plot(
    drift_table,
    feature_table,
    feature_column_name,
    chunk_column_name='chunk',
    start_date_column_name: str = None,
    end_date_column_name: str = None,
    chunk_index_column_name='chunk_index',
    chunk_type_column_name='period',
    drift_column_name='drift',
    chunk_types=None,
    hue_legend_labels=None,
    chunk_hover_label='Chunk',
    hue_joy_hover_labels=None,
    quartiles_legend_label='Quartiles',
    date_label_hover_format='%d/%b/%y',
    joy_hover_format='{0:.2f}',
    joy_overlap=1,
    figure=None,
    title=None,
    x_axis_title='Feature',
    x_axis_lim=None,
    y_axis_title=None,
    alpha=0.2,
    colors=None,
    kde_cut=3,
    kde_clip=(-np.inf, np.inf),
    post_kde_clip=None,
    style='horizontal',
):
    """Renders a joy plot showing the evolution of feature distribution over time.

    For more info check: https://www.data-to-viz.com/graph/ridgeline.html
    """
    if chunk_types is None:
        chunk_types = ['reference', 'analysis']

    if hue_legend_labels is None:
        hue_legend_labels = ['Reference period', 'Analysis period', 'Period with data drift']

    if hue_joy_hover_labels is None:
        hue_joy_hover_labels = ['Reference', 'No data drift', 'Data drift']

    if colors is None:
        colors = [Colors.BLUE_SKY_CRAYOLA, Colors.INDIGO_PERSIAN, Colors.GRAY_DARK, Colors.RED_IMPERIAL]

    is_time_based_x_axis = start_date_column_name and end_date_column_name

    if not x_axis_title:
        x_axis_title = (
            'Feature distribution over time' if is_time_based_x_axis else 'Feature distribution across chunks'
        )

    if not y_axis_title:
        y_axis_title = 'Time' if is_time_based_x_axis else 'Chunk index'

    kde_table = _create_kde_table(
        feature_table, feature_column_name, chunk_column_name, chunk_type_column_name, kde_cut, kde_clip, post_kde_clip
    )

    joy_table = _create_joy_table(
        drift_table,
        kde_table,
        feature_column_name,
        chunk_column_name,
        chunk_type_column_name,
        end_date_column_name,
        chunk_index_column_name,
        drift_column_name,
        chunk_types,
    )

    fig = _create_joy_plot(
        joy_table,
        chunk_column_name,
        start_date_column_name,
        end_date_column_name,
        chunk_index_column_name,
        chunk_type_column_name,
        drift_column_name,
        chunk_types,
        hue_legend_labels,
        chunk_hover_label,
        hue_joy_hover_labels,
        quartiles_legend_label,
        date_label_hover_format,
        joy_hover_format,
        joy_overlap,
        figure,
        title,
        x_axis_title,
        x_axis_lim,
        post_kde_clip,
        y_axis_title,
        alpha,
        colors,
        style,
    )

    return fig
