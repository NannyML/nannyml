#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Wiljan Cools    <wiljan@nannyml.com>
#            Niels Nuyttens  <niels@nannyml.com>
#  License: Apache Software License 2.0

"""Module containing plotting logic."""
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .colors import Colors


def _data_prep_step_plot(
    data: pd.DataFrame,
    metric_column_name: str,
    start_date_column_name: str,
    end_date_column_name: str,
    partial_target_column_name: str,
    hover_metric_format='{0:.4f}',
    hover_date_label_format='%b-%d-%Y',
):
    data = data.copy()
    data['mid_point_date'] = (
        data[start_date_column_name] + (data[end_date_column_name] - data[start_date_column_name]) / 2
    )
    data['metric_label'] = data[metric_column_name].apply(lambda x: hover_metric_format.format(x))
    data['start_date_label'] = data[start_date_column_name].dt.strftime(hover_date_label_format)
    data['end_date_label'] = data[end_date_column_name].dt.strftime(hover_date_label_format)

    if partial_target_column_name and partial_target_column_name in data:
        data['incomplete_target_percentage'] = (data[partial_target_column_name] * 100).apply(
            lambda p: format(p, '.2f')
        )
    return data


def _add_artificial_end_point(data: pd.DataFrame, start_date_column_name: str, end_date_column_name: str):
    data_point_hack = data.tail(1).copy()
    data_point_hack[start_date_column_name] = data_point_hack[end_date_column_name]
    data_point_hack[end_date_column_name] = pd.NaT
    data_point_hack['mid_point_date'] = pd.NaT
    data_point_hack.index = data_point_hack.index + 1
    return pd.concat([data, data_point_hack], axis=0)


def _step_plot(
    table,
    metric_column_name,
    estimated_column_name=None,
    confidence_column_name=None,
    threshold_column_name=None,
    statistically_significant_column_name=None,
    drift_column_name=None,
    partial_target_column_name=None,
    chunk_column_name='chunk',
    start_date_column_name='start_date',
    end_date_column_name='end_date',
    chunk_type_column_name='partition',
    chunk_types=None,
    confidence_legend_label='Confidence band',
    threshold_legend_label='Data drift threshold',
    statistically_significant_legend_label='P-value is signficant',
    drift_legend_label='Data drift',
    chunk_legend_labels=None,
    partial_target_legend_label='Incomplete target data',
    hover_marker_labels=None,
    hover_labels=None,
    hover_date_label_format='%b-%d-%Y',
    hover_metric_format='{0:.4f}',
    threshold_value_format='{0:.2f}',
    v_line_separating_analysis_period=True,
    figure=None,
    title='Metric over time',
    x_axis_title='Time',
    y_axis_title='Metric',
    y_axis_lim=None,
    alpha=0.2,
    colors=None,
):
    if chunk_types is None:
        chunk_types = ['reference', 'analysis']

    if chunk_legend_labels is None:
        chunk_legend_labels = ['Reference period', 'Analysis period']

    if hover_marker_labels is None:
        hover_marker_labels = ['Reference', 'No data drift', 'Data drift']

    if hover_labels is None:
        hover_labels = ['Chunk', 'Metric', 'Target data']

    if colors is None:
        colors = [Colors.BLUE_SKY_CRAYOLA, Colors.INDIGO_PERSIAN, Colors.GRAY_DARK, Colors.RED_IMPERIAL]

    data = _data_prep_step_plot(
        table,
        metric_column_name,
        start_date_column_name,
        end_date_column_name,
        partial_target_column_name,
        hover_metric_format,
        hover_date_label_format,
    )

    colors_transparent = [
        'rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha)) for color in colors
    ]

    hover_template = (
        hover_labels[0]
        + ': <b>%{customdata[0]}</b>&nbsp;&nbsp;&nbsp;&nbsp;'
        + 'From <b>%{customdata[1]}</b> to <b>%{customdata[2]}</b>&nbsp;&nbsp;&nbsp;&nbsp;'
        + hover_labels[1]
        + ': <b>%{customdata[3]}</b>'
    )

    custom_data_columns = [chunk_column_name, 'start_date_label', 'end_date_label', 'metric_label']

    if partial_target_column_name and partial_target_column_name in data:
        hover_template += '&nbsp;&nbsp;&nbsp;&nbsp;' + hover_labels[2] + ': <b>%{customdata[4]}% incomplete</b>'
        custom_data_columns += ['incomplete_target_percentage']

    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_axis_title, linecolor=colors[2], showgrid=False, mirror=True, zeroline=False),
        yaxis=dict(
            title=y_axis_title, linecolor=colors[2], showgrid=False, range=y_axis_lim, mirror=True, zeroline=False
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

    # ____Plot elements, order matters___#

    # Plot thresholds
    _plot_thresholds(fig, data, threshold_column_name, threshold_value_format, colors)

    # Plot line separating reference and analysis period
    _plot_reference_analysis_separator(
        fig, data, colors, v_line_separating_analysis_period, chunk_type_column_name, chunk_types
    )

    # Plot confidence band, if the metric estimated
    _plot_confidence_band(
        fig,
        data,
        chunk_type_column_name,
        chunk_types,
        colors_transparent,
        confidence_column_name,
        start_date_column_name,
        end_date_column_name,
        estimated_column_name,
        metric_column_name,
    )

    # Plot statistically significant band
    _plot_statistical_significance_band(
        fig,
        data,
        colors_transparent,
        statistically_significant_column_name,
        metric_column_name,
        start_date_column_name,
        end_date_column_name,
    )

    # Plot metric for reference and analysis period
    _plot_metric(
        fig,
        data,
        colors,
        metric_column_name,
        estimated_column_name,
        chunk_legend_labels,
        chunk_type_column_name,
        chunk_types,
        start_date_column_name,
        end_date_column_name,
        partial_target_column_name,
    )

    # Plot metric if partial target in analysis period
    _plot_metric_partial_target(
        fig,
        data,
        colors,
        metric_column_name,
        chunk_type_column_name,
        chunk_types,
        start_date_column_name,
        end_date_column_name,
        partial_target_column_name,
        partial_target_legend_label,
    )

    # Plot reference and analysis markers that did not drift
    _plot_non_drifted_markers(
        fig,
        data,
        colors,
        metric_column_name,
        drift_column_name,
        hover_marker_labels,
        hover_template,
        custom_data_columns,
        chunk_column_name,
        chunk_type_column_name,
        chunk_types,
        end_date_column_name,
    )

    # Plot data drifted markers and areas
    _plot_drifted_markers_and_areas(
        fig,
        data,
        colors,
        alpha,
        metric_column_name,
        drift_column_name,
        hover_marker_labels,
        hover_template,
        custom_data_columns,
        chunk_column_name,
        start_date_column_name,
        end_date_column_name,
    )

    # ____Add elements to legend, order matters___#

    x = [data['mid_point_date'].head(1).values, data['mid_point_date'].tail(1).values]
    y = [np.nan, np.nan]

    # Add confidence band
    if confidence_column_name and confidence_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    mode='lines',
                    x=x,
                    y=y,
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                ),
                go.Scatter(
                    name=confidence_legend_label,
                    mode='lines',
                    x=x,
                    y=y,
                    line_color='rgba(0,0,0,0)',
                    fill='tonexty',
                    fillcolor=colors_transparent[1],
                    legendgroup=1,
                ),
            ]
        )

    # Add statistically significant
    if statistically_significant_column_name and statistically_significant_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    name=statistically_significant_legend_label,
                    mode='lines',
                    x=x,
                    y=y,
                    line=dict(color=colors_transparent[1], width=9),
                    legendgroup=1,
                )
            ]
        )

    # Add threshold to legend
    if threshold_column_name and threshold_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    name=threshold_legend_label,
                    mode='lines',
                    x=x,
                    y=y,
                    line=dict(color=colors[-1], width=1, dash='dash'),
                    legendgroup=1,
                )
            ]
        )

    # Add shaded drift area to legend
    if drift_column_name and drift_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    mode='lines',
                    x=x,
                    y=y,
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                ),
                go.Scatter(
                    name=drift_legend_label,
                    mode='lines+markers',
                    x=x,
                    y=y,
                    line_color='rgba(0,0,0,0)',
                    fill='tonexty',
                    fillcolor=colors_transparent[-1],
                    marker=dict(color=colors[-1], size=6, symbol='diamond'),
                    legendgroup=1,
                ),
            ]
        )

    fig.update_layout(legend=dict(traceorder='normal', yanchor="top", y=-0.25, xanchor="left", x=0, orientation="h"))

    return fig


def _plot_metric(
    fig,
    data,
    colors,
    metric_column_name,
    estimated_column_name,
    chunk_legend_labels,
    chunk_type_column_name,
    chunk_types,
    start_date_column_name,
    end_date_column_name,
    partial_target_column_name,
):
    if partial_target_column_name and partial_target_column_name in data.columns:
        subset = data.loc[data[partial_target_column_name] == 0]
    else:
        subset = data.copy()

    for i, chunk_type in enumerate(chunk_types):
        data_subset = subset.loc[subset[chunk_type_column_name] == chunk_type]
        data_subset = _add_artificial_end_point(data_subset, start_date_column_name, end_date_column_name)
        dash = None
        if estimated_column_name and estimated_column_name in data.columns:
            if not data_subset.empty and data_subset[estimated_column_name].head(1).values[0]:
                dash = 'dot'
        fig.add_trace(
            go.Scatter(
                name=chunk_legend_labels[i],
                mode='lines',
                x=data_subset[start_date_column_name],
                y=data_subset[metric_column_name],
                line=dict(shape='hv', color=colors[i], width=2, dash=dash),
                hoverinfo='skip',
                legendgroup=1,
            )
        )


def _plot_metric_partial_target(
    fig,
    data,
    colors,
    metric_column_name,
    chunk_type_column_name,
    chunk_types,
    start_date_column_name,
    end_date_column_name,
    partial_target_column_name,
    partial_target_legend_label,
):
    if partial_target_column_name and partial_target_column_name in data.columns:
        data_subset = data.loc[(data[chunk_type_column_name] == chunk_types[1])]
        data_subset = _add_artificial_end_point(data_subset, start_date_column_name, end_date_column_name)
        fig.add_trace(
            go.Scatter(
                name=partial_target_legend_label,
                mode='lines',
                x=data_subset[start_date_column_name],
                y=data_subset[metric_column_name],
                line=dict(shape='hv', color=colors[1], width=2, dash='dot'),
                hoverinfo='skip',
                legendgroup=1,
            )
        )


def _plot_non_drifted_markers(
    fig,
    data,
    colors,
    metric_column_name,
    drift_column_name,
    hover_marker_labels,
    hover_template,
    custom_data_columns,
    chunk_column_name,
    chunk_type_column_name,
    chunk_types,
    end_date_column_name,
):
    for i, chunk_type in enumerate(chunk_types):
        if drift_column_name and drift_column_name in data.columns:
            data_subset = data.loc[(data[chunk_type_column_name] == chunk_type) & ~data[drift_column_name]]
        else:
            data_subset = data.loc[(data[chunk_type_column_name] == chunk_type)]

        fig.add_trace(
            go.Scatter(
                name=hover_marker_labels[i],
                mode='markers',
                x=data_subset['mid_point_date'],
                y=data_subset[metric_column_name],
                marker=dict(color=colors[i], size=6, symbol='square'),
                customdata=data_subset[custom_data_columns].values,
                hovertemplate=hover_template,
                showlegend=False,
            )
        )


def _plot_drifted_markers_and_areas(
    fig,
    data,
    colors,
    alpha,
    metric_column_name,
    drift_column_name,
    hover_marker_labels,
    hover_template,
    custom_data_columns,
    chunk_column_name,
    start_date_column_name,
    end_date_column_name,
):
    if drift_column_name and drift_column_name in data.columns:
        for i, row in data.loc[data[drift_column_name], :].iterrows():
            fig.add_vrect(
                x0=row[start_date_column_name],
                x1=row[end_date_column_name],
                fillcolor=colors[-1],
                opacity=alpha,
                layer='below',
                line_width=0,
            )

        data_subset = data.loc[data[drift_column_name]]
        fig.add_trace(
            go.Scatter(
                name=hover_marker_labels[2],
                mode='markers',
                x=data_subset['mid_point_date'],
                y=data_subset[metric_column_name],
                marker=dict(color=colors[-1], size=6, symbol='diamond'),
                customdata=data_subset[custom_data_columns].values,
                hovertemplate=hover_template,
                showlegend=False,
            )
        )


def _plot_thresholds(
    fig: go.Figure, data: pd.DataFrame, threshold_column_name: str, threshold_value_format: str, colors: List[str]
):
    if threshold_column_name and threshold_column_name in data.columns:
        threshold_values = data[threshold_column_name].values[0]
        if not isinstance(threshold_values, tuple):
            threshold_values = [threshold_values]
        for threshold_value in threshold_values:
            fig.add_hline(
                threshold_value,
                annotation_text=threshold_value_format.format(threshold_value),
                annotation_position="top right",
                annotation=dict(font_color=colors[-1]),
                line=dict(color=colors[-1], width=1, dash='dash'),
                layer='below',
            )


def _plot_reference_analysis_separator(
    fig: go.Figure,
    data: pd.DataFrame,
    colors: List[str],
    v_line_separating_analysis_period: bool,
    chunk_type_column_name: str,
    chunk_types: List[str],
):
    if v_line_separating_analysis_period:
        data_subset = data.loc[
            data[chunk_type_column_name] == chunk_types[1],
        ].head(1)
        fig.add_vline(
            x=pd.to_datetime(data_subset['start_date'].values[0]),
            line=dict(color=colors[1], width=1, dash='dash'),
            layer='below',
        )


def _plot_confidence_band(
    fig: go.Figure,
    data: pd.DataFrame,
    chunk_type_column_name: str,
    chunk_types: List[str],
    colors_transparent: List[str],
    confidence_column_name: str,
    start_date_column_name: str,
    end_date_column_name: str,
    estimated_column_name: str,
    metric_column_name: str,
):
    if (
        confidence_column_name
        and estimated_column_name
        and {confidence_column_name, estimated_column_name}.issubset(data.columns)
    ):
        data_subset = data.loc[data[chunk_type_column_name] == chunk_types[1]]
        data_subset = _add_artificial_end_point(data_subset, start_date_column_name, end_date_column_name)
        fig.add_traces(
            [
                go.Scatter(
                    mode='lines',
                    x=data_subset[start_date_column_name],
                    y=data_subset[metric_column_name] + data_subset[confidence_column_name],
                    line=dict(shape='hv', color='rgba(0,0,0,0)'),
                    hoverinfo='skip',
                    showlegend=False,
                ),
                go.Scatter(
                    mode='lines',
                    x=data_subset[start_date_column_name],
                    y=data_subset[metric_column_name] - data_subset[confidence_column_name],
                    line=dict(shape='hv', color='rgba(0,0,0,0)'),
                    fill='tonexty',
                    fillcolor=colors_transparent[1],
                    hoverinfo='skip',
                    showlegend=False,
                ),
            ]
        )


def _plot_statistical_significance_band(
    fig,
    data,
    colors_transparent,
    statistically_significant_column_name,
    metric_column_name,
    start_date_column_name,
    end_date_column_name,
):
    if statistically_significant_column_name is not None and statistically_significant_column_name in data.columns:
        data_subset = data.loc[data[statistically_significant_column_name]]
        for i, row in data_subset.iterrows():
            fig.add_trace(
                go.Scatter(
                    mode='lines',
                    x=[row[start_date_column_name], row[end_date_column_name]],
                    y=[row[metric_column_name], row[metric_column_name]],
                    line=dict(color=colors_transparent[1], width=9),
                    hoverinfo='skip',
                    showlegend=False,
                )
            )
