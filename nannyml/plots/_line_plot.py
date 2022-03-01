#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

#  Author:   Wiljan Cools    <wiljan@nannyml.com>
#            Niels Nuyttens  <niels@nannyml.com>
#  License: Apache Software License 2.0

"""Module containing plotting logic."""
from enum import Enum
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _data_prep_line_plot(
    data: pd.DataFrame,
    metric_column_name,
    chunk_type_column_name: str,
    start_date_column_name: str,
    end_date_column_name: str,
    metric_format='{0:.4f}',
    hover_date_label_format='%d/%b',
):
    data = data.copy()
    data['chunk_type_first'] = data[chunk_type_column_name] != data[chunk_type_column_name].shift()
    data['chunk_type_last'] = data[chunk_type_column_name] != data[chunk_type_column_name].shift(-1)
    data['metric_previous'] = data[metric_column_name].shift()
    data['metric_label'] = data[metric_column_name].apply(lambda x: metric_format.format(x))
    data['start_date_label'] = data[start_date_column_name].dt.strftime(hover_date_label_format)
    data['end_date_label'] = data[end_date_column_name].dt.strftime(hover_date_label_format)
    return data


def _line_plot(
    table,
    metric_column_name,
    estimated_column_name=None,
    confidence_column_name=None,
    threshold_column_name=None,
    statistically_significant_column_name=None,
    drift_column_name=None,
    chunk_column_name='chunk',
    start_date_column_name='start_date',
    end_date_column_name='end_date',
    chunk_type_column_name='partition',
    chunk_types=None,
    confidence_label='Confidence band',
    threshold_label='Data drift threshold',
    statistically_significant_label='P-value is signficant',
    drift_label='Probable data drift',
    chunk_label='Chunk',
    chunk_labels=None,
    marker_labels=None,
    metric_format='{0:.4f}',
    hover_date_label_format='%d/%b/%y',
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

    if chunk_labels is None:
        chunk_labels = ['Reference period', 'Analysis period']

    if marker_labels is None:
        marker_labels = ['Reference', 'No drift', 'Drifted']

    if colors is None:
        colors = [Colors.BLUE_SKY_CRAYOLA, Colors.INDIGO_PERSIAN, Colors.GRAY_DARK, Colors.RED_IMPERIAL]

    data = _data_prep_line_plot(
        data=table,
        metric_column_name=metric_column_name,
        chunk_type_column_name=chunk_type_column_name,
        start_date_column_name=start_date_column_name,
        end_date_column_name=end_date_column_name,
        metric_format=metric_format,
        hover_date_label_format=hover_date_label_format,
    )

    colors_transparent = [
        'rgba({},{})'.format(str([int(i * 255) for i in matplotlib.colors.to_rgb(color)])[1:-1].replace(' ', ''), alpha)
        for color in colors
    ]

    hover_template = chunk_label + ' %{customdata[0]}: %{customdata[1]} - %{customdata[2]}, %{customdata[3]}'

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

    # Plot confidence bands, if the metric estimated
    _plot_confidence_bands(
        fig,
        data,
        chunk_type_column_name,
        chunk_types,
        colors_transparent,
        confidence_column_name,
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

    # Plot metric for reference period
    _plot_reference_period_metric(
        fig,
        data,
        colors,
        metric_column_name,
        chunk_labels,
        chunk_type_column_name,
        chunk_types,
        start_date_column_name,
        end_date_column_name,
    )

    # Plot metric for analysis period
    _plot_metric_for_analysis_period(
        fig,
        data,
        colors,
        metric_column_name,
        estimated_column_name,
        chunk_labels,
        chunk_type_column_name,
        chunk_types,
        end_date_column_name,
    )

    # Plot reference and analysis markers that did not drift
    _plot_non_drifting_markers(
        fig,
        data,
        colors,
        metric_column_name,
        drift_column_name,
        marker_labels,
        hover_template,
        chunk_column_name,
        chunk_type_column_name,
        chunk_types,
        end_date_column_name,
    )

    # Plot data drift areas
    _plot_drift_markers_and_areas(
        fig,
        data,
        colors,
        alpha,
        metric_column_name,
        drift_column_name,
        marker_labels,
        hover_template,
        chunk_column_name,
        start_date_column_name,
        end_date_column_name,
    )

    # ____Add elements to legend, order matters___#

    # TODO: review this
    # https://plotly.com/python/legend/#legends-with-graph-objects

    data_subset = pd.concat(
        [
            data.loc[(data[chunk_type_column_name] != chunk_types[1]) & data['chunk_type_last']],
            data.loc[(data[chunk_type_column_name] == chunk_types[1])],
        ]
    )

    # Add confidence bands
    if confidence_column_name is not None and confidence_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    x=data_subset[end_date_column_name],
                    y=[np.nan] * len(data_subset),
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                ),
                go.Scatter(
                    name=confidence_label,
                    x=data_subset[end_date_column_name],
                    y=[np.nan] * len(data),
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    fill='tonexty',
                    fillcolor=colors_transparent[1],
                    legendgroup=1,
                ),
            ]
        )

    # Add statistically significant
    if statistically_significant_column_name is not None and statistically_significant_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    name=statistically_significant_label,
                    x=data_subset[end_date_column_name],
                    y=[np.nan] * len(data_subset),
                    mode='lines',
                    line=dict(color=colors_transparent[1], width=9),
                    legendgroup=1,
                )
            ]
        )

    # Add threshold to legend
    if threshold_column_name is not None and threshold_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    name=threshold_label,
                    x=data_subset[end_date_column_name],
                    y=[np.nan] * len(data_subset),
                    mode='lines',
                    line=dict(color=colors[-1], width=1, dash='dash'),
                    legendgroup=1,
                )
            ]
        )

    # Add shaded drift area to legend
    if drift_column_name is not None and drift_column_name in data.columns:
        fig.add_traces(
            [
                go.Scatter(
                    x=data_subset[end_date_column_name],
                    y=[np.nan] * len(data_subset),
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                ),
                go.Scatter(
                    name=drift_label,
                    x=data_subset[end_date_column_name],
                    y=[np.nan] * len(data),
                    mode='lines+markers',
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


def _plot_drift_markers_and_areas(
    fig,
    data,
    colors,
    alpha,
    metric_column_name,
    drift_column_name,
    marker_labels,
    hover_template,
    chunk_column_name,
    start_date_column_name,
    end_date_column_name,
):
    if drift_column_name is not None and drift_column_name in data.columns:
        for i, row in data.loc[
            data[drift_column_name],
        ].iterrows():
            fig.add_vrect(
                x0=row[start_date_column_name],
                x1=row[end_date_column_name],
                fillcolor=colors[-1],
                opacity=alpha,
                layer='below',
                line_width=0,
            )
            # Plot markers that drifted
        data_subset = data.loc[data[drift_column_name]]
        fig.add_trace(
            go.Scatter(
                name=marker_labels[2],
                mode='markers',
                x=data_subset[end_date_column_name],
                y=data_subset[metric_column_name],
                marker=dict(color=colors[-1], size=6, symbol='diamond'),
                customdata=data_subset[
                    [chunk_column_name, 'start_date_label', 'end_date_label', 'metric_label']
                ].values,
                hovertemplate=hover_template,
                legendgroup=1,
                showlegend=False,
            )
        )


def _plot_non_drifting_markers(
    fig,
    data,
    colors,
    metric_column_name,
    drift_column_name,
    marker_labels,
    hover_template,
    chunk_column_name,
    chunk_type_column_name,
    chunk_types,
    end_date_column_name,
):
    for i, chunk_type in enumerate(chunk_types):
        if drift_column_name is not None and drift_column_name in data.columns:
            data_subset = data.loc[(data[chunk_type_column_name] == chunk_type) & ~data[drift_column_name]]
        else:
            data_subset = data.loc[(data[chunk_type_column_name] == chunk_type)]
        fig.add_trace(
            go.Scatter(
                name=marker_labels[i],
                mode='markers',
                x=data_subset[end_date_column_name],
                y=data_subset[metric_column_name],
                marker=dict(color=colors[i], size=6, symbol='square'),
                customdata=data_subset[
                    [chunk_column_name, 'start_date_label', 'end_date_label', 'metric_label']
                ].values,
                hovertemplate=hover_template,
                showlegend=False,
                legendgroup=i,
            )
        )


def _plot_metric_for_analysis_period(
    fig,
    data,
    colors,
    metric_column_name,
    estimated_column_name,
    chunk_labels,
    chunk_type_column_name,
    chunk_types,
    end_date_column_name,
):
    data_subset = pd.concat(
        [
            data.loc[(data[chunk_type_column_name] != chunk_types[1]) & data['chunk_type_last']],
            data.loc[(data[chunk_type_column_name] == chunk_types[1])],
        ]
    )
    fig.add_trace(
        go.Scatter(
            name=chunk_labels[1],
            mode='lines',
            x=data_subset[end_date_column_name],
            y=data_subset[metric_column_name],
            line=dict(
                color=colors[1],
                width=2,
                dash='dash' if estimated_column_name is not None and estimated_column_name in data.columns else None,
            ),
            hoverinfo='skip',
            legendgroup=1,
        )
    )


def _plot_reference_period_metric(
    fig,
    data,
    colors,
    metric_column_name,
    chunk_labels,
    chunk_type_column_name,
    chunk_types,
    start_date_column_name,
    end_date_column_name,
):
    x = pd.concat(
        [
            data.loc[
                (data[chunk_type_column_name] == chunk_types[0]) & data['chunk_type_first'], start_date_column_name
            ],
            data.loc[(data[chunk_type_column_name] == chunk_types[0]), end_date_column_name],
        ]
    )
    y = pd.concat(
        [
            data.loc[(data[chunk_type_column_name] == chunk_types[0]) & data['chunk_type_first'], metric_column_name],
            data.loc[(data[chunk_type_column_name] == chunk_types[0]), metric_column_name],
        ]
    )
    fig.add_trace(
        go.Scatter(
            name=chunk_labels[0],
            mode='lines',
            x=x,
            y=y,
            line=dict(color=colors[0], width=2),
            hoverinfo='skip',
            legendgroup=0,
        )
    )


def _plot_thresholds(
    fig: go.Figure, data: pd.DataFrame, threshold_column_name: str, threshold_value_format: str, colors: List[str]
):
    if threshold_column_name is not None and threshold_column_name in data.columns:
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
            data['chunk_type_first'] & (data[chunk_type_column_name] == chunk_types[1]),
        ]
        fig.add_vline(
            x=pd.to_datetime(data_subset['start_date'].values[0]),
            line=dict(color=colors[1], width=1, dash='dash'),
            layer='below',
        )


def _plot_confidence_bands(
    fig: go.Figure,
    data: pd.DataFrame,
    chunk_type_column_name: str,
    chunk_types: List[str],
    colors_transparent: List[str],
    confidence_column_name: str,
    end_date_column_name: str,
    estimated_column_name: str,
    metric_column_name: str,
):
    if (
        confidence_column_name is not None
        and estimated_column_name is not None
        and {confidence_column_name, estimated_column_name}.issubset(data.columns)
    ):
        data_subset = pd.concat(
            [
                data.loc[(data[chunk_type_column_name] != chunk_types[1]) & data['chunk_type_last']],
                data.loc[(data[chunk_type_column_name] == chunk_types[1])],
            ]
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=data_subset[end_date_column_name],
                    y=data_subset[metric_column_name] + data_subset[confidence_column_name].fillna(0),
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    hoverinfo='skip',
                    showlegend=False,
                ),
                go.Scatter(
                    x=data_subset[end_date_column_name],
                    y=data_subset[metric_column_name] - data_subset[confidence_column_name].fillna(0),
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    hoverinfo='skip',
                    fill='tonexty',
                    fillcolor=colors_transparent[1],
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
                    y=[row['metric_previous'], row[metric_column_name]],
                    line=dict(color=colors_transparent[1], width=9),
                    hoverinfo='skip',
                    showlegend=False,
                )
            )


class Colors(str, Enum):
    """Color presets for plotting."""

    INDIGO_PERSIAN = "#3b0280"
    BLUE_SKY_CRAYOLA = "#00c8e5"
    RED_IMPERIAL = "#DD4040"
    SAFFRON = "#E1BC29"
    GREEN_SEA = "#3BB273"
    GRAY_DARK = "#666666"
    GRAY = "#E4E4E4"
    LIGHT_GRAY = "#F5F5F5"
