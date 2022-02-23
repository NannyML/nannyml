#  Author:   Wiljan Cools    <wiljan@nannyml.com>
#            Niels Nuyttens  <niels@nannyml.com>
#  License: Apache Software License 2.0

"""Module containing functionality to plot stacked bar charts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nannyml.plots.colors import Colors


def _create_value_counts_table(
    feature_table,
    feature_column_name,
    chunk_column_name,
    missing_category_label,
    max_number_of_categories,
):
    value_counts_table = feature_table[[chunk_column_name, feature_column_name]].copy()
    value_counts_table[feature_column_name] = value_counts_table[feature_column_name].fillna(missing_category_label)

    if max_number_of_categories:
        top_categories = (
            value_counts_table[feature_column_name].value_counts().index.tolist()[:max_number_of_categories]
        )
        value_counts_table.loc[
            ~value_counts_table[feature_column_name].isin(top_categories), feature_column_name
        ] = 'Other'

    categories_ordered = value_counts_table[feature_column_name].value_counts().index.tolist()
    value_counts_table[feature_column_name] = pd.Categorical(
        value_counts_table[feature_column_name], categories_ordered
    )
    value_counts_table = (
        value_counts_table.groupby(chunk_column_name)[feature_column_name]
        .value_counts()
        .to_frame('value_counts')
        .reset_index()
        .rename(columns={'level_1': feature_column_name})
    )
    value_counts_table['value_counts_total'] = value_counts_table[chunk_column_name].map(
        value_counts_table.groupby(chunk_column_name)['value_counts'].sum()
    )
    value_counts_table['value_counts_normalised'] = (
        value_counts_table['value_counts'] / value_counts_table['value_counts_total']
    )

    return value_counts_table


def _create_stacked_bar_table(
    drift_table,
    value_counts_table,
    chunk_column_name,
    end_date_column_name,
    chunk_type_column_name,
    drift_column_name,
    chunk_types,
):
    stacked_bar_table = pd.merge(drift_table, value_counts_table, on=chunk_column_name)

    stacked_bar_table['hue'] = 0
    if chunk_types and chunk_type_column_name and chunk_column_name in drift_table.columns:
        for i, chunk_type in enumerate(chunk_types):
            stacked_bar_table.loc[stacked_bar_table[chunk_type_column_name] == chunk_types[i], 'hue'] = i
    if drift_column_name and drift_column_name in stacked_bar_table.columns:
        stacked_bar_table.loc[stacked_bar_table[drift_column_name], 'hue'] = -1

    stacked_bar_table = stacked_bar_table.sort_values(end_date_column_name, ascending=True).reset_index(drop=True)

    return stacked_bar_table


def _create_stacked_bar_plot(
    stacked_bar_table,
    feature_column_name,
    end_date_column_name,
    chunk_types,
    hue_legend_labels,
    figure,
    title,
    xaxis_title,
    xaxis_lim,
    yaxis_title,
    alpha,
    colors,
    category_colors,
):
    category_colors_transparant = [
        i.replace(')', ', {})'.format(alpha)).replace('rgb', 'rgba') for i in category_colors
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis_title, linecolor=colors[2], showgrid=False, mirror=True, range=xaxis_lim),
        yaxis=dict(title=yaxis_title, linecolor=colors[2], showgrid=False, mirror=True, autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(itemclick=False, itemdoubleclick=False),
        barmode='relative',
    )
    if figure:
        fig = figure
        fig.update_layout(layout)
    else:
        fig = go.Figure(layout=layout)

    categories = stacked_bar_table[feature_column_name].cat.categories
    for i, category in enumerate(categories):
        data = stacked_bar_table.loc[
            stacked_bar_table[feature_column_name] == category,
        ]
        # ____Plot elements___#
        fig.add_trace(
            go.Bar(
                x=data['value_counts_normalised'],
                y=data[end_date_column_name],
                name=category,
                orientation='h',
                marker_line_color=data['hue'].apply(lambda hue: colors[hue]),
                marker_color=category_colors_transparant[i],
                marker_line_width=1,
                hoverinfo='skip',
                showlegend=False,
            )
        )

    # ____Add elements to legend___#
    x = [np.nan] * len(data)
    y = data[end_date_column_name]

    # Add line coloring
    for i, hue_label in enumerate(hue_legend_labels):
        if i == len(chunk_types):
            i = -1
        fig.add_traces([go.Scatter(name=hue_label, x=x, y=y, mode='lines', line=dict(color=colors[i], width=1))])

    # Add categories
    for i, category in enumerate(categories):
        fig.add_trace(
            go.Bar(
                y=y,
                x=x,
                name=category,
                orientation='h',
                marker_line_color=category_colors_transparant[i],
                marker_color=category_colors_transparant[i],
                marker_line_width=1,
                hoverinfo='skip',
            )
        )

    fig.update_layout(
        barmode='stack', legend=dict(traceorder='normal', yanchor="top", y=-0.25, xanchor="left", x=0, orientation="h")
    )

    return fig


def _stacked_bar_plot(
    feature_table,
    drift_table,
    feature_column_name,
    chunk_column_name='chunk',
    end_date_column_name='end_date',
    chunk_type_column_name='partition',
    drift_column_name='drift',
    chunk_types=None,
    hue_legend_labels=None,
    figure=None,
    title='Feature: distribution over time',
    x_axis_title='Relative frequency',
    x_axis_lim=(0, 1.001),
    yaxis_title='Time',
    alpha=0.2,
    colors=None,
    category_colors=None,
    missing_category_label='Missing',
    max_number_of_categories=4,
):
    if chunk_types is None:
        chunk_types = ['reference', 'analysis']

    if hue_legend_labels is None:
        hue_legend_labels = ['Reference period', 'Analysis period', 'Period with probable data drift']

    if colors is None:
        colors = [Colors.BLUE_SKY_CRAYOLA, Colors.INDIGO_PERSIAN, Colors.GRAY_DARK, Colors.RED_IMPERIAL]

    if category_colors is None:
        category_colors = ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)', 'rgb(231,41,138)', 'rgb(102,166,30)']

    value_counts_table = _create_value_counts_table(
        feature_table,
        feature_column_name,
        chunk_column_name,
        missing_category_label,
        max_number_of_categories,
    )

    stacked_bar_table = _create_stacked_bar_table(
        drift_table,
        value_counts_table,
        chunk_column_name,
        end_date_column_name,
        chunk_type_column_name,
        drift_column_name,
        chunk_types,
    )

    fig = _create_stacked_bar_plot(
        stacked_bar_table,
        feature_column_name,
        end_date_column_name,
        chunk_types,
        hue_legend_labels,
        figure,
        title,
        x_axis_title,
        x_axis_lim,
        yaxis_title,
        alpha,
        colors,
        category_colors,
    )

    return fig
