#  Author:   Wiljan Cools    <wiljan@nannyml.com>
#            Niels Nuyttens  <niels@nannyml.com>
#  License: Apache Software License 2.0

"""Module containing functionality to plot stacked bar charts."""
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

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
        if value_counts_table[feature_column_name].nunique() > max_number_of_categories + 1:
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
    start_date_column_name,
    end_date_column_name,
    chunk_type_column_name,
    chunk_column_name,
    drift_column_name,
    chunk_types,
    date_label_hover_format,
):
    stacked_bar_table = pd.merge(drift_table, value_counts_table, on=chunk_column_name)

    stacked_bar_table['hue'] = 0
    if chunk_types and chunk_type_column_name and chunk_column_name in drift_table.columns:
        for i, chunk_type in enumerate(chunk_types):
            stacked_bar_table.loc[stacked_bar_table[chunk_type_column_name] == chunk_types[i], 'hue'] = i
    if drift_column_name and drift_column_name in stacked_bar_table.columns:
        stacked_bar_table.loc[stacked_bar_table[drift_column_name], 'hue'] = -1

    stacked_bar_table = stacked_bar_table.sort_values(end_date_column_name, ascending=True).reset_index(drop=True)
    stacked_bar_table['next_end_date'] = stacked_bar_table[end_date_column_name].shift(-1)

    stacked_bar_table['start_date_label_hover'] = stacked_bar_table[start_date_column_name].dt.strftime(
        date_label_hover_format
    )
    stacked_bar_table['end_date_label_hover'] = stacked_bar_table[end_date_column_name].dt.strftime(
        date_label_hover_format
    )

    return stacked_bar_table


def _create_stacked_bar_plot(
    stacked_bar_table,
    feature_column_name,
    start_date_column_name,
    end_date_column_name,
    chunk_type_column_name,
    chunk_column_name,
    chunk_types,
    chunk_type_labels,
    hue_legend_labels,
    chunk_hover_label,
    figure,
    title,
    xaxis_title,
    yaxis_title,
    alpha,
    alpha_chunk_type,
    colors,
):

    categories = stacked_bar_table[feature_column_name].cat.categories
    category_colors = list(
        sns.blend_palette(
            [Colors.INDIGO_PERSIAN, Colors.GRAY, Colors.BLUE_SKY_CRAYOLA], n_colors=len(categories)
        ).as_hex()
    )
    category_colors_transparant = [
        'rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha)) for color in category_colors
    ]
    colors_transparant = [
        'rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), alpha_chunk_type))
        for color in colors
    ]

    hover_template = (
        chunk_hover_label
        + ' %{customdata[0]}: %{customdata[1]} - %{customdata[2]}; (%{customdata[3]}, %{customdata[4]})'
    )

    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis_title, linecolor=colors[2], showgrid=False, mirror=True, zeroline=False),
        yaxis=dict(
            title=yaxis_title, linecolor=colors[2], showgrid=False, mirror=True, autorange="reversed", zeroline=False
        ),
        paper_bgcolor='rgba(255,255,255,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        legend=dict(itemclick=False, itemdoubleclick=False),
        barmode='relative',
    )
    if figure:
        fig = figure
        fig.update_layout(layout)
    else:
        fig = go.Figure(layout=layout)

    # ____Plot elements___#
    # Plot bars
    for i, category in enumerate(categories):
        data = stacked_bar_table.loc[
            stacked_bar_table[feature_column_name] == category,
        ]

        hover_data = data[
            [
                chunk_column_name,
                'start_date_label_hover',
                'end_date_label_hover',
                'value_counts_normalised',
                'value_counts',
            ]
        ].values

        fig.add_trace(
            go.Bar(
                name=category,
                x=data['value_counts_normalised'],
                y=data[start_date_column_name],
                orientation='h',
                marker_line_color=data['hue'].apply(lambda hue: colors[hue] if hue == -1 else 'rgba(255,255,255,1)'),
                marker_color=category_colors_transparant[i],
                marker_line_width=data['hue'].apply(lambda hue: 2 if hue == -1 else 1),
                yperiodalignment="start",
                offset=0,
                showlegend=False,
                customdata=hover_data,
                hovertemplate=hover_template,
            )
        )

    # Shade chunk types
    for i, chunk_type in enumerate(chunk_types):
        subset = stacked_bar_table.loc[stacked_bar_table[chunk_type_column_name] == chunk_type]
        fig.add_shape(
            y0=subset[start_date_column_name].min(),
            y1=subset[end_date_column_name].max(),
            x0=0,
            x1=1.05,
            line_color=colors_transparant[i],
            layer='above',
            line_width=2,
            line=dict(dash='dash'),
        ),
        fig.add_annotation(
            x=1.025,
            y=subset[start_date_column_name].mean(),
            text=chunk_type_labels[i],
            font=dict(color=colors[i]),
            align="center",
            textangle=90,
            showarrow=False,
        )

    # ____Add elements to legend___#
    x = [np.nan] * len(data)
    y = data[start_date_column_name]

    # Add chunk types
    for i, hue_label in enumerate(chunk_types):
        fig.add_trace(
            go.Scatter(
                mode='lines',
                y=y,
                x=x,
                name=hue_legend_labels[i],
                line=dict(color=colors_transparant[i], dash='dash', width=2),
                hoverinfo='skip',
            )
        )

    # Add drift
    fig.add_trace(
        go.Bar(
            y=y,
            x=x,
            name=hue_legend_labels[-1],
            orientation='h',
            marker_line_color=colors[-1],
            marker_color='rgba(0,0,0,0)',
            marker_line_width=2,
            hoverinfo='skip',
        )
    )

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
    start_date_column_name='start_date',
    end_date_column_name='end_date',
    chunk_type_column_name='partition',
    chunk_column_name='chunk',
    drift_column_name='drift',
    chunk_types=None,
    chunk_type_labels=None,
    hue_legend_labels=None,
    chunk_hover_label='Chunk',
    date_label_hover_format='%d/%b/%y',
    figure=None,
    title='Feature: distribution over time',
    x_axis_title='Relative frequency',
    yaxis_title='Time',
    alpha=1,
    alpha_chunk_type=0.5,
    colors=None,
    category_colors=None,
    missing_category_label='Missing',
    max_number_of_categories=4,
):
    if chunk_types is None:
        chunk_types = ['reference', 'analysis']

    if chunk_type_labels is None:
        chunk_type_labels = ['Reference', 'Analysis']

    if hue_legend_labels is None:
        hue_legend_labels = ['Reference period', 'Analysis period', 'Period with data drift']

    if colors is None:
        colors = [Colors.BLUE_SKY_CRAYOLA, Colors.INDIGO_PERSIAN, Colors.GRAY_DARK, Colors.RED_IMPERIAL]

    if category_colors is None:
        category_colors = [
            'rgb(107, 0, 236)',
            'rgb(204, 163, 255)',
            'rgb(0, 67, 239)',
            'rgb(0, 200, 229)',
            'rgb(128, 228, 242)',
        ]

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
        start_date_column_name,
        end_date_column_name,
        chunk_type_column_name,
        chunk_column_name,
        drift_column_name,
        chunk_types,
        date_label_hover_format,
    )

    fig = _create_stacked_bar_plot(
        stacked_bar_table,
        feature_column_name,
        start_date_column_name,
        end_date_column_name,
        chunk_type_column_name,
        chunk_column_name,
        chunk_types,
        chunk_type_labels,
        hue_legend_labels,
        chunk_hover_label,
        figure,
        title,
        x_axis_title,
        yaxis_title,
        alpha,
        alpha_chunk_type,
        colors,
    )

    return fig
