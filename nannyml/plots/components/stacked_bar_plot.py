#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, Optional, Union, cast

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.graph_objects import Bar, Figure

from nannyml.chunk import Chunker
from nannyml.exceptions import InvalidArgumentsException
from nannyml.plots.colors import Colors
from nannyml.plots.components.hover import Hover, render_x_coordinate
from nannyml.plots.util import ensure_numpy, is_time_based_x_axis


def calculate_value_counts(
    data: Union[np.ndarray, pd.Series],
    chunker: Chunker,
    missing_category_label,
    max_number_of_categories,
    timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
    column_name: Optional[str] = None,
):
    if isinstance(data, np.ndarray):
        if column_name is None:
            raise InvalidArgumentsException("'column_name' can not be None when 'data' is of type 'np.ndarray'.")
        data = pd.Series(data, name=column_name)
    else:
        column_name = data.name

    data = data.astype("category")
    cat_str = [str(value) for value in data.cat.categories.values]
    data = data.cat.rename_categories(cat_str)
    data = data.cat.add_categories([missing_category_label, 'Other'])
    data = data.fillna(missing_category_label)

    if max_number_of_categories:
        top_categories = data.value_counts().index.tolist()[:max_number_of_categories]
        if data.nunique() > max_number_of_categories + 1:
            data.loc[~data.isin(top_categories)] = 'Other'

    data = data.cat.remove_unused_categories()

    categories_ordered = data.value_counts().index.tolist()
    categorical_data = pd.Categorical(data, categories_ordered)

    # TODO: deal with None timestamps
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.reset_index()

    chunks = chunker.split(pd.concat([pd.Series(categorical_data, name=column_name), timestamps], axis=1))
    data_with_chunk_keys = pd.concat([chunk.data.assign(chunk_index=chunk.chunk_index) for chunk in chunks])

    chunk_keys_lookup = {chunk.chunk_index: chunk.key for chunk in chunks}

    value_counts_table = (
        data_with_chunk_keys.groupby(['chunk_index'])[column_name]
        .value_counts()
        .to_frame('value_counts')
        .sort_values(by=['chunk_index', 'value_counts'])
        .reset_index()
        .rename(columns={'chunk_index': 'chunk_indices'})
    )

    value_counts_table['chunk_key'] = value_counts_table['chunk_indices'].map(lambda i: chunk_keys_lookup[i])

    value_counts_table['value_counts_total'] = value_counts_table['chunk_key'].map(
        value_counts_table.groupby('chunk_key')['value_counts'].sum()
    )
    value_counts_table['value_counts_normalised'] = (
        value_counts_table['value_counts'] / value_counts_table['value_counts_total']
    )

    return value_counts_table


def stacked_bar(
    figure: Figure,
    stacked_bar_table: pd.DataFrame,
    color: str,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    subplot_args: Optional[Dict[str, Any]] = None,
    annotation: Optional[str] = None,
    **kwargs,
) -> Figure:
    if subplot_args is None:
        subplot_args = {}

    chunk_indices, chunk_start_dates, chunk_end_dates = ensure_numpy(chunk_indices, chunk_start_dates, chunk_end_dates)
    column_name = [
        col for col in stacked_bar_table.columns if col not in ('chunk_key', 'chunk_indices', 'value_counts')
    ][0]
    categories = stacked_bar_table[column_name].cat.categories
    category_colors = list(
        sns.blend_palette(
            [Colors.INDIGO_PERSIAN, Colors.GRAY, Colors.BLUE_SKY_CRAYOLA], n_colors=len(categories)
        ).as_hex()
    )
    category_colors_transparent = [
        'rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), 1)) for color in category_colors
    ]

    figure.update_layout(
        dict(barmode='relative', paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)')
    )

    # plot bars
    for i, category in enumerate(categories):
        data = stacked_bar_table.loc[stacked_bar_table[column_name] == category]

        if is_time_based_x_axis(chunk_start_dates, chunk_end_dates):
            x = data.get('start_datetime', chunk_start_dates)
        else:
            x = data.get('chunk_indices', chunk_indices)

        hover = Hover(template="Chunk %{chunk_key}: %{x_coordinate}; (%{value_counts_normalised}, %{value_counts})")
        hover.add(data['chunk_key'], name='chunk_key')
        hover.add(
            render_x_coordinate(
                data['chunk_indices'],
                data.get('start_datetime', chunk_start_dates),
                data.get('end_datetime', chunk_end_dates),
            ),
            name='x_coordinate',
        )
        # hover.add(render_x_coordinate(data['chunk_indices'], chunk_start_dates, chunk_end_dates), name='x_coordinate')
        hover.add(data['value_counts_normalised'], name='value_counts_normalised')
        hover.add(data['value_counts'], name='value_counts')

        figure.add_trace(
            Bar(
                name=category,
                x=x,
                y=data['value_counts_normalised'],
                orientation='v',
                marker=dict(line_color=color, color=category_colors_transparent[i], line_width=0),
                yperiodalignment="start",
                offset=0,
                customdata=hover.get_custom_data(),
                hovertemplate=hover.get_template(),
                hoverlabel=dict(bgcolor=category_colors_transparent[i], font=dict(color='white')),
                **kwargs,
            ),
            **subplot_args,
        )

    # Shade chunk type
    x0 = chunk_start_dates.min() if is_time_based_x_axis(chunk_start_dates, chunk_end_dates) else chunk_indices.min()
    x1 = chunk_end_dates.max() if is_time_based_x_axis(chunk_start_dates, chunk_end_dates) else chunk_indices.max() + 1
    figure.add_shape(
        y0=0,
        y1=1.05,
        x0=x0,
        x1=x1,
        line_color='rgba{}'.format(matplotlib.colors.to_rgba(matplotlib.colors.to_rgb(color), 0.5)),
        layer='above',
        line_width=2,
        line=dict(dash='dash'),
        **subplot_args,
    ),
    if annotation:
        figure.add_annotation(
            x=pd.Series(chunk_start_dates).mean()
            if is_time_based_x_axis(chunk_start_dates, chunk_end_dates)
            else chunk_indices.mean(),
            y=1.025,
            text=annotation,
            font=dict(color=color),
            align="center",
            textangle=0,
            showarrow=False,
            **subplot_args,
        )

    return figure


def alert(
    figure: Figure,
    stacked_bar_table: pd.DataFrame,
    alerts: Union[np.ndarray, pd.Series],
    color: str,
    chunk_start_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_end_dates: Optional[Union[np.ndarray, pd.Series]] = None,
    chunk_indices: Optional[Union[np.ndarray, pd.Series]] = None,
    subplot_args: Optional[Dict[str, Any]] = None,
    annotation: Optional[str] = None,
    **kwargs,
) -> Figure:
    if chunk_start_dates is None and chunk_end_dates is None and chunk_indices is None:
        raise InvalidArgumentsException(
            "you must provide either 'chunk_indices' or 'chunk_start_dates' and " "'chunk_end_dates."
        )

    prv_color = figure.data[0].marker.line.color or figure.data[0].line.color
    marker_line_colors = [color if val else prv_color for val in list(alerts)]
    marker_line_widths = [2 if val else 0 for val in list(alerts)]

    if is_time_based_x_axis(chunk_start_dates, chunk_end_dates):
        anchor = cast(Union[np.ndarray, pd.Series], chunk_start_dates)[0]
    else:
        anchor = cast(Union[np.ndarray, pd.Series], chunk_indices)[0]

    max_x_axis = max([bar.xaxis for bar in figure.data])

    for bars in [bar for bar in figure.data if bar.x[0] == anchor and bar.xaxis == max_x_axis]:
        bars.marker.line.color = marker_line_colors
        bars.marker.line.width = marker_line_widths

    return figure
