#  Author:  Alejandro Cuenca <alejandro@nannyml.com>
#           Niels Nuyttens   <niels@nannyml.com>
#
#  License: Apache Software License 2.0

import pandas as pd


def print_table(df):
    print(df.to_markdown(tablefmt="grid", stralign='left', numalign='left'))


def print_multi_index_markdown(df):
    print_table(_format_multiindex_df(df))


def _format_multiindex_columns(df):
    size = len(df.columns.values[0])
    headers = []
    depths = list(range(0, size))
    visited = list(map(lambda _: [], depths))

    for col in df.columns.values:
        if isinstance(col, str):
            headers.append(col)
            continue

        header = ''
        for depth in depths:
            line = col[depth]
            is_visited = line in visited[depth]
            is_last_depth = depth == (len(depths) - 1)
            if not is_visited:
                header += '| ' + line.strip()
                visited[depth].append(line)
                if not is_last_depth:
                    visited[depth + 1] = []
                    header += '\n'
            else:
                header += '\n'

        headers.append(header)

    return headers


def _format_multiindex_df(df):
    headers = _format_multiindex_columns(df)
    single_level_data = df.copy(deep=True)
    single_level_data.columns = headers
    return single_level_data


def print_some_of_the_columns_only_markdown(df, left=2, right=5):
    to_display = df.head().copy()
    to_display = pd.concat(
        [to_display.iloc[:, :left], pd.Series(['...'] * 5, name='...'), to_display.iloc[:, -right:]], axis=1
    )
    print_table(to_display)
