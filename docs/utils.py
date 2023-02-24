#  Author:  Alejandro Cuenca <alejandro@nannyml.com>
#           Niels Nuyttens   <niels@nannyml.com>
#
#  License: Apache Software License 2.0


def print_multi_index_markdown(df):
    print(_format_multiindex_df(df).to_markdown(tablefmt='grid', stralign='left', numalign='left'))


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
