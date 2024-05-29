from typing import List, Optional, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from nannyml import Chunker
from nannyml.base import AbstractCalculator, _list_missing
from nannyml.distribution.categorical.result import Result
from nannyml.exceptions import InvalidArgumentsException


class CategoricalDistributionCalculator(AbstractCalculator):
    def __init__(
        self,
        column_names: Union[str, List[str]],
        timestamp_column_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
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
        self._was_fitted: bool = False

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> Self:
        self.result = self._calculate(reference_data)
        self._was_fitted = True

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, data)

        # result_data = pd.DataFrame(columns=_create_multilevel_index(self.column_names))
        result_data = pd.DataFrame()

        chunks = self.chunker.split(data)
        chunks_data = pd.DataFrame(
            {
                'key': [c.key for c in chunks],
                'chunk_index': [c.chunk_index for c in chunks],
                'start_datetime': [c.start_datetime for c in chunks],
                'end_datetime': [c.end_datetime for c in chunks],
                'start_index': [c.start_index for c in chunks],
                'end_index': [c.end_index for c in chunks],
                'period': ['analysis' if self._was_fitted else 'reference' for _ in chunks],
            }
        )

        for column in self.column_names:
            value_counts = calculate_value_counts(
                data=data[column],
                chunker=self.chunker,
                timestamps=data.get(self.timestamp_column_name, default=None),
                max_number_of_categories=5,
                missing_category_label='Missing',
                column_name=column,
            )
            result_data = pd.concat([result_data, pd.merge(chunks_data, value_counts, on='chunk_index')])

        # result_data.index = pd.MultiIndex.from_tuples(list(zip(result_data['column_name'], result_data['value'])))

        if self.result is None:
            self.result = Result(result_data, self.column_names, self.timestamp_column_name, self.chunker)
        else:
            # self.result = self.result.data.loc[self.result.data['period'] == 'reference', :]
            self.result.data = pd.concat([self.result.data, result_data], ignore_index=True)

        return self.result


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

    value_counts_table = (
        data_with_chunk_keys.groupby(['chunk_index'])[column_name]
        .value_counts()
        .to_frame('value_counts')
        .sort_values(by=['chunk_index', 'value_counts'])
        .reset_index()
        .rename(columns={column_name: 'value'})
        .assign(column_name=column_name)
    )

    value_counts_table['value_counts_total'] = value_counts_table['chunk_index'].map(
        value_counts_table.groupby('chunk_index')['value_counts'].sum()
    )
    value_counts_table['value_counts_normalised'] = (
        value_counts_table['value_counts'] / value_counts_table['value_counts_total']
    )

    return value_counts_table
