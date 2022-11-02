#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for individual features using the `Kolmogorov-Smirnov` and `chi2-contingency` statistical tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import MultiIndex

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.drift.univariate.methods import FeatureType, Method, MethodFactory
from nannyml.drift.univariate.result import Result
from nannyml.exceptions import InvalidArgumentsException


class UnivariateDriftCalculator(AbstractCalculator):
    """Calculates drift for individual features."""

    def __init__(
        self,
        column_names: List[str],
        timestamp_column_name: Optional[str] = None,
        categorical_methods: List[str] = None,
        continuous_methods: List[str] = None,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new UnivariateDriftCalculator instance.

        Parameters
        ----------
        column_names: List[str]
            A list containing the names of features in the provided data set.
            A drift score will be calculated for each entry in this list.
        timestamp_column_name: str
            The name of the column containing the timestamp of the model prediction.
        categorical_methods: List[str], default=None
            A list of method names that will be performed on categorical columns.
            When not given all available methods supporting categorical columns will be used.
        continuous_methods: List[str], default=None
            A list of method names that will be performed on continuous columns.
            When not given all available methods supporting continuous columns will be used.
        chunk_size: int
            Splits the data into chunks containing `chunks_size` observations.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_number: int
            Splits the data into `chunk_number` pieces.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunk_period: str
            Splits the data according to the given period.
            Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
        chunker : Chunker
            The `Chunker` used to split the data sets into a lists of chunks.

        Examples
        --------
        >>> import nannyml as nml
        >>> reference, analysis, _ = nml.load_synthetic_car_price_dataset()
        >>> column_names = [col for col in reference.columns if col not in ['timestamp', 'y_pred', 'y_true']]
        >>> calc = nml.UnivariateDriftCalculator(
        ...   column_names=column_names,
        ...   timestamp_column_name='timestamp',
        ...   continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        ...   categorical_methods=['chi2', 'jensen_shannon'],
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> res = res.filter(period='analysis')
        >>> for column_name in res.continuous_column_names:
        ...  for method in res.continuous_method_names:
        ...    res.plot(kind='drift', column_name=column_name, method=method).show()
        """
        super(UnivariateDriftCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )

        self.column_names = column_names
        self.continuous_method_names = continuous_methods or ['kolmogorov_smirnov']
        self.categorical_method_names = categorical_methods or ['chi2']

        self._column_to_models_mapping: Dict[str, List[Method]] = {column_name: [] for column_name in column_names}

        # required for distribution plots
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

        self.result: Optional[Result] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> UnivariateDriftCalculator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, reference_data)

        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
            reference_data, self.column_names
        )

        for column_name in self.continuous_column_names:
            self._column_to_models_mapping[column_name] += [
                MethodFactory.create(key=method, feature_type=FeatureType.CONTINUOUS).fit(reference_data[column_name])
                for method in self.continuous_method_names
            ]

        for column_name in self.categorical_column_names:
            self._column_to_models_mapping[column_name] += [
                MethodFactory.create(key=method, feature_type=FeatureType.CATEGORICAL).fit(reference_data[column_name])
                for method in self.categorical_method_names
            ]

        self.result = self._calculate(reference_data)
        self.result.data['chunk', 'chunk', 'period'] = 'reference'
        self.result.reference_data = reference_data.copy()

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates methods for both categorical and continuous columns."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, data)

        chunks = self.chunker.split(data)

        rows = []
        for chunk in chunks:
            row = {
                'key': chunk.key,
                'chunk_index': chunk.chunk_index,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_datetime': chunk.start_datetime,
                'end_datetime': chunk.end_datetime,
                'period': 'analysis',
            }

            for column_name in self.continuous_column_names:
                for method in self._column_to_models_mapping[column_name]:
                    for k, v in _calculate_for_column(chunk.data, column_name, method).items():
                        row[f'{column_name}_{method.column_name}_{k}'] = v

            for column_name in self.categorical_column_names:
                for method in self._column_to_models_mapping[column_name]:
                    for k, v in _calculate_for_column(chunk.data, column_name, method).items():
                        row[f'{column_name}_{method.column_name}_{k}'] = v

            rows.append(row)

        result_index = _create_multilevel_index(
            continuous_column_names=self.continuous_column_names,
            continuous_method_names=[m for m in self.continuous_method_names],
            categorical_column_names=self.categorical_column_names,
            categorical_method_names=[m for m in self.categorical_method_names],
        )
        res = pd.DataFrame(rows)
        res.columns = result_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self.result = Result(
                results_data=res,
                column_names=self.column_names,
                continuous_column_names=self.continuous_column_names,
                categorical_column_names=self.categorical_column_names,
                continuous_method_names=self.continuous_method_names,
                categorical_method_names=self.categorical_method_names,
                timestamp_column_name=self.timestamp_column_name,
                chunker=self.chunker,
            )
        else:
            # TODO: review subclassing setup => superclass + '_filter' is screwing up typing.
            #       Dropping the intermediate '_filter' and directly returning the correct 'Result' class works OK
            #       but this causes us to lose the "common behavior" in the top level 'filter' method when overriding.
            #       Applicable here but to many of the base classes as well (e.g. fitting and calculating)
            self.result = self.result.filter(period='reference')  # type: ignore
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)
            self.result.analysis_data = data.copy()

        return self.result


def _calculate_for_column(data: pd.DataFrame, column_name: str, method: Method) -> Dict[str, Any]:
    result = {}
    value = method.calculate(data[column_name])
    result['value'] = value
    result['upper_threshold'] = method.upper_threshold
    result['lower_threshold'] = method.lower_threshold
    result['alert'] = method.alert(data[column_name])
    return result


def _create_multilevel_index(
    continuous_column_names: List[str],
    categorical_column_names: List[str],
    continuous_method_names: List[str],
    categorical_method_names: List[str],
):
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    method_column_names = ['value', 'upper_threshold', 'lower_threshold', 'alert']
    chunk_tuples = [('chunk', 'chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    continuous_column_tuples = [
        (column_name, method_name, method_column_name)
        for column_name in continuous_column_names
        for method_name in continuous_method_names
        for method_column_name in method_column_names
    ]

    categorical_column_tuples = [
        (column_name, method_name, method_column_name)
        for column_name in categorical_column_names
        for method_name in categorical_method_names
        for method_column_name in method_column_names
    ]

    tuples = chunk_tuples + continuous_column_tuples + categorical_column_tuples

    return MultiIndex.from_tuples(tuples)
