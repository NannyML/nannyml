#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates drift for individual columns.

Supported drift detection methods are:

- Kolmogorov-Smirnov statistic (continuous)
- Wasserstein distance (continuous)
- Chi-squared statistic (categorical)
- L-infinity distance (categorical)
- Jensen-Shannon distance
- Hellinger distance

For more information, check out the `tutorial`_ or the `deep dive`_.

For help selecting the correct univariate drift detection method for your use case, check the `method selection guide`_.

.. _tutorial:
    https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/univariate_drift_detection.html

.. _deep dive:
    https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html

.. _method selection guide:
    https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_comparison.html
"""

from __future__ import annotations

import warnings
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import MultiIndex

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.drift.univariate.methods import FeatureType, Method, MethodFactory
from nannyml.drift.univariate.result import Result
from nannyml.exceptions import CalculatorException, InvalidArgumentsException
from nannyml.thresholds import StandardDeviationThreshold, Threshold
from nannyml.usage_logging import UsageEvent, log_usage

DEFAULT_THRESHOLDS: Dict[str, Threshold] = {
    'kolmogorov_smirnov': StandardDeviationThreshold(std_lower_multiplier=None),
    'chi2': StandardDeviationThreshold(std_lower_multiplier=None),  # currently ignored
    'jensen_shannon': StandardDeviationThreshold(std_lower_multiplier=None),
    'wasserstein': StandardDeviationThreshold(std_lower_multiplier=None),
    'hellinger': StandardDeviationThreshold(std_lower_multiplier=None),
    'l_infinity': StandardDeviationThreshold(std_lower_multiplier=None),
}


class UnivariateDriftCalculator(AbstractCalculator):
    """Calculates drift for individual features."""

    def __init__(
        self,
        column_names: Union[str, List[str]],
        treat_as_numerical: Optional[Union[str, List[str]]] = None,
        treat_as_categorical: Optional[Union[str, List[str]]] = None,
        timestamp_column_name: Optional[str] = None,
        categorical_methods: Optional[Union[str, List[str]]] = None,
        continuous_methods: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        thresholds: Optional[Dict[str, Threshold]] = None,
        computation_params: Optional[dict[str, Any]] = None,
    ):
        """Creates a new UnivariateDriftCalculator instance.

        Parameters
        ----------
        column_names: Union[str, List[str]]
            A string or list containing the names of features in the provided data set.
            A drift score will be calculated for each entry in this list.
        treat_as_numerical: Union[str, List[str]]
            A single column name or list of column names to be treated as numerical by the calculator.
        treat_as_categorical: Union[str, List[str]]
            A single column name or list of column names to be treated as categorical by the calculator.
        timestamp_column_name: str
            The name of the column containing the timestamp of the model prediction.
        categorical_methods: Union[str, List[str]], default=['jensen_shannon']
            A method name or list of method names that will be performed on categorical columns.
            Supported methods for categorical variables:

                - `jensen_shannon`
                - `chi2`
                - `hellinger`
                - `l_infinity`
        continuous_methods: Union[str, List[str]], default=['jensen_shannon']
            A method name list of method names that will be performed on continuous columns.
            Supported methods for continuous variables:

                - `jensen_shannon`
                - `kolmogorov_smirnov`
                - `hellinger`
                - `wasserstein`
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
        thresholds: dict
            Defaults to::

                {
                    'kolmogorov_smirnov': StandardDeviationThreshold(std_lower_multiplier=None),
                    'jensen_shannon': StandardDeviationThreshold(std_lower_multiplier=None),
                    'wasserstein': StandardDeviationThreshold(std_lower_multiplier=None),
                    'hellinger': StandardDeviationThreshold(std_lower_multiplier=None),
                    'l_infinity': StandardDeviationThreshold(std_lower_multiplier=None),
                }

            A dictionary allowing users to set a custom threshold for each method. It links a `Threshold` subclass
            to a method name. This dictionary is optional.
            When a dictionary is given its values will override the default values. If no dictionary is given a default
            will be applied. The default method thresholds are as follows:

                - `kolmogorov_smirnov`: `StandardDeviationThreshold(std_lower_multiplier=None)`
                - `jensen_shannon`: `StandardDeviationThreshold(std_lower_multiplier=None)`
                - `wasserstein`: `StandardDeviationThreshold(std_lower_multiplier=None)`
                - `hellinger`: `StandardDeviationThreshold(std_lower_multiplier=None)`
                - `l_infinity`: `StandardDeviationThreshold(std_lower_multiplier=None)`

            The `chi2` method does not support custom thresholds for now. Additional research is required to determine
            how to transition from its current p-value based implementation.

        computation_params: dict
            Defaults to::

                {
                    'kolmogorov_smirnov': {
                        'calculation_method': 'auto',
                        'n_bins':10 000
                    },
                    'wasserstein': {
                        'calculation_method': 'auto',
                        'n_bins':10 000
                    }
                }

            A dictionary which allows users to specify whether they want drift calculated on
            the exact reference data or an estimated distribution of the reference data obtained
            using binning techniques. Applicable only to Kolmogorov-Smirnov and Wasserstein.

            `calculation_method`: Specify whether the entire or the binned reference data will be stored.

                The default value is `auto`.

                - `auto` : Use `exact` for reference data smaller than 10 000 rows, `estimated` for larger.
                - `exact` : Store the whole reference data.

                    When calculating on chunk `scipy.stats.ks_2samp(reference, chunk,  method = `exact` )`
                    is called and whole reference and chunk vectors are passed.
                - `estimated` : Store reference data binned into `n_bins` (default=10 000).

                    The D-statistic will be calculated based on binned eCDF.
                    Bins are quantile-based for Kolmogorov-Smirnov and equal-width based for Wasserstein.
                    Notice that for the reference data of 10 000 rows the resulting D-statistic for exact and
                    estimated methods should be the same. The pvalue in that method is calculated using asymptotic
                    distribution of test statistic (as it is in the `scipy.stats.ks_2samp` with method = `asymp` ).

            `n_bins` : Number of bins used to bin data when calculation_method = `estimated`.

                The default value is 10 000. The larger the value the more precise the calculation
                (closer to  calculation_method = `exact` ) but more data will be stored in the fitted calculator.


        Examples
        --------
        >>> import nannyml as nml
        >>> reference, analysis, _ = nml.load_synthetic_car_price_dataset()
        >>> column_names = [col for col in reference.columns if col not in ['timestamp', 'y_pred', 'y_true']]
        >>> calc = nml.UnivariateDriftCalculator(
        ...   column_names=column_names,
        ...   timestamp_column_name='timestamp',
        ...   continuous_methods=['kolmogorov_smirnov', 'jensen_shannon', 'wasserstein'],
        ...   categorical_methods=['chi2', 'jensen_shannon', 'l_infinity'],
        ... ).fit(reference)
        >>> res = calc.calculate(analysis)
        >>> res = res.filter(period='analysis')
        >>> for column_name in res.continuous_column_names:
        ...  for method in res.continuous_method_names:
        ...    res.plot(kind='drift', column_name=column_name, method=method).show()
        """
        super(UnivariateDriftCalculator, self).__init__(
            chunk_size,
            chunk_number,
            chunk_period,
            chunker,
            timestamp_column_name,
        )
        if isinstance(column_names, str):
            column_names = [column_names]
        self.column_names = column_names

        if not treat_as_numerical:
            treat_as_numerical = []
        if isinstance(treat_as_numerical, str):
            treat_as_numerical = [treat_as_numerical]
        self.treat_as_numerical = treat_as_numerical

        if not treat_as_categorical:
            treat_as_categorical = []
        if isinstance(treat_as_categorical, str):
            treat_as_categorical = [treat_as_categorical]
        self.treat_as_categorical = treat_as_categorical

        if not continuous_methods:
            continuous_methods = ['jensen_shannon']
        elif isinstance(continuous_methods, str):
            continuous_methods = [continuous_methods]
        self.continuous_method_names = continuous_methods

        if not categorical_methods:
            categorical_methods = ['jensen_shannon']
        elif isinstance(categorical_methods, str):
            categorical_methods = [categorical_methods]
        self.categorical_method_names: List[str] = categorical_methods

        self.computation_params: Optional[Dict[str, Any]] = computation_params

        # Setting thresholds: update default values with custom values if given
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds is not None:
            if 'chi2' in thresholds:
                msg = "ignoring custom threshold for 'chi2' as it does not support custom thresholds for now."
                self._logger.warning(msg)
                warnings.warn(msg)

                # thresholds.pop('chi2')  # chi2 has no custom threshold support for now
            self.thresholds.update(**thresholds)

        # set to default values within the method function in methods.py

        self._column_to_models_mapping: Dict[str, List[Method]] = {column_name: [] for column_name in column_names}

        # required for distribution plots
        self.previous_reference_results: Optional[pd.DataFrame] = None
        self.previous_analysis_data: Optional[pd.DataFrame] = None

        self.result: Optional[Result] = None

    @log_usage(
        UsageEvent.UNIVAR_DRIFT_CALC_FIT, metadata_from_self=['continuous_method_names', 'categorical_method_names']
    )
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs) -> UnivariateDriftCalculator:
        """Fits the drift calculator using a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, reference_data)

        self.continuous_column_names, self.categorical_column_names = self._split_continuous_and_categorical(
            reference_data
        )

        timestamps = reference_data[self.timestamp_column_name] if self.timestamp_column_name else None
        for column_name in self.continuous_column_names:
            methods = []
            for method in self.continuous_method_names:
                try:
                    methods.append(
                        MethodFactory.create(
                            key=method,
                            feature_type=FeatureType.CONTINUOUS,
                            chunker=self.chunker,
                            computation_params=self.computation_params or {},
                            threshold=self.thresholds[method],
                        ).fit(
                            reference_data=reference_data[column_name],
                            timestamps=timestamps,
                        )
                    )
                except Exception as ex:
                    raise CalculatorException(f"Failed to fit method {method} for column {column_name}: {ex!r}") from ex
            self._column_to_models_mapping[column_name] = methods

        for column_name in self.categorical_column_names:
            methods = []
            for method in self.categorical_method_names:
                try:
                    methods.append(
                        MethodFactory.create(
                            key=method,
                            feature_type=FeatureType.CATEGORICAL,
                            chunker=self.chunker,
                            threshold=self.thresholds[method],
                        ).fit(
                            reference_data=reference_data[column_name],
                            timestamps=timestamps,
                        )
                    )
                except Exception as ex:
                    raise CalculatorException(f"Failed to fit method {method} for column {column_name}: {ex!r}") from ex
            self._column_to_models_mapping[column_name] = methods

        self.result = self._calculate(reference_data)
        self.result.data['chunk', 'chunk', 'period'] = 'reference'
        self.result.reference_data = reference_data.copy()

        return self

    @log_usage(
        UsageEvent.UNIVAR_DRIFT_CALC_RUN, metadata_from_self=['continuous_method_names', 'categorical_method_names']
    )
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
                    try:
                        for k, v in _calculate_for_column(chunk.data, column_name, method, self._logger).items():
                            row[f'{column_name}_{method.column_name}_{k}'] = v
                    except Exception as exc:
                        self._logger.error(
                            f"an unexpected exception occurred during calculation of method '{method.display_name}': "
                            f"{exc}"
                        )
                        continue

            for column_name in self.categorical_column_names:
                for method in self._column_to_models_mapping[column_name]:
                    try:
                        for k, v in _calculate_for_column(chunk.data, column_name, method, self._logger).items():
                            row[f'{column_name}_{method.column_name}_{k}'] = v
                    except Exception as exc:
                        self._logger.error(
                            f"an unexpected exception occurred during calculation of method '{method.display_name}': "
                            f"{exc}"
                        )
                        continue

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
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, res], ignore_index=True)
            self.result.analysis_data = data.copy()

        return self.result

    def _split_continuous_and_categorical(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Splits the features in the data set into continuous and categorical features."""
        treat_as_numerical_set, treat_as_categorical_set = set(self.treat_as_numerical), set(self.treat_as_categorical)
        column_names_set = set(self.column_names)

        invalid_continuous_column_names = treat_as_numerical_set - column_names_set
        treat_as_numerical_set = treat_as_numerical_set - invalid_continuous_column_names
        if invalid_continuous_column_names:
            self._logger.info(
                f"ignoring 'treat_as_numerical' values {list(invalid_continuous_column_names)} because "
                f"they were not in listed column names"
            )

        invalid_categorical_column_names = treat_as_categorical_set - column_names_set
        treat_as_categorical_set = treat_as_categorical_set - invalid_categorical_column_names
        if invalid_categorical_column_names:
            self._logger.info(
                f"ignoring 'treat_as_categorical' values {list(invalid_categorical_column_names)} because "
                f"they were not in listed column names"
            )

        unspecified_columns = column_names_set - treat_as_numerical_set - treat_as_categorical_set
        continuous_column_names, categorical_column_names = _split_features_by_type(data, unspecified_columns)

        continuous_column_names = continuous_column_names + list(treat_as_numerical_set)
        categorical_column_names = categorical_column_names + list(treat_as_categorical_set)

        return continuous_column_names, categorical_column_names


def _calculate_for_column(
    data: pd.DataFrame, column_name: str, method: Method, logger: Optional[Logger] = None
) -> Dict[str, Any]:
    result = {}
    try:
        value = method.calculate(data[column_name])
        result['value'] = value
        result['upper_threshold'] = method.upper_threshold_value
        result['lower_threshold'] = method.lower_threshold_value
        result['alert'] = method.alert(value)
    except Exception as exc:
        if logger:
            logger.error(
                f"an unexpected exception occurred during calculation of method '{method.display_name}': " f"{exc}"
            )
        result['value'] = np.nan
        result['upper_threshold'] = method.upper_threshold_value
        result['lower_threshold'] = method.lower_threshold_value
        result['alert'] = np.nan
    finally:
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
