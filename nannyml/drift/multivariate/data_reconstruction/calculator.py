#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Calculates the data reconstruction error on unseen analysis data after fitting on reference data.

This calculator wraps a PCA transformation. It will be fitted on reference data when the `fit` method is called.
On calling the `calculate` method it will perform the inverse transformation on the analysis data and calculate
the euclidian distance between the analysis data and the reconstructed version of it.

This is the data reconstruction error, and it can be used as a measure of drift between
the reference and analysis data sets.

"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from pandas import MultiIndex
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.drift.multivariate.data_reconstruction.result import Result
from nannyml.exceptions import InvalidArgumentsException
from nannyml.sampling_error import SAMPLING_ERROR_RANGE
from nannyml.thresholds import StandardDeviationThreshold, Threshold, calculate_threshold_values
from nannyml.usage_logging import UsageEvent, log_usage


class DataReconstructionDriftCalculator(AbstractCalculator):
    """Multivariate Drift Calculator using PCA Reconstruction Error as a measure of drift."""

    def __init__(
        self,
        column_names: List[str],
        timestamp_column_name: Optional[str] = None,
        n_components: Union[int, float, str] = 0.65,
        chunk_size: Optional[int] = None,
        chunk_number: Optional[int] = None,
        chunk_period: Optional[str] = None,
        chunker: Optional[Chunker] = None,
        imputer_categorical: Optional[SimpleImputer] = None,
        imputer_continuous: Optional[SimpleImputer] = None,
        threshold: Threshold = StandardDeviationThreshold(),
    ):
        """Creates a new DataReconstructionDriftCalculator instance.

        Parameters:
            column_names: List[str]
                A list containing the names of features in the provided data set. All of these features will be used by
                the multivariate data reconstruction drift calculator to calculate an aggregate drift score.
            timestamp_column_name: str, default=None
                The name of the column containing the timestamp of the model prediction.
            n_components: Union[int, float, str], default=0.65
                The n_components parameter as passed to the sklearn.decomposition.PCA constructor.
                See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
            chunk_size: int, default=None
                Splits the data into chunks containing `chunks_size` observations.
                Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
            chunk_number: int, default=None
                Splits the data into `chunk_number` pieces.
                Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
            chunk_period: str, default=None
                Splits the data according to the given period.
                Only one of `chunk_size`, `chunk_number` or `chunk_period` should be given.
            chunker: Chunker, default=None
                The `Chunker` used to split the data sets into a lists of chunks.
            imputer_categorical: SimpleImputer, default=None
                The SimpleImputer used to impute categorical features in the data.
                Defaults to using most_frequent value.
            imputer_continuous: SimpleImputer, default=None
                The SimpleImputer used to impute continuous features in the data. Defaults to using mean value.
            threshold: Threshold, default=StandardDeviationThreshold
                The threshold you wish to evaluate values on. Defaults to a StandardDeviationThreshold with default
                options. The other allowed value is ConstantThreshold.

        Examples:
        >>> import nannyml as nml
        >>> # Load synthetic data
        >>> reference, analysis, _ = nml.load_synthetic_car_loan_dataset()
        >>> feature_column_names = [
        ...     'car_value',
        ...     'salary_range',
        ...     'debt_to_income_ratio',
        ...     'loan_length',
        ...     'repaid_loan_on_prev_car',
        ...     'size_of_downpayment',
        ...     'driver_tenure',
        >>> ]
        >>> calc = nml.DataReconstructionDriftCalculator(
        ...     column_names=feature_column_names,
        ...     timestamp_column_name='timestamp',
        ...     chunk_size=5000
        >>> )
        >>> calc.fit(reference)
        >>> results = calc.calculate(analysis)
        >>> figure = results.plot()
        >>> figure.show()
        """
        super(DataReconstructionDriftCalculator, self).__init__(
            chunk_size, chunk_number, chunk_period, chunker, timestamp_column_name
        )
        self.column_names = column_names
        self.continuous_column_names: List[str] = []
        self.categorical_column_names: List[str] = []
        self.column_name = 'reconstruction_error'
        self._n_components = n_components
        self.threshold = threshold

        self.lower_threshold_value: Optional[float]
        self.upper_threshold_value: Optional[float]
        self.lower_threshold_value_limit: float = 0

        if imputer_categorical:
            if not isinstance(imputer_categorical, SimpleImputer):
                raise TypeError("imputer_categorical needs to be an instantiated SimpleImputer object.")
            if imputer_categorical.strategy not in ["most_frequent", "constant"]:
                raise ValueError("Please use a SimpleImputer strategy appropriate for categorical features.")
        else:
            imputer_categorical = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self._imputer_categorical = imputer_categorical
        if imputer_continuous:
            if not isinstance(imputer_continuous, SimpleImputer):
                raise TypeError("imputer_continuous needs to be an instantiated SimpleImputer object.")
        else:
            imputer_continuous = SimpleImputer(missing_values=np.nan, strategy='mean')
        self._imputer_continuous = imputer_continuous
        self.result: Optional[Result] = None

    @log_usage(UsageEvent.MULTIVAR_DRIFT_CALC_FIT)
    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the drift calculator to a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, reference_data)

        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(
            reference_data, self.column_names
        )

        # TODO: We duplicate the reference data 3 times, here. Improve to something more memory efficient?
        data = reference_data.copy(deep=True)
        if self.categorical_column_names:
            data[self.categorical_column_names] = self._imputer_categorical.fit_transform(
                data[self.categorical_column_names]
            )
        if self.continuous_column_names:
            data[self.continuous_column_names] = self._imputer_continuous.fit_transform(
                data[self.continuous_column_names]
            )

        encoder = CountEncoder(cols=self.categorical_column_names, normalize=True)
        data = encoder.fit_transform(data[self.column_names]).to_numpy()

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        pca = PCA(n_components=self._n_components, random_state=16)
        pca.fit(data)

        self._encoder: CountEncoder = encoder
        self._scaler: StandardScaler = scaler
        self._pca: PCA = pca

        self.result = self._calculate(data=reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    @log_usage(UsageEvent.MULTIVAR_DRIFT_CALC_RUN)
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates the data reconstruction drift for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, data)

        chunks = self.chunker.split(data, columns=self.column_names)

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'chunk_index': chunk.chunk_index,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'period': 'analysis',
                    **self._calculate_chunk_record(chunk.data),
                }
                for chunk in chunks
            ]
        )
        multilevel_index = _create_multilevel_index()
        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self._set_thresholds(results=res)
            res = self._populate_thresholds(results=res)
            self.result = Result(
                results_data=res,
                timestamp_column_name=self.timestamp_column_name,
                column_names=self.column_names,
                categorical_column_names=self.categorical_column_names,
                continuous_column_names=self.continuous_column_names,
            )
        else:
            self.result = self.result.filter(period='reference')
            res = self._populate_thresholds(results=res)
            self.result.data = pd.concat([self.result.data, res], ignore_index=True)
        return self.result

    def _calculate_chunk_record(self, data: pd.DataFrame) -> Dict[str, float]:
        _size = data.shape[0]
        rcerr_mean, rcerr_std = self._calculate_dre_results(data)
        # sampling error based on data distribution on chunk - it's simple std err of mean
        sampling_error = rcerr_std / np.sqrt(_size)
        record = {}
        try:
            record['reconstruction_error'] = rcerr_mean
            record['sampling_error'] = sampling_error
            record['upper_confidence_bound'] = rcerr_mean + SAMPLING_ERROR_RANGE * sampling_error
            record['lower_confidence_bound'] = np.maximum(
                rcerr_mean - SAMPLING_ERROR_RANGE * sampling_error,
                self.lower_threshold_value_limit,
            )
        except Exception as exc:
            record['reconstruction_error'] = np.nan
            record['sampling_error'] = np.nan
            record['upper_confidence_bound'] = np.nan
            record['lower_confidence_bound'] = np.nan
            self._logger.error(
                f"An unexpected error occurred while calculating reconstruction error, returning NaN's: {exc}"
            )
        finally:
            return record

    def _calculate_dre_results(self, data: pd.DataFrame) -> Tuple[float, float]:
        # Impute missing values
        if self.categorical_column_names:
            data[self.categorical_column_names] = self._imputer_categorical.transform(
                data[self.categorical_column_names]
            )  # noqa: E501
        if self.continuous_column_names:
            data[self.continuous_column_names] = self._imputer_continuous.transform(data[self.continuous_column_names])

        data = self._encoder.transform(data[self.column_names]).to_numpy()
        data = self._scaler.transform(data)

        tmp = self._pca.transform(data)
        tmp = self._pca.inverse_transform(tmp)
        tmp = data - tmp
        tmp = np.linalg.norm(tmp, axis=1)

        # std returns nan there is only 1 row
        return (np.mean(tmp), np.std(tmp, ddof=1))

    def _set_thresholds(self, results: pd.DataFrame):
        lower, upper = calculate_threshold_values(
            threshold=self.threshold,
            data=results[(self.column_name, 'value')].to_numpy(),
            lower_threshold_value_limit=self.lower_threshold_value_limit,
            upper_threshold_value_limit=None,
            override_using_none=True,
            logger=self._logger,
            metric_name=self.column_name,
        )
        self.lower_threshold_value = lower
        self.upper_threshold_value = upper

    def _populate_thresholds(self, results: pd.DataFrame):
        results[(self.column_name, 'upper_threshold')] = self.upper_threshold_value
        results[(self.column_name, 'lower_threshold')] = self.lower_threshold_value

        lower_threshold = float('-inf') if self.lower_threshold_value is None else self.lower_threshold_value
        upper_threshold = float('inf') if self.upper_threshold_value is None else self.upper_threshold_value
        results[(self.column_name, 'alert')] = results.apply(
            lambda row: not (lower_threshold < row[(self.column_name, 'value')] < upper_threshold),
            axis=1,
        )
        return results


def _create_multilevel_index():
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    method_column_names = [
        'value',
        'sampling_error',
        'upper_confidence_boundary',
        'lower_confidence_boundary',
    ]
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    reconstruction_tuples = [('reconstruction_error', method_column_name) for method_column_name in method_column_names]
    tuples = chunk_tuples + reconstruction_tuples
    return MultiIndex.from_tuples(tuples)
