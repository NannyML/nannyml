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

from typing import List, Optional, Tuple, Union

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
from nannyml.thresholds import StandardDeviationThreshold, Threshold
from nannyml.usage_logging import UsageEvent, log_usage


class DataReconstructionDriftCalculator(AbstractCalculator):
    """BaseDriftCalculator implementation using Reconstruction Error as a measure of drift."""

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
            chunker : Chunker, default=None
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
        >>> non_feature_columns = ['timestamp', 'y_pred_proba', 'y_pred', 'repaid']
        >>> feature_column_names = [
        ...     col for col in reference.columns
        ...     if col not in non_feature_columns
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

        self._n_components = n_components

        self.threshold = threshold

        self._scaler = None
        self._encoder = None
        self._pca = None

        self._upper_alert_threshold: Optional[float]
        self._lower_alert_threshold: Optional[float]

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

        # sampling error
        self._sampling_error_components: Tuple = ()

        self.previous_reference_results: Optional[pd.DataFrame] = None

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
        imputed_reference_data = reference_data.copy(deep=True)
        if self.categorical_column_names:
            imputed_reference_data[self.categorical_column_names] = self._imputer_categorical.fit_transform(
                imputed_reference_data[self.categorical_column_names]
            )
        if self.continuous_column_names:
            imputed_reference_data[self.continuous_column_names] = self._imputer_continuous.fit_transform(
                imputed_reference_data[self.continuous_column_names]
            )

        encoder = CountEncoder(cols=self.categorical_column_names, normalize=True)
        encoded_reference_data = imputed_reference_data.copy(deep=True)
        encoded_reference_data[self.column_names] = encoder.fit_transform(encoded_reference_data[self.column_names])

        scaler = StandardScaler()
        scaled_reference_data = pd.DataFrame(
            scaler.fit_transform(encoded_reference_data[self.column_names]), columns=self.column_names
        )

        pca = PCA(n_components=self._n_components, random_state=16)
        pca.fit(scaled_reference_data[self.column_names])

        self._encoder = encoder
        self._scaler = scaler
        self._pca = pca

        # Calculate thresholds
        self._lower_alert_threshold, self._upper_alert_threshold = self._calculate_alert_thresholds(reference_data)

        # Reference stability
        self._sampling_error_components = (
            _calculate_reconstruction_error_for_data(
                column_names=self.column_names,
                categorical_column_names=self.categorical_column_names,
                continuous_column_names=self.continuous_column_names,
                data=reference_data,  # TODO: check with Nikos if this needs to be chunked or not?
                encoder=self._encoder,
                scaler=self._scaler,
                pca=self._pca,
                imputer_categorical=self._imputer_categorical,
                imputer_continuous=self._imputer_continuous,
            ).std(),
        )

        self.result = self._calculate(data=reference_data)
        self.result.data[('chunk', 'period')] = 'reference'

        return self

    @log_usage(UsageEvent.MULTIVAR_DRIFT_CALC_RUN)
    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Calculates the data reconstruction drift for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.column_names, data)

        self.continuous_column_names, self.categorical_column_names = _split_features_by_type(data, self.column_names)

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
                    'sampling_error': sampling_error(self._sampling_error_components, chunk.data),
                    'reconstruction_error': _calculate_reconstruction_error_for_data(
                        column_names=self.column_names,
                        categorical_column_names=self.categorical_column_names,
                        continuous_column_names=self.continuous_column_names,
                        data=chunk.data,
                        encoder=self._encoder,
                        scaler=self._scaler,
                        pca=self._pca,
                        imputer_categorical=self._imputer_categorical,
                        imputer_continuous=self._imputer_continuous,
                    ).mean(),
                }
                for chunk in chunks
            ]
        )
        res['upper_confidence_bound'] = res['reconstruction_error'] + SAMPLING_ERROR_RANGE * res['sampling_error']
        res['lower_confidence_bound'] = res['reconstruction_error'] - SAMPLING_ERROR_RANGE * res['sampling_error']
        res['upper_threshold'] = [self._upper_alert_threshold] * len(res)
        res['lower_threshold'] = [self._lower_alert_threshold] * len(res)
        res['alert'] = _add_alert_flag(res, self._upper_alert_threshold, self._lower_alert_threshold)

        multilevel_index = _create_multilevel_index()
        res.columns = multilevel_index
        res = res.reset_index(drop=True)

        if self.result is None:
            self.result = Result(
                results_data=res,
                timestamp_column_name=self.timestamp_column_name,
                column_names=self.column_names,
                categorical_column_names=self.categorical_column_names,
                continuous_column_names=self.continuous_column_names,
            )
        else:
            self.result = self.result.filter(period='reference')
            self.result.data = pd.concat([self.result.data, res]).reset_index(drop=True)

        return self.result

    def _calculate_alert_thresholds(self, reference_data) -> Tuple[Optional[float], Optional[float]]:
        reference_chunks = self.chunker.split(reference_data)
        reference_reconstruction_error = np.asarray(
            [
                _calculate_reconstruction_error_for_data(
                    column_names=self.column_names,
                    categorical_column_names=self.categorical_column_names,
                    continuous_column_names=self.continuous_column_names,
                    data=chunk.data,
                    encoder=self._encoder,
                    scaler=self._scaler,
                    pca=self._pca,
                    imputer_categorical=self._imputer_categorical,
                    imputer_continuous=self._imputer_continuous,
                ).mean()
                for chunk in reference_chunks
            ]
        )

        return self.threshold.thresholds(reference_reconstruction_error)


def _calculate_reconstruction_error_for_data(
    column_names: List[str],
    categorical_column_names: List[str],
    continuous_column_names: List[str],
    data: pd.DataFrame,
    encoder: CountEncoder,
    scaler: StandardScaler,
    pca: PCA,
    imputer_categorical: SimpleImputer,
    imputer_continuous: SimpleImputer,
) -> pd.Series:
    """Calculates reconstruction error for a single Chunk.

    Parameters
    ----------
    column_names : List[str]
        Subset of features to be included in calculation.
    categorical_column_names : List[str]
        Subset of categorical features to be included in calculation.
    continuous_column_names : List[str]
        Subset of continuous features to be included in calculation.
    data : pd.DataFrame
        The dataset to calculate reconstruction error on
    encoder : category_encoders.CountEncoder
        Encoder used to transform categorical features into a numerical representation
    scaler : sklearn.preprocessing.StandardScaler
        Standardize features by removing the mean and scaling to unit variance
    pca : sklearn.decomposition.PCA
        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space.
    imputer_categorical: SimpleImputer
        The SimpleImputer fitted to impute categorical features in the data.
    imputer_continuous: SimpleImputer
        The SimpleImputer fitted to impute continuous features in the data.

    Returns
    -------
    rce_for_chunk: pd.DataFrame
        A pandas.DataFrame containing the Chunk key and reconstruction error for the given Chunk data.

    """
    # encode categorical features
    data = data.copy(deep=True).reset_index(drop=True)

    # Impute missing values
    if categorical_column_names:
        data[categorical_column_names] = imputer_categorical.transform(data[categorical_column_names])
    if continuous_column_names:
        data[continuous_column_names] = imputer_continuous.transform(data[continuous_column_names])

    data[column_names] = encoder.transform(data[column_names])

    # scale all features
    data[column_names] = scaler.transform(data[column_names])

    # perform dimensionality reduction
    reduced_data = pca.transform(data[column_names])

    # perform reconstruction
    reconstructed = pca.inverse_transform(reduced_data)
    reconstructed_feature_column_names = [f'rf_{col}' for col in column_names]
    reconstructed_data = pd.DataFrame(reconstructed, columns=reconstructed_feature_column_names)

    # combine preprocessed rows with reconstructed rows
    data = pd.concat([data, reconstructed_data], axis=1)

    # calculate reconstruction error using euclidian norm (row-wise between preprocessed and reconstructed value)
    data = data.assign(rc_error=lambda x: _calculate_distance(data, column_names, reconstructed_feature_column_names))

    return data['rc_error']


def _calculate_distance(df: pd.DataFrame, features_preprocessed: List[str], features_reconstructed: List[str]):
    """Calculate row-wise euclidian distance between preprocessed and reconstructed feature values."""
    x1 = df[features_preprocessed]
    x2 = df[features_reconstructed]
    x2.columns = x1.columns

    x = x1.subtract(x2)

    x['rc_error'] = x.apply(lambda row: np.linalg.norm(row), axis=1)
    return x['rc_error']


def _add_alert_flag(
    drift_result: pd.DataFrame, upper_threshold: Optional[float], lower_threshold: Optional[float]
) -> pd.Series:
    alert = drift_result.apply(
        lambda row: True
        if (
            (upper_threshold is not None and row['reconstruction_error'] > upper_threshold)
            or (lower_threshold is not None and row['reconstruction_error'] < lower_threshold)
        )
        else False,
        axis=1,
    )

    return alert


def sampling_error(components: Tuple, data: pd.DataFrame) -> float:
    """Calculates the sampling error with respect to the reference data for a given chunk of data.

    Parameters
    ----------
    components: Tuple
    data: pd.DataFrame
        The data to calculate the sampling error on, with respect to the reference data.

    Returns
    -------
    sampling_error: float
        The expected sampling error.
    """
    return components[0] / np.sqrt(len(data))


def _create_multilevel_index():
    chunk_column_names = ['key', 'chunk_index', 'start_index', 'end_index', 'start_date', 'end_date', 'period']
    method_column_names = [
        'sampling_error',
        'value',
        'upper_confidence_boundary',
        'lower_confidence_boundary',
        'upper_threshold',
        'lower_threshold',
        'alert',
    ]
    chunk_tuples = [('chunk', chunk_column_name) for chunk_column_name in chunk_column_names]
    reconstruction_tuples = [('reconstruction_error', column_name) for column_name in method_column_names]

    tuples = chunk_tuples + reconstruction_tuples

    return MultiIndex.from_tuples(tuples)
