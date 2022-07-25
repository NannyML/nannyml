#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Nikolaos Perrakis  <nikos@nannyml.com>
#  License: Apache Software License 2.0

"""Drift calculator using Reconstruction Error as a measure of drift."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nannyml.base import AbstractCalculator, _list_missing, _split_features_by_type
from nannyml.chunk import Chunker
from nannyml.drift.model_inputs.multivariate.data_reconstruction.results import DataReconstructionDriftCalculatorResult
from nannyml.exceptions import InvalidArgumentsException


class DataReconstructionDriftCalculator(AbstractCalculator):
    """BaseDriftCalculator implementation using Reconstruction Error as a measure of drift."""

    def __init__(
        self,
        feature_column_names: List[str],
        timestamp_column_name: str,
        n_components: Union[int, float, str] = 0.65,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
        imputer_categorical: SimpleImputer = None,
        imputer_continuous: SimpleImputer = None,
    ):
        """Creates a new DataReconstructionDriftCalculator instance.

        Parameters
        ----------
        feature_column_names: List[str]
            A list containing the names of features in the provided data set. All of these features will be used by
            the multivariate data reconstruction drift calculator to calculate an aggregate drift score.
        timestamp_column_name: str
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
            The SimpleImputer used to impute categorical features in the data. Defaults to using most_frequent value.
        imputer_continuous: SimpleImputer, default=None
            The SimpleImputer used to impute continuous features in the data. Defaults to using mean value.

        Examples
        --------
        >>> import nannyml as nml
        >>> reference_df, analysis_df, _ = nml.load_synthetic_binary_classification_dataset()
        >>>
        >>> feature_column_names = [col for col in reference_df.columns
        >>>                         if col not in ['y_pred', 'y_pred_proba', 'work_home_actual', 'timestamp']]
        >>> calc = nml.DataReconstructionDriftCalculator(
        >>>     feature_column_names=feature_column_names,
        >>>     timestamp_column_name='timestamp'
        >>> )
        >>> calc.fit(reference_df)
        >>> results = calc.calculate(analysis_df)
        >>> print(results.data)  # access the numbers
                             key  start_index  ...  upper_threshold alert
        0       [0:4999]            0  ...         1.511762  True
        1    [5000:9999]         5000  ...         1.511762  True
        2  [10000:14999]        10000  ...         1.511762  True
        3  [15000:19999]        15000  ...         1.511762  True
        4  [20000:24999]        20000  ...         1.511762  True
        5  [25000:29999]        25000  ...         1.511762  True
        6  [30000:34999]        30000  ...         1.511762  True
        7  [35000:39999]        35000  ...         1.511762  True
        8  [40000:44999]        40000  ...         1.511762  True
        9  [45000:49999]        45000  ...         1.511762  True
        >>> fig = results.plot(kind='drift', plot_reference=True)
        >>> fig.show()
        """
        super(DataReconstructionDriftCalculator, self).__init__(chunk_size, chunk_number, chunk_period, chunker)
        self.feature_column_names = feature_column_names
        self.continuous_feature_column_names: List[str] = []
        self.categorical_feature_column_names: List[str] = []

        self.timestamp_column_name = timestamp_column_name
        self._n_components = n_components

        self._scaler = None
        self._encoder = None
        self._pca = None

        self._upper_alert_threshold: Optional[float] = None
        self._lower_alert_threshold: Optional[float] = None

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

        self.previous_reference_results: Optional[pd.DataFrame] = None

    def _fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the drift calculator to a set of reference data."""
        if reference_data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, reference_data)

        self.continuous_feature_column_names, self.categorical_feature_column_names = _split_features_by_type(
            reference_data, self.feature_column_names
        )

        # TODO: We duplicate the reference data 3 times, here. Improve to something more memory efficient?
        imputed_reference_data = reference_data.copy(deep=True)
        if len(self.categorical_feature_column_names) > 0:
            imputed_reference_data[self.categorical_feature_column_names] = self._imputer_categorical.fit_transform(
                imputed_reference_data[self.categorical_feature_column_names]
            )
        if len(self.continuous_feature_column_names) > 0:
            imputed_reference_data[self.continuous_feature_column_names] = self._imputer_continuous.fit_transform(
                imputed_reference_data[self.continuous_feature_column_names]
            )

        encoder = CountEncoder(cols=self.categorical_feature_column_names, normalize=True)
        encoded_reference_data = imputed_reference_data.copy(deep=True)
        encoded_reference_data[self.feature_column_names] = encoder.fit_transform(
            encoded_reference_data[self.feature_column_names]
        )

        scaler = StandardScaler()
        scaled_reference_data = pd.DataFrame(
            scaler.fit_transform(encoded_reference_data[self.feature_column_names]), columns=self.feature_column_names
        )

        pca = PCA(n_components=self._n_components, random_state=16)
        pca.fit(scaled_reference_data[self.feature_column_names])

        self._encoder = encoder
        self._scaler = scaler
        self._pca = pca

        # Calculate thresholds
        self._upper_alert_threshold, self._lower_alert_threshold = self._calculate_alert_thresholds(reference_data)

        self.previous_reference_results = self._calculate(data=reference_data).data

        return self

    def _calculate(self, data: pd.DataFrame, *args, **kwargs) -> DataReconstructionDriftCalculatorResult:
        """Calculates the data reconstruction drift for a given data set."""
        if data.empty:
            raise InvalidArgumentsException('data contains no rows. Please provide a valid data set.')

        _list_missing(self.feature_column_names, data)

        self.continuous_feature_column_names, self.categorical_feature_column_names = _split_features_by_type(
            data, self.feature_column_names
        )

        chunks = self.chunker.split(
            data,
            columns=self.feature_column_names,
            minimum_chunk_size=_minimum_chunk_size(self.feature_column_names),
            timestamp_column_name=self.timestamp_column_name,
        )

        res = pd.DataFrame.from_records(
            [
                {
                    'key': chunk.key,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'start_date': chunk.start_datetime,
                    'end_date': chunk.end_datetime,
                    'reconstruction_error': _calculate_reconstruction_error_for_data(
                        feature_column_names=self.feature_column_names,
                        categorical_feature_column_names=self.categorical_feature_column_names,
                        continuous_feature_column_names=self.continuous_feature_column_names,
                        data=chunk.data,
                        encoder=self._encoder,
                        scaler=self._scaler,
                        pca=self._pca,
                        imputer_categorical=self._imputer_categorical,
                        imputer_continuous=self._imputer_continuous,
                    ),
                }
                for chunk in chunks
            ]
        )

        res['lower_threshold'] = [self._lower_alert_threshold] * len(res)
        res['upper_threshold'] = [self._upper_alert_threshold] * len(res)
        res['alert'] = _add_alert_flag(res, self._upper_alert_threshold, self._lower_alert_threshold)  # type: ignore
        res = res.reset_index(drop=True)
        return DataReconstructionDriftCalculatorResult(results_data=res, calculator=self)

    def _calculate_alert_thresholds(self, reference_data) -> Tuple[float, float]:
        reference_chunks = self.chunker.split(reference_data, self.timestamp_column_name)  # type: ignore
        reference_reconstruction_error = pd.Series(
            [
                _calculate_reconstruction_error_for_data(
                    feature_column_names=self.feature_column_names,
                    categorical_feature_column_names=self.categorical_feature_column_names,
                    continuous_feature_column_names=self.continuous_feature_column_names,
                    data=chunk.data,
                    encoder=self._encoder,
                    scaler=self._scaler,
                    pca=self._pca,
                    imputer_categorical=self._imputer_categorical,
                    imputer_continuous=self._imputer_continuous,
                )
                for chunk in reference_chunks
            ]
        )

        return (
            reference_reconstruction_error.mean() + 3 * reference_reconstruction_error.std(),
            reference_reconstruction_error.mean() - 3 * reference_reconstruction_error.std(),
        )


def _calculate_reconstruction_error_for_data(
    feature_column_names: List[str],
    categorical_feature_column_names: List[str],
    continuous_feature_column_names: List[str],
    data: pd.DataFrame,
    encoder: CountEncoder,
    scaler: StandardScaler,
    pca: PCA,
    imputer_categorical: SimpleImputer,
    imputer_continuous: SimpleImputer,
) -> pd.DataFrame:
    """Calculates reconstruction error for a single Chunk.

    Parameters
    ----------
    feature_column_names : List[str]
        Subset of features to be included in calculation.
    categorical_feature_column_names : List[str]
        Subset of categorical features to be included in calculation.
    continuous_feature_column_names : List[str]
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
    data = data.reset_index(drop=True)

    # Impute missing values
    if len(categorical_feature_column_names) > 0:
        data[categorical_feature_column_names] = imputer_categorical.transform(data[categorical_feature_column_names])
    if len(continuous_feature_column_names) > 0:
        data[continuous_feature_column_names] = imputer_continuous.transform(data[continuous_feature_column_names])

    data[feature_column_names] = encoder.transform(data[feature_column_names])

    # scale all features
    data[feature_column_names] = scaler.transform(data[feature_column_names])

    # perform dimensionality reduction
    reduced_data = pca.transform(data[feature_column_names])

    # perform reconstruction
    reconstructed = pca.inverse_transform(reduced_data)
    reconstructed_feature_column_names = [f'rf_{col}' for col in feature_column_names]
    reconstructed_data = pd.DataFrame(reconstructed, columns=reconstructed_feature_column_names)

    # combine preprocessed rows with reconstructed rows
    data = pd.concat([data, reconstructed_data], axis=1)

    # calculate reconstruction error using euclidian norm (row-wise between preprocessed and reconstructed value)
    data = data.assign(
        rc_error=lambda x: _calculate_distance(data, feature_column_names, reconstructed_feature_column_names)
    )

    res = data['rc_error'].mean()
    return res


def _calculate_distance(df: pd.DataFrame, features_preprocessed: List[str], features_reconstructed: List[str]):
    """Calculate row-wise euclidian distance between preprocessed and reconstructed feature values."""
    x1 = df[features_preprocessed]
    x2 = df[features_reconstructed]
    x2.columns = x1.columns

    x = x1.subtract(x2)

    x['rc_error'] = x.apply(lambda row: np.linalg.norm(row), axis=1)
    return x['rc_error']


def _add_alert_flag(drift_result: pd.DataFrame, upper_threshold: float, lower_threshold: float) -> pd.Series:
    alert = drift_result.apply(
        lambda row: True
        if (row['reconstruction_error'] > upper_threshold or row['reconstruction_error'] < lower_threshold)
        else False,
        axis=1,
    )

    return alert


def _minimum_chunk_size(
    features: List[str] = None,
) -> int:

    return int(20 * np.power(len(features), 5 / 6))  # type: ignore
