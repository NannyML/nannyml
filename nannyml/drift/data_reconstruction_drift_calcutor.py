#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Nikolaos Perrakis  <nikos@nannyml.com>
#  License: Apache Software License 2.0

"""Drift calculator using Reconstruction Error as a measure of drift."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nannyml.chunk import Chunk, Chunker
from nannyml.drift import BaseDriftCalculator
from nannyml.metadata import Feature


class DataReconstructionDriftCalculator(BaseDriftCalculator):
    """BaseDriftCalculator implementation using Reconstruction Error as a measure of drift."""

    def __init__(
        self,
        model_metadata,
        features: List[str] = None,
        n_components: Union[int, float, str] = 0.65,
        chunk_size: int = None,
        chunk_number: int = None,
        chunk_period: str = None,
        chunker: Chunker = None,
    ):
        """Creates a new DataReconstructionDriftCalculator instance.

        Parameters
        ----------
        model_metadata: ModelMetadata
            Metadata for the model whose data is to be processed.
        features: List[str], default=None
            An optional list of feature names to use during drift calculation. None by default, in this case
            all features are used during calculation.
        n_components: Union[int, float, str]
            The n_components parameter as passed to the sklearn.decomposition.PCA constructor.
            See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
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
        """
        super(DataReconstructionDriftCalculator, self).__init__(
            model_metadata, features, chunk_size, chunk_number, chunk_period, chunker
        )
        self._n_components = n_components

        self._scaler = None
        self._encoder = None
        self._pca = None

        self._upper_alert_threshold: Optional[float] = None
        self._lower_alert_threshold: Optional[float] = None

    def _fit(self, reference_data: pd.DataFrame):
        selected_categorical_column_names = _get_selected_feature_names(
            self.selected_features, self.model_metadata.categorical_features
        )

        encoder = CountEncoder(cols=selected_categorical_column_names, normalize=True)
        encoded_reference_data = reference_data.copy(deep=True)
        encoded_reference_data[self.selected_features] = encoder.fit_transform(reference_data[self.selected_features])

        scaler = StandardScaler()
        scaled_reference_data = pd.DataFrame(
            scaler.fit_transform(encoded_reference_data[self.selected_features]), columns=self.selected_features
        )

        pca = PCA(n_components=self._n_components, random_state=16)
        pca.fit(scaled_reference_data[self.selected_features])

        self._encoder = encoder
        self._scaler = scaler
        self._pca = pca

        # Calculate thresholds
        self._upper_alert_threshold, self._lower_alert_threshold = self._calculate_alert_thresholds(reference_data)

    def _calculate_drift(
        self,
        chunks: List[Chunk],
    ) -> pd.DataFrame:
        res = pd.DataFrame()

        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'key': chunk.key,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'start_date': chunk.start_datetime,
                'end_date': chunk.end_datetime,
                'partition': 'analysis' if chunk.is_transition else chunk.partition,
                'reconstruction_error': [
                    _calculate_reconstruction_error_for_data(
                        self.selected_features, chunk.data, self._encoder, self._scaler, self._pca
                    )
                ],
            }
            res = res.append(pd.DataFrame(chunk_drift))

        res['lower_threshold'] = [self._lower_alert_threshold] * len(res)
        res['upper_threshold'] = [self._upper_alert_threshold] * len(res)
        res['alert'] = _add_alert_flag(res, self._upper_alert_threshold, self._lower_alert_threshold)  # type: ignore
        res = res.reset_index(drop=True)
        return res

    def _calculate_alert_thresholds(self, reference_data) -> Tuple[float, float]:
        reference_chunks = self.chunker.split(reference_data)  # type: ignore

        reference_reconstruction_error = pd.Series(
            [
                _calculate_reconstruction_error_for_data(
                    self.selected_features, chunk.data, self._encoder, self._scaler, self._pca
                )
                for chunk in reference_chunks
            ]
        )

        return (
            reference_reconstruction_error.mean() + 2 * reference_reconstruction_error.std(),
            reference_reconstruction_error.mean() - 2 * reference_reconstruction_error.std(),
        )


def _calculate_reconstruction_error_for_data(
    selected_features: List[str], data: pd.DataFrame, encoder: CountEncoder, scaler: StandardScaler, pca: PCA
) -> pd.DataFrame:
    """Calculates reconstruction error for a single Chunk.

    Parameters
    ----------
    selected_features : List[str]
        Subset of features to be included in calculation.
    data : pd.DataFrame
        The dataset to calculate reconstruction error on
    encoder : category_encoders.CountEncoder
        Encoder used to transform categorical features into a numerical representation
    scaler : sklearn.preprocessing.StandardScaler
        Standardize features by removing the mean and scaling to unit variance
    pca : sklearn.decomposition.PCA
        Linear dimensionality reduction using Singular Value Decomposition of the
        data to project it to a lower dimensional space.

    Returns
    -------
    rce_for_chunk: pd.DataFrame
        A pandas.DataFrame containing the Chunk key and reconstruction error for the given Chunk data.

    """
    # encode categorical features
    data = data.reset_index(drop=True)
    data[selected_features] = encoder.transform(data[selected_features])

    # scale all features
    data[selected_features] = scaler.transform(data[selected_features])

    # perform dimensionality reduction
    reduced_data = pca.transform(data[selected_features])

    # perform reconstruction
    reconstructed = pca.inverse_transform(reduced_data)
    reconstructed_feature_column_names = [f'rf_{col}' for col in selected_features]
    reconstructed_data = pd.DataFrame(reconstructed, columns=reconstructed_feature_column_names)

    # combine preprocessed rows with reconstructed rows
    data = pd.concat([data, reconstructed_data], axis=1)

    # calculate reconstruction error using euclidian norm (row-wise between preprocessed and reconstructed value)
    data = data.assign(
        rc_error=lambda x: _calculate_distance(data, selected_features, reconstructed_feature_column_names)
    )

    res = data['rc_error'].mean()
    return res


def _get_selected_feature_names(selected_features: List[str], features: List[Feature]) -> List[str]:
    feature_column_names = [f.column_name for f in features]
    # Calculate intersection
    return list(set(selected_features) & set(feature_column_names))


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
        if row['reconstruction_error'] > upper_threshold or row['reconstruction_error'] < lower_threshold
        else False,
        axis=1,
    )

    return alert
