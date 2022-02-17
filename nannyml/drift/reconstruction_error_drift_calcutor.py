#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Nikolaos Perrakis  <nikos@nannyml.com>
#  License: Apache Software License 2.0

"""Drift calculator using Reconstruction Error as a measure of drift."""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nannyml.chunk import Chunk
from nannyml.drift import BaseDriftCalculator
from nannyml.metadata import Feature


class ReconstructionErrorDriftCalculator(BaseDriftCalculator):
    """BaseDriftCalculator implementation using Reconstruction Error as a measure of drift."""

    def __init__(self, model_metadata, features: List[str] = None, n_components: Union[int, float, str] = 0.65):
        """Creates a new ReconstructionErrorDriftCalculator instance.

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

        """
        super(ReconstructionErrorDriftCalculator, self).__init__(model_metadata, features)
        self._n_components = n_components

        self._scaler = None
        self._encoder = None
        self._pca = None

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
                    _calculate_reconstruction_error_for_chunk(
                        self.selected_features, chunk, self._encoder, self._scaler, self._pca
                    )
                ],
            }
            res = res.append(pd.DataFrame(chunk_drift))

        res = res.reset_index(drop=True)
        return res


def _calculate_reconstruction_error_for_chunk(
    selected_features: List[str], chunk: Chunk, encoder: CountEncoder, scaler: StandardScaler, pca: PCA
) -> pd.DataFrame:
    """Calculates reconstruction error for a single Chunk.

    Parameters
    ----------
    selected_features : List[str]
        Subset of features to be included in calculation.
    chunk : Chunk
        The chunk containing data to calculate reconstruction error on
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
    data = chunk.data.reset_index(drop=True)
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
