#  Author:   Niels Nuyttens  <niels@nannyml.com>
#            Nikolaos Perrakis  <nikos@nannyml.com>
#  License: Apache Software License 2.0
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from category_encoders import CountEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nannyml import BaseDriftCalculator, Chunk, Feature, ModelMetadata


class ReconstructionErrorDriftCalculator(BaseDriftCalculator):
    def __init__(self, n_components: Union[int, float, str] = 0.65):
        """Creates a new ReconstructionErrorDriftCalculator instance.

        Parameters
        ----------
        n_components: Union[int, float, str]
            The n_components parameter as passed to the sklearn.decomposition.PCA constructor.
            See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        """
        super(ReconstructionErrorDriftCalculator, self).__init__()
        self._n_components = n_components

    def _calculate_drift(
        self,
        reference_data: pd.DataFrame,
        chunks: List[Chunk],
        model_metadata: ModelMetadata,
        selected_features: List[str],
    ) -> pd.DataFrame:
        res = pd.DataFrame()
        present_categorical_column_names = _get_present_feature_names(
            selected_features, model_metadata.categorical_features
        )
        encoder = CountEncoder(cols=present_categorical_column_names, normalize=True)
        encoded_reference_data = reference_data.copy(deep=True)
        encoded_reference_data[selected_features] = encoder.fit_transform(reference_data[selected_features])

        scaler = StandardScaler()
        scaled_reference_data = pd.DataFrame(
            scaler.fit_transform(encoded_reference_data[selected_features]), columns=selected_features
        )

        pca = PCA(n_components=self._n_components, random_state=16)
        pca.fit(scaled_reference_data[selected_features])

        for chunk in chunks:
            chunk_drift: Dict[str, Any] = {
                'chunk': chunk.key,
                'reconstruction_error': [
                    _calculate_reconstruction_error_for_chunk(selected_features, chunk, encoder, scaler, pca)
                ],
            }
            res = res.append(pd.DataFrame(chunk_drift))

        res = res.reset_index(drop=True)
        return res


def _calculate_reconstruction_error_for_chunk(
    selected_features: List[str], chunk: Chunk, encoder: CountEncoder, scaler: StandardScaler, pca: PCA
):
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


def _get_present_feature_names(selected_features: List[str], features: List[Feature]) -> List[str]:
    feature_column_names = [f.column_name for f in features]
    return list(set(selected_features) & set(feature_column_names))


def _calculate_distance(df: pd.DataFrame, features_preprocessed: List[str], features_reconstructed: List[str]):
    """
    Calculate distance row wise.
    """

    x1 = df[features_preprocessed]
    x2 = df[features_reconstructed]
    x2.columns = x1.columns

    x = x1.subtract(x2)

    x['rc_error'] = x.apply(lambda row: np.linalg.norm(row), axis=1)
    return x['rc_error']
