#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from nannyml.metadata.base import ModelMetadata, ModelType, _check_for_nan, _extract_features
from nannyml.metadata.feature import FeatureType

NML_METADATA_PREDICTION_COLUMN_NAME = 'nml_meta_prediction'


class RegressionMetadata(ModelMetadata):
    def __init__(self, prediction_column_name: str = None, *args, **kwargs):
        super().__init__(ModelType.REGRESSION, *args, **kwargs)
        self._prediction_column_name = prediction_column_name

    @property
    def prediction_column_name(self):  # noqa: D102
        return self._prediction_column_name

    @prediction_column_name.setter
    def prediction_column_name(self, column_name: str):  # noqa: D102
        self._prediction_column_name = column_name
        self._remove_from_features(column_name)

    @property
    def metadata_columns(self):
        return [NML_METADATA_PREDICTION_COLUMN_NAME]

    def to_dict(self) -> Dict[str, Any]:
        res = super().to_dict()
        res['prediction_column_name'] = self.prediction_column_name
        return res

    def to_df(self) -> pd.DataFrame:
        res = super().to_df()
        df = pd.DataFrame(
            [
                {
                    'label': 'prediction_column_name',
                    'column_name': self.prediction_column_name,
                    'type': FeatureType.CONTINUOUS.value,
                    'description': 'predicted value',
                }
            ]
        )
        return res.append(df, ignore_index=True).reset_index(drop=True)

    def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
        df = super().enrich(data)

        if self.prediction_column_name in data.columns:
            df[NML_METADATA_PREDICTION_COLUMN_NAME] = data[self.prediction_column_name]
        else:
            df[NML_METADATA_PREDICTION_COLUMN_NAME] = np.NAN

        return df

    def is_complete(self) -> Tuple[bool, List[str]]:
        ok, missing = super().is_complete()

        if self.prediction_column_name is None:
            ok = False
            missing.append('prediction_column_name')

        return ok, missing

    def extract(self, data: pd.DataFrame, model_name: str = None, exclude_columns: List[str] = None):
        if super().extract(data, model_name, exclude_columns) is None:
            return None

        predictions = _guess_predictions(data)
        _check_for_nan(data, predictions)
        self.prediction_column_name = None if len(predictions) == 0 else predictions[0]  # type: ignore

        not_feature_cols = []
        if exclude_columns:
            not_feature_cols = exclude_columns

        if self.prediction_column_name:
            not_feature_cols += [self.prediction_column_name]

        self.features = _extract_features(data, exclude_columns=not_feature_cols)

        return self


def _guess_predictions(data: pd.DataFrame) -> List[str]:
    def _guess_if_prediction(col: pd.Series) -> bool:
        return col.name in ['p', 'pred', 'prediction', 'out', 'output', 'y_pred']

    return [col for col in data.columns if _guess_if_prediction(data[col])]
