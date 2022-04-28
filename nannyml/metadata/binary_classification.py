#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import List, Tuple

import pandas as pd

from nannyml.metadata.base import ModelMetadata, ModelType, _check_for_nan, _extract_features


class BinaryClassificationMetadata(ModelMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(model_type=ModelType.CLASSIFICATION_BINARY, *args, **kwargs)

    def is_complete(self) -> Tuple[bool, List[str]]:
        ok, missing = super().is_complete()
        return ok, missing

    def extract(self, data: pd.DataFrame, model_name: str = None, exclude_columns: List[str] = None):
        if super().extract(data, model_name, exclude_columns) is None:
            return None

        targets = _guess_targets(data)
        _check_for_nan(data, targets)
        self.target_column_name = None if len(targets) == 0 else targets[0]  # type: ignore

        not_feature_cols = []
        if exclude_columns:
            not_feature_cols = exclude_columns
        if self.target_column_name:
            not_feature_cols += [self.target_column_name]

        self.features = _extract_features(data, exclude_columns=not_feature_cols)

        return self


def _guess_targets(data: pd.DataFrame) -> List[str]:
    def _guess_if_ground_truth(col: pd.Series) -> bool:
        return col.name in ['target', 'ground_truth', 'actual', 'actuals']

    return [col for col in data.columns if _guess_if_ground_truth(data[col])]
