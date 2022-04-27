#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import List, Tuple

import pandas as pd

from nannyml.metadata.base import ModelMetadata, ModelType


class BinaryClassificationMetadata(ModelMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(model_type=ModelType.CLASSIFICATION_BINARY, *args, **kwargs)

    def is_complete(self) -> Tuple[bool, List[str]]:
        ok, missing = super().is_complete()
        return ok, missing

    def extract(self, data: pd.DataFrame, model_name: str = None, exclude_columns: List[str] = None):
        md = super().extract(data, model_name, exclude_columns)
        return md
