#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
import pandas as pd

from nannyml.metadata import ModelMetadata


class TargetDistributionResult:
    def __init__(self, target_distribution: pd.DataFrame, model_metadata: ModelMetadata):
        self.data = target_distribution
        self.metadata = model_metadata
