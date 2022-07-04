#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from enum import Enum
from typing import Dict, Protocol, Union  # noqa: TYP001

import pandas as pd
from plotly.graph_objs import Figure

from nannyml.exceptions import InvalidArgumentsException


class Result(Protocol):
    """the data that was calculated or estimated"""

    data: pd.DataFrame

    """all available plots"""
    plots: Dict[str, Figure]

    """name of the calculator that created it"""
    calculator_name: str


class Calculator(Protocol):
    def fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the calculator on reference data."""

    def calculate(self, data: pd.DataFrame, *args, **kwargs):
        """Perform a calculation based on analysis data."""


class Estimator(Protocol):
    def fit(self, reference_data: pd.DataFrame, *args, **kwargs):
        """Fits the estimator on reference data."""

    def estimate(self, data: pd.DataFrame, *args, **kwargs) -> Result:
        """Perform an estimation based on analysis data."""


ModelOutputsType = Union[str, Dict[str, str]]


class UseCase(str, Enum):
    CLASSIFICATION_BINARY = 'classification_binary'
    CLASSIFICATION_MULTICLASS = 'classification_multiclass'
    REGRESSION = 'regression'


def derive_use_case(y_pred_proba: ModelOutputsType) -> UseCase:
    if isinstance(y_pred_proba, Dict):
        return UseCase.CLASSIFICATION_MULTICLASS
    elif isinstance(y_pred_proba, str):
        return UseCase.CLASSIFICATION_BINARY
    else:
        raise InvalidArgumentsException(
            "parameter 'y_pred_proba' is of type '{type(y_pred_proba)}' "
            "but should be of type 'Union[str, Dict[str, str].'"
        )