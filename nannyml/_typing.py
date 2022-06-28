#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0
from typing import Dict, Protocol  # noqa: TYP001

import pandas as pd
from plotly.graph_objs import Figure


class Result(Protocol):

    """the data that was calculated or estimated"""

    data: pd.DataFrame

    """all available plots"""
    plots: Dict[str, Figure]

    """name of the calculator that created it"""
    calculator_name: str


class Calculator(Protocol):
    def fit(self, reference_data: pd.DataFrame):
        """Fits the calculator on reference data."""

    def calculate(self, analysis_data: pd.DataFrame):
        """Perform a calculation based on analysis data."""


class Estimator(Protocol):
    def fit(self, reference_data: pd.DataFrame):
        """Fits the estimator on reference data."""

    def estimate(self, analysis_data: pd.DataFrame) -> Result:
        """Perform an estimation based on analysis data."""
