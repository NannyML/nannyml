#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

"""Module containing the Direct Error Estimation implementation."""

SUPPORTED_METRIC_VALUES = ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']

from .dle import DLE  # noqa: E402
from .result import Result  # noqa: E402
