#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

"""Module containing the Direct Error Estimation implementation."""

DEFAULT_METRICS = ['mae', 'mape', 'mse', 'rmse', 'msle', 'rmsle']

from .dle import DLE  # noqa: E402
from .result import Result  # noqa: E402
