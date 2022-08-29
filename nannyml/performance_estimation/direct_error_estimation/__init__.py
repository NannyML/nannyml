#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  #
#  License: Apache Software License 2.0

DEFAULT_METRICS = [
    'mae',
    'mape',
    'mse',
    'rmse',
]  # 'msle', 'rmsle']

from .dee import DEE  # noqa: E402
from .result import Result  # noqa: E402
