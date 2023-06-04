#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import moment

def summary_stats_std_sampling_error_components(col: pd.Series) -> Tuple:
    """
    Calculate sampling error components for Summary Stats Standard Deviation
    using reference data.

    Parameters
    ----------
    col: pd.Series
        column for which we are calculating sampling error components

    Returns
    -------
    (std, moment_4th): Tuple[np.ndarray]
    """
    std = col.std()
    moment_4th = moment(col.to_numpy(), 4)
    return (std,moment_4th)


def summary_stats_std_sampling_error(sampling_error_components, col) -> float:
    """
    Calculate sampling error for Summary Stats Standard Deviation
    using reference data.

    Parameters
    ----------
    sampling_error_components : a set of parameters that were derived from reference data.
    col : the (analysis) column you want to calculate sampling error for.

    Returns
    -------
    sampling_error: float

    """
    _std = sampling_error_components[0]
    _mu4 = sampling_error_components[1]
    _size = col.shape[0]
    err_var = np.sqrt((1/_size)*(_mu4-((_size-3)*(_std**4)/(_size-1))))
    return  (1/(2*_std))*err_var
