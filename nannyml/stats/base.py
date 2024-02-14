#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for data quality calculations."""

from numpy import isfinite, issubdtype, number


def _add_alert_flag(row_result: dict) -> bool:
    flag = False
    # issubdtype checks for numeric type
    # # np.nan and np.inf pass
    # isfinite makes check go false for them
    if not ( issubdtype(type(row_result['value']), number) and isfinite(row_result['value'])):
        flag = True
    if row_result['upper_threshold'] is not None:
        if row_result['value'] > row_result['upper_threshold']:
            flag = True
    if row_result['lower_threshold'] is not None:
        if row_result['value'] < row_result['lower_threshold']:
            flag = True
    return flag
