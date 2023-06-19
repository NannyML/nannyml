#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Module containing base classes for data quality calculations."""


def _add_alert_flag(row_result: dict) -> bool:
    flag = False
    if row_result['upper_threshold'] is not None:
        if row_result['value'] > row_result['upper_threshold']:
            flag = True
    if row_result['lower_threshold'] is not None:
        if row_result['value'] < row_result['lower_threshold']:
            flag = True
    return flag
