#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""Univariate drift detection methods, considering a single column at a time.

Some methods are applicable exclusively to continuous or categorical columns, others are applicable to both.

This package currently holds an implementation for the following univariate drift detection methods:

- Kolmogorov-Smirnov statistic (continuous)
- Wasserstein distance (continuous)
- Chi-squared statistic (categorical)
- L-infinity distance (categorical)
- Jensen-Shannon distance
- Hellinger distance

For more information, check out the `tutorial`_ or the `deep dive`_.

For help selecting the correct univariate drift detection method for your use case, check the `method selection guide`_.

.. _tutorial:
    https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/univariate_drift_detection.html

.. _deep dive:
    https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_detection.html

.. _method selection guide:
    https://nannyml.readthedocs.io/en/stable/how_it_works/univariate_drift_comparison.html

"""
from .calculator import UnivariateDriftCalculator
from .methods import FeatureType, Method, MethodFactory
from .result import Result
