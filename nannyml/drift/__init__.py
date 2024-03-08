#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""NannyML drift detection module.

This module contains ways to detect both univariate (within a single continuous or categorical column)
and multivariate (across multiple columns) drift.

The univariate drift detection methods include:

- Kolmogorov-Smirnov statistic (continuous)
- Wasserstein distance (continuous)
- Chi-squared statistic (categorical)
- L-infinity distance (categorical)
- Jensen-Shannon distance
- Hellinger distance

The multivariate drift detection methods include:

- Data reconstruction error: detects drift by performing dimensionality reduction on the model
  inputs using PCA and then applying the inverse transformation on the latent (reduced) space.
- Domain Classifer: detects drift by looking at how performance a domain classifier is at distinguising
  between the reference and the chunk datasets.
"""
from .multivariate.data_reconstruction import DataReconstructionDriftCalculator
from .multivariate.domain_classifier import DomainClassifierCalculator
from .ranker import AlertCountRanker, CorrelationRanker
from .univariate import FeatureType, Method, MethodFactory, UnivariateDriftCalculator
