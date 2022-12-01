#  Author:   Niels Nuyttens  <niels@nannyml.com>
#
#  License: Apache Software License 2.0

"""NannyML drift calculation module.

This module currently contains the following implementations of drift calculation:

- Statistical drift calculation: Calculating drift using Kolmogorov-Smirnov test for continuous features
  and Chi-squared test for categorical features.
- Reconstruction error drift calculation: Detect drift by performing dimensionality reduction on the model
  inputs and then applying the inverse transofrmation on the latent (reduced) space.

"""
from .multivariate.data_reconstruction import DataReconstructionDriftCalculator
from .ranker import AlertCountRanker, CorrelationRanker
from .univariate import FeatureType, Method, MethodFactory, UnivariateDriftCalculator
