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
from .base import BaseDriftCalculator, DriftCalculator
from .model_inputs.multivariate.data_reconstruction import (
    DataReconstructionDriftCalculator,
    DataReconstructionDriftCalculatorResult,
)
from .model_inputs.univariate.statistical import UnivariateDriftResult, UnivariateStatisticalDriftCalculator
from .ranking import AlertCountRanking, Ranker, Ranking
from .target.target_distribution import TargetDistributionCalculator, TargetDistributionResult
