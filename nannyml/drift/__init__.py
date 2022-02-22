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

from ._base import BaseDriftCalculator, DriftCalculator
from .ranking import AlertCountRanking, Ranking, rank_drifted_features
from .reconstruction_error_drift_calcutor import ReconstructionErrorDriftCalculator
from .statistical_drift_calculator import StatisticalDriftCalculator, calculate_statistical_drift
