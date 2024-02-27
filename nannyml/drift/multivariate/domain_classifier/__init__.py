#  Author:   Niels Nuyttens  <niels@nannyml.com>
#  Author:   Nikolaos Perrakis  <nikos@nannyml.com>
#
#  License: Apache Software License 2.0

"""Package containing the Classifier For Drift Detection Calculator implementation.

The data reconstruction error drift detection method is used to detect multivariate drift, i.e. it works on a subset or
all features of a model.

It will perform a dimensionality reduction, currently only using Principal Component Analysis (PCA). In doing so, the
dimensionality reducer learns about the internal structure of the data. When new, unseen data arrives, applying the
inverse transformation with the reducer renders a reconstruction according to the original reference data set.

By evaluating the euclidian distance between the unseen data and its reconstruction using the reducer, i.e. the
reconstruction error, we get a notion of how the unseen data differs or drifts from the reference data set.

For more information, check out the `tutorial`_ or the `deep dive`_.

.. _tutorial:
    https://nannyml.readthedocs.io/en/stable/tutorials/detecting_data_drift/multivariate_drift_detection.html

.. _deep dive:
    https://nannyml.readthedocs.io/en/stable/how_it_works/data_reconstruction.html#id1

"""

from .calculator import DomainClassifierCalculator
from .result import Result
