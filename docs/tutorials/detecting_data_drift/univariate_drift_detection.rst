.. _univariate_drift_detection:

==========================
Univariate Drift Detection
==========================

Univariate Drift Detection looks at each feature individually and checks whether its
distribution has changed. It's a simple, fully explainable form of data drift detection
and is the most straightforward to understand and communicate.


There are two types of Univariate Drift Detection in NannyML.
The Statistical Drift Detection that uses statistical two sample tests
to look for drift and the Distance Drift Detection that uses distance measures
to look for drift.


.. toctree::
   :maxdepth: 2

   univariate_drift_detection/univariate_distance_drift.rst
   univariate_drift_detection/univariate_statistical_drift

