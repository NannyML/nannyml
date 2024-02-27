.. _multivariate_drift_detection:

============================
Multivariate Drift Detection
============================

Multivariate data drift detection compliments :ref:`univariate data drift detection methods<univariate_drift_detection>`.
It provides one summary number reducing the risk of false alerts, and detects more subtle changes
in the data structure that cannot be detected with univariate approaches. The trade off is that
multivariate drift results are less explainable compared to univariate drift results.

.. toctree::
   :maxdepth: 2

   multivariate_drift_detection/pca
   multivariate_drift_detection/dc
