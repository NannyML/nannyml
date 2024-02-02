.. _standard-metric-estimation:

=================================================================
Estimating Standard Performance Metrics for Binary Classification
=================================================================

This tutorial explains how to use NannyML to estimate the standard performance
metrics of binary classification
models in the absence of target data. NannyML provides two algorithms to do this:

- Confidence Based Performance Estimator, through the :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` calculator
- Importance Weighting, through the :class:`~nannyml.performance_estimation.importance_weighting.iw.IW` calculator

You can read more about how those algorithms work in the :ref:`Performance Estimation, How it Works<performance-estimation-deep-dive>`
page. The tutorials below show how you can use them:


.. toctree::
   :maxdepth: 2

   standard_metric_estimation/cbpe
   standard_metric_estimation/iw
