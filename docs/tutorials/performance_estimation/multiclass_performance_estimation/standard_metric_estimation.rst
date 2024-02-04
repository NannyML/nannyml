.. _multiclass_standard-metric-estimation:

====================================================
Estimating Performance for Multiclass Classification
====================================================

The tutorials linked below show how to use NannyML to estimate the standard performance
metrics of multiclass classification models in the absence of target data.
NannyML provides two algorithms to do this:

- Confidence Based Performance Estimator, through the
  :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator
- Importance Weighting, through the :class:`~nannyml.performance_estimation.importance_weighting.iw.IW` estimator

You can read more about how those algorithms work in the
:ref:`Performance Estimation, How it Works<performance-estimation-deep-dive>`
page. The following standard performance metrics are available for multiclass classification:

  - ``roc_auc`` - one-vs-the-rest, macro-averaged
  - ``f1`` - macro-averaged
  - ``precision`` - macro-averaged
  - ``recall`` - macro-averaged
  - ``specificity`` - macro-averaged
  - ``accuracy``

The tutorials below show how you can use CBPE and IW to esti:


.. toctree::
   :maxdepth: 2

   standard_metric_estimation/cbpe
   standard_metric_estimation/iw
