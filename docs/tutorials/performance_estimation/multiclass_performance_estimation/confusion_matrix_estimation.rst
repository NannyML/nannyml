.. _multiclass-confusion-matrix-estimation:

==================================================================
Estimating Confusion Matrix Elements for Multiclass Classification
==================================================================

The tutorials linked below show how to use NannyML to estimate the :term:`confusion matrix<Confusion Matrix>`
for multiclass classification models in the absence of target data. NannyML provides two algorithms to do this:

- Confidence Based Performance Estimator, through the :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator
- Importance Weighting, through the :class:`~nannyml.performance_estimation.importance_weighting.iw.IW` estimator

You can read more about how those algorithms work in the :ref:`Performance Estimation, How it Works<performance-estimation-deep-dive>`
page. The tutorials below show how you can use them:


.. toctree::
   :maxdepth: 2

   confusion_matrix_estimation/cbpe
   confusion_matrix_estimation/iw
