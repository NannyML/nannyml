.. _multiclass-performance-estimation:

====================================================
Estimating Performance for Multiclass Classification
====================================================

We currently support the following **standard** metrics for multiclass classification performance estimation:

    * **roc_auc** - one-vs-the-rest, macro-averaged
    * **f1** - macro-averaged
    * **precision** - macro-averaged
    * **recall** - macro-averaged
    * **specificity** - macro-averaged
    * **accuracy**

For more information about estimating these metrics, refer to the :ref:`standard-metric-estimation` section.

We also support the following *complex* metrics for multiclass classification performance estimation:

    * **confusion_matrix**

For more information about estimating the confusion matrix for multiclass problems,
refer to the :ref:`multiclass-confusion-matrix-estimation` section.

.. toctree::
   :maxdepth: 2

   multiclass_performance_estimation/standard_metric_estimation
   multiclass_performance_estimation/confusion_matrix_estimation
