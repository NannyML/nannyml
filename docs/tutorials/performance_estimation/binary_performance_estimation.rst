.. _binary-performance-estimation:

================================================
Estimating Performance for Binary Classification
================================================

We currently support the following **standard** metrics for bianry classification performance estimation:

    * roc_auc
    * f1
    * precision
    * recall
    * specificity
    * accuracy

For more information about estimating these metrics, refer to the :ref:`standard-metric-estimation` section.

We also support the following metrics for binary classification performance estimation:

    * confusion_matrix
    * business_value

For more information about estimating these metrics, refer to the :ref:`confusion-matrix-estimation` and :ref:`business-value-estimation` sections.

.. toctree::
   :maxdepth: 2

   binary_performance_estimation/standard_metric_estimation
   binary_performance_estimation/confusion_matrix_estimation
   binary_performance_estimation/business_value_estimation
