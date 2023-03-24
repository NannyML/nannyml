.. _binary-performance-calculation:

================================================================
Monitoring Realized Performance for Binary Classification
================================================================

We currently support the following **standard** metrics for bianry classification performance calculation:

    * roc_auc
    * f1
    * precision
    * recall
    * specificity
    * accuracy

For more information about estimating these metrics, refer to the :ref:`standard-metric-calculation` section.

We also support the following metrics for binary classification performance calculation:

    * confusion_matrix
    * business_value

For more information about estimating these metrics, refer to the :ref:`confusion-matrix-calculation` and :ref:`business-value-calculation` sections.

.. toctree::
   :maxdepth: 2

   binary_performance_calculation/standard_metric_calculation
   binary_performance_calculation/confusion_matrix_calculation
   binary_performance_calculation/business_value_calculation
