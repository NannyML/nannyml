.. _multiclass-performance-calculation:

================================================================
Monitoring Realized Performance for Multiclass Classification
================================================================

We currently support the following **standard** metrics for multiclass classification performance calculation:

    * **roc_auc**
    * **f1**
    * **precision**
    * **recall**
    * **specificity**
    * **accuracy**

For more information about estimating these metrics, refer to the :ref:`multiclass-standard-metric-calculation` section.

We also support the following *complex* metric for multiclass classification performance calculation:

    * **confusion_matrix**

For more information about estimating this metrics, refer to the :ref:`multiclass-confusion-matrix-estimation` section.

.. toctree::
   :maxdepth: 2

   multiclass_performance_calculation/standard_metric_calculation
   multiclass_performance_calculation/confusion_matrix_calculation
