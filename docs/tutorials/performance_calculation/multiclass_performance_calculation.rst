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
    * **average_precision**

For more information about estimating these metrics, refer to the :ref:`multiclass-standard-metric-calculation` section.

We also support the following *complex* metric for multiclass classification performance calculation:

    * **confusion_matrix**
    * **business_value:** a metric that combines the components of the confusion matrix using
      user-specified weights for each element, allowing for a connection between model performance and
      business results.

For more information about calculating these metrics, refer to the :ref:`multiclass-confusion-matrix-calculation`
and :ref:`multiclass-business-value-calculation` sections.

.. toctree::
   :maxdepth: 2

   multiclass_performance_calculation/standard_metric_calculation
   multiclass_performance_calculation/confusion_matrix_calculation
   multiclass_performance_calculation/business_value_calculation
