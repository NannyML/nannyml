.. _multiclass-performance-calculation:

================================================================
Monitoring Realized Performance for Multiclass Classification
================================================================

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.



Just The Code
==============

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cells: 1 3 4 6 8



Walkthrough
=============


For simplicity the guide is based on a synthetic dataset where the monitored model predicts
which type of credit card product new customers should be assigned to. You can :ref:`learn more about this dataset<dataset-synthetic-multiclass>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to calculate performance, so the first thing we need to in this case is set up the right data in the right dataframes.  The analysis target values are joined on the analysis frame by the ``identifier`` column.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cell: 2

Next a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using a list of metrics to calculate (or just one metric), the data columns required for these metrics, and an optional :ref:`chunking<chunking>` specification.

The list of metrics specifies which performance metrics of the monitored model will be calculated.
The following metrics are currently supported:

- ``roc_auc`` - one-vs-the-rest, macro-averaged
- ``f1`` - macro-averaged
- ``precision`` - macro-averaged
- ``recall`` - macro-averaged
- ``specificity`` - macro-averaged
- ``accuracy``

For more information on metrics, check the :mod:`~nannyml.performance_calculation.metrics` module.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cells: 3


The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available with the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.calculate` method.
NannyML can output a dataframe that contains all the results of the analysis data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cell: 5

There results from the reference data are also available.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cell: 7

Apart from chunking and chunk and period-related columns, the results data have the a set of columns for each
calculated metric. When taking ``roc_auc`` as an example:

 - ``targets_missing_rate`` - The fraction of missing target data.
 - ``<metric>`` - The value of the metric for a specific chunk.
 - ``<metric>_lower_threshold>`` and ``<metric>_upper_threshold>`` - Lower and upper thresholds for performance metric.
   Crossing them will raise an alert that there is a significant metric change.
   The thresholds are calculated based on the realized performance of chunks in the ``reference`` period.
   The thresholds are 3 standard deviations away from the mean performance calculated on ``reference`` chunks.
   They are calculated during ``fit`` phase.
 - ``<metric>_alert`` - A flag indicating potentially significant performance change. ``True`` if realized performance
   crosses upper or lower threshold.
 - ``<metric>_sampling_error`` - Estimated :term:`Sampling Error` for the relevant metric.

The results can be plotted for visual inspection:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Multiclass Classification.ipynb
    :cells: 8


.. image:: /_static/tutorials/performance_calculation/multiclass/tutorial-performance-calculation-multiclass-f1.svg

.. image:: /_static/tutorials/performance_calculation/multiclass/tutorial-performance-calculation-multiclass-roc_auc.svg


Insights
========

After reviewing the performance calculation results, we should be able to clearly see how the model is performing against
the targets, according to whatever metrics we wish to track.


What Next
=========

If we decide further investigation is needed, the :ref:`Data Drift<data-drift>` functionality can help us to see
what feature changes may be contributing to any performance changes.

It is also wise to check whether the model's performance is satisfactory
according to business requirements. This is an ad-hoc investigation that is not covered by NannyML.
