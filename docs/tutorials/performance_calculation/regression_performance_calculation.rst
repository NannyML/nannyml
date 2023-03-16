.. _regression-performance-calculation:

==============================================
Monitoring Realized Performance for Regression
==============================================

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.


Just The Code
=============

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cells: 1 3 4 6 8

.. admonition:: **Advanced configuration**
    :class: hint

    - Set up custom chunking [:ref:`tutorial <chunking>`] [`API reference <../../nannyml/nannyml.chunk.html>`__]
    - Set up custom thresholds [:ref:`tutorial <thresholds>`] [`API reference <../../nannyml/nannyml.thresholds.html>`__]

Walkthrough
===========


For simplicity the guide is based on a synthetic dataset where the monitored model predicts the selling price of a used car.
You can :ref:`learn more about this dataset<dataset-synthetic-regression>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation<performance-estimation>`.
But as it is required to calculate performance, the first thing to do in this case is to join the analysis target values with the analysis data.


.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cell: 2

Next a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using a list of metrics to calculate (or just one metric),
the data columns required for these metrics, an optional :ref:`chunking<chunking>` specification and the type of machine learning problem being addressed.

The list of metrics specifies which performance metrics of the monitored model will be calculated.
The following metrics are currently supported:

- ``mae`` - mean absolute error
- ``mape`` - mean absolute percentage error
- ``mse`` - mean squared error
- ``rmse`` - root mean squared error
- ``msle`` - mean squared logarithmic error
- ``rmsle`` - root mean squared logarithmic error

For more information on metrics, check the :mod:`~nannyml.performance_calculation.metrics` module.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cells: 3

The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available with the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.calculate` method.
NannyML can output a dataframe that contains all the results of the analysis data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cell: 5

There results from the reference data are also available.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cell: 7

Apart from chunking and chunk and period-related columns, the results data have a set of columns for each
calculated metric.

 - ``targets_missing_rate`` - The fraction of missing target data.
 - ``value`` - the realized metric value for a specific chunk.
 - ``sampling_error`` - the estimate of the :term:`Sampling Error`.
 - ``upper_threshold`` and ``lower_threshold`` - crossing these thresholds will raise an alert on significant
   performance change. The thresholds are calculated based on the actual performance of the monitored model on chunks in
   the ``reference`` partition. The thresholds are 3 standard deviations away from the mean performance calculated on
   chunks.
   They are calculated during ``fit`` phase.
 - ``alert`` - flag indicating potentially significant performance change. ``True`` if estimated performance crosses
   upper or lower threshold.

The results can be plotted for visual inspection:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Realized Performance - Regression.ipynb
    :cells: 8

.. image:: /_static/tutorials/performance_calculation/regression/tutorial-performance-calculation-regression.svg


Insights
========

From looking at the RMSE and RMSLE performance results we can observe an interesting effect. We know that RMSE penalizes
mispredictions symmetrically while RMSLE penalizes underprediction more than overprediction. Hence while our model has become a little
bit more accurate according to RMSE, the increase in RMSLE tells us that our model is now underpredicting more than it was before!


What Next
=========

If we decide further investigation is needed, the :ref:`Data Drift<data-drift>` functionality can help us to see
what feature changes may be contributing to any performance changes.

It is also wise to check whether the model's performance is satisfactory
according to business requirements. This is an ad-hoc investigation that is not covered by NannyML.
