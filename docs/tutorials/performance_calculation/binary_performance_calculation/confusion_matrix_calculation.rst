.. _confusion-matrix-calculation:

========================================================================================
Calculating Confusion Matrix Elements for Binary Classification
========================================================================================

This tutorial explains how to use NannyML to calculate the confusion matrix for binary classification
models. 

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.

.. _confusion-matrix-calculation-binary-just-the-code:

Just The Code
----------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion MAtrix - Binary Classification.ipynb
    :cells: 1 3 4 5 7 9


Walkthrough
--------------

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model
predicts whether a customer will repay a loan to buy a car.
You can read more about this synthetic dataset :ref:`here<dataset-synthetic-binary-car-loan>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to calculate performance, so the first thing we need to in this case is set up the right data in the right dataframes.

The analysis target values are joined on the analysis frame by their index. Your dataset may already contain the ``target`` column, so you may skip this join.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cell: 2

Next a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using
the following:

    * **The names of the data columns required for these metrics:** for binary classification performance estimation,
      NannyML needs to know where to find the true class, the predicted class, and the predicted probability values 
      in the provided data. In our example, the true class values are found
      in the ``repaid`` column, the predicted values are found in the ``y_pred`` column, and the predicted probabilities
      are found in the ``y_pred_proba`` column.
    * **An optional timestamp specification:** timestamps are optional but have an impact on the way data is chunked
      and results are plotted. You can read more about them in the :ref:`data requirements page<data_requirements_columns_timestamp>`.
    * **A problem type specification:** this specifies whether the problem is a binary classification problem, a
      multiclass classification problem, or a regression problem. In this tutorial we will be using a binary classification problem.
    * **A list of metrics to calculate:** In this tutorial
      we will specifically focus on the ``confusion_matrix``
      metric, so our list of metrics will only contain this metric.
    * **An optional** ``normalize_confusion_matrix`` **specification:** This parameter specifies how the confusion matrix
      should be normalized. The options are:

        * ``None`` - no normalization, counts are returned (default)
        * ``'true'`` - normalize over the true class counts
        * ``'pred'`` - normalize over the predicted class counts
        * ``'all'`` - normalize over the total number of samples
    * **An optional chunking specification:** for more information about :term:`chunking<Data Chunk>`
      you can check the :ref:`chunking page<chunking>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 3

.. note::
  The list of metrics specifies which performance metrics of the monitored model will be calculated.
  This tutorial is specific to the ``confusion_matrix`` metric, but you can find more information about
  other standard metrics such as ``roc_auc``, ``f1``, ``precision``, ``recall``, ``specificity``,
  and ``accuracy`` in the
  :ref:`calculation of standard performance metrics tutorial<standard-metric-calculation>`. Additionally,
  you can find more information about the ``business_value`` metric in the
  :ref:`business value calculation tutorial<business-value-calculation>`.

The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 4

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available with the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.calculate` method.
NannyML can output a dataframe that contains all the results of the analysis data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 5

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cell: 6

The results from the reference data are also available.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 7

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cell: 8

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

The results can be plotted for visual inspection.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 9

.. image:: /_static/tutorials/performance_calculation/binary/tutorial-confusion-matrix-calculation-binary-car-loan-analysis.svg


Insights
--------

After reviewing the performance calculation results, we should be able to clearly see how the model is performing against
the targets, according to whatever metrics we wish to track.


What's Next
-----------

If we decide further investigation is needed, the :ref:`Data Drift<data-drift>` functionality can help us to see
what feature changes may be contributing to any performance changes.

It is also wise to check whether the model's performance is satisfactory
according to business requirements. This is an ad-hoc investigation that is not covered by NannyML.
