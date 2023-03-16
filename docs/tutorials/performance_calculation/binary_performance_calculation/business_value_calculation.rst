.. _business-value-calculation:

========================================================================================
Calculating Business Value for Binary Classification
========================================================================================
This tutorial explains how to use NannyML to calculate business value for binary classification
models. 

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.

.. _business-value-calculation-binary-just-the-code:

Just The Code
----------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
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
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
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
      we will specifically focus on the ``business_value``
      metric, so our list of metrics will only contain this metric.
    * **The business value matrix:** This is required when calculating the ``business_value`` metric and is a 2x2
      matrix that specifies the value of each cell in the confusion
      matrix. The top left cell is the value of a true negative, the top right cell is the value of a false positive,
      the bottom left cell is the value of a false negative, and the bottom right cell is the value of a true positive.
      For example, if in your use case a false positive costs $10, the top right cell of your value matrix would be -10.
      If a true positive is worth $100, the bottom right cell of your value matrix would be 100, and so on.
    * **An optional** ``normalize_business_value`` **specification:** if this is set to ``"per_prediction"``, the business value
      will be normalized over the number of observations in a given chunk. If this is set to ``None`` (or not specified),
      the business value will be the total value per chunk.
    * **An optional chunking specification:** for more information about :term:`chunking<Data Chunk>`
      you can check the :ref:`chunking page<chunking>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cells: 3

In our example, true negatives have a value of 5,
false positives have a value of -10, false negatives have a value of -50,
and true positives have a value of 50.

.. note::
  The list of metrics specifies which performance metrics of the monitored model will be caclulated.
  This tutorial is specific to the ``business_value`` metric, but you can find more information about
  other standard metrics such as ``roc_auc``, ``f1``, ``precision``, ``recall``, ``specificity``,
  and ``accuracy`` in the
  :ref:`calculation of standard performance metrics tutorial<standard-metric-calculation>`. Additionally,
  you can find more information about the ``confusion_matrix`` metric in the
  :ref:`confusion matrix calculation tutorial<confusion-matrix-calculation>`.

The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cells: 4

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available with the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.calculate` method.
NannyML can output a dataframe that contains all the results of the analysis data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cells: 5

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cell: 6

The results from the reference data are also available.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cells: 7

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
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
    :path: ./example_notebooks/Tutorial - Calculating Business Value - Binary Classification.ipynb
    :cells: 9

.. image:: /_static/tutorials/performance_calculation/binary/tutorial-business-value-calculation-binary-car-loan-analysis.svg


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
