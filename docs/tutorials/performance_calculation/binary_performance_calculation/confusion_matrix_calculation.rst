.. _confusion-matrix-calculation:

========================================================================================
Calculating Confusion Matrix Elements for Binary Classification
========================================================================================

This tutorial explains how to use NannyML to calculate the :term:`confusion matrix<Confusion Matrix>` for binary classification
models.

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.

.. _confusion-matrix-calculation-binary-just-the-code:

Just The Code
----------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 1 3 4 5 7 9


Walkthrough
--------------

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model
predicts whether a customer will repay a loan to buy a car.
Check out :ref:`Car Loan Dataset<dataset-synthetic-binary-car-loan>` to learn more about this dataset.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation<performance-estimation>`. But it is required to calculate performance, so the first thing we need to in this case is set up the right data in the right dataframes.

The analysis target values are joined on the analysis frame by their index. Your dataset may already contain the **target** column, so you may skip this join.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cell: 2

Next a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using
the following:

  - **y_pred_proba:** the name of the column in the reference data that
    contains the predicted probabilities.
  - **y_pred:** the name of the column in the reference data that
    contains the predicted classes.
  - **y_true:** the name of the column in the reference data that
    contains the true classes.
  - **timestamp_column_name (Optional):** the name of the column in the reference data that
    contains timestamps.
  - **problem_type:** the type of problem being monitored. In this example we
    will monitor a binary classification problem.
  - **metrics:** a list of metrics to calculate. In this example we
    will calculate the ``confusion_matrix`` metric.
  - **normalize_confusion_matrix (Optional):**  how to normalize the confusion matrix.
    The normalization options are:

    * **None** : returns counts for each cell
    * **"true"** : normalize over the true class of observations.
    * **"pred"** : normalize over the predicted class of observations
    * **"all"** : normalize over all observations

  - **chunk_size (Optional):** the number of observations in each chunk of data
    used to calculate performance. For more information about
    :term:`chunking<Data Chunk>` other chunking options check out the :ref:`chunking tutorial<chunking>`.
  - **thresholds (Optional):** the thresholds used to calculate the alert flag. For more information about
    thresholds, check out the :ref:`thresholds tutorial<thresholds>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 3


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

Apart from chunk and period-related columns, the results data have a set of columns for each
calculated metric.

- **targets_missing_rate** - The fraction of missing target data.
- **value** - the realized metric value for a specific chunk.
- **sampling_error** - the estimate of the :term:`Sampling Error`.
- **upper_threshold** and **lower_threshold** - crossing these thresholds will raise an alert on significant
  performance change. The thresholds are calculated based on the actual performance of the monitored model on chunks in
  the **reference** partition. The thresholds are 3 standard deviations away from the mean performance calculated on
  chunks.
  They are calculated during **fit** phase.
- **alert** - flag indicating potentially significant performance change. ``True`` if estimated performance crosses
  upper or lower threshold.

The results can be plotted for visual inspection. Our plot contains several key elements.

* *The purple step plot* shows the performance in each chunk of the analysis period. Thick squared point
  markers indicate the middle of these chunks.

* *The blue step plot* shows the performance in each chunk of the reference period. Thick squared point markers indicate
  the middle of these chunks.

* *The gray vertical line* splits the reference and analysis periods.

* *The red horizontal dashed lines* show upper and lower thresholds for alerting purposes.

* *The red diamond-shaped point markers* in the middle of a chunk indicate that an alert has been raised. Alerts are caused by the performance crossing the upper or lower threshold.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Calculating Confusion Matrix - Binary Classification.ipynb
    :cells: 9

.. image:: /_static/tutorials/performance_calculation/binary/tutorial-confusion-matrix-calculation-binary-car-loan-analysis.svg

Additional information such as the chunk index range and chunk date range (if timestamps were provided) is shown in the hover for each chunk (these are
interactive plots, though only static views are included here).

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
