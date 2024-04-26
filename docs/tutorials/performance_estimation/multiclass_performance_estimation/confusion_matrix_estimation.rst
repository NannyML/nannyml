.. _multiclass-confusion-matrix-estimation:

==================================================================
Estimating Confusion Matrix Elements for Multiclass Classification
==================================================================

This tutorial explains how to use NannyML to estimate the :term:`confusion matrix<Confusion Matrix>` for multiclass classification
models in the absence of target data. To find out how CBPE estimates performance, read the :ref:`explanation of Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.

.. _confusion-matrix-estimation-multiclass-just-the-code:

Just The Code
----------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cells: 1 3 4 5 7


Walkthrough
--------------

For simplicity this guide is based on a synthetic dataset where the monitored model predicts
which type of credit card product new customers should be assigned to.
Check out :ref:`Credit Card Dataset<dataset-synthetic-multiclass>` to learn more about this dataset.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual monitored, provided as the monitored dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

We start by loading the dataset we'll be using:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cell: 2

Next we create the Confidence-based Performance Estimation
(:class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`)
estimator. To initialize an estimator that estimates the **confusion_matrix**, we specify the following
parameters:

  - **y_pred_proba:** a dictionary that maps the class names to the
    name of the column in the reference data that contains the
    predicted probabilities for that class.
  - **y_pred:** the name of the column in the reference data that
    contains the predicted classes.
  - **y_true:** the name of the column in the reference data that
    contains the true classes.
  - **timestamp_column_name (Optional):** the name of the column in the reference data that
    contains timestamps.
  - **metrics:** a list of metrics to estimate. In this example we
    will estimate the ``confusion_matrix`` metric.
  - **chunk_size (Optional):** the number of observations in each chunk of data
    used to estimate performance. For more information about
    :term:`chunking<Data Chunk>` configurations check out the :ref:`chunking tutorial<chunking>`.
  - **problem_type:** the type of problem being monitored. In this example we
    will monitor a multiclass classification problem.
  - **normalize_confusion_matrix (Optional):** how to normalize the confusion matrix.
    The normalization options are:

    * **None** : returns counts for each cell
    * **"true"** : normalize over the true class of observations.
    * **"pred"** : normalize over the predicted class of observations
    * **"all"** : normalize over all observations

  - **thresholds (Optional):** the thresholds used to calculate the alert flag. For more information about
    thresholds, check out the :ref:`thresholds tutorial<thresholds>`.

.. note::
    Since we are estimating the confusion matrix, the count values
    in each cell of the confusion matrix are estimates. We normalize the
    estimates just as if they were true counts. This means that when we
    normalize over the true class, the estimates in each row will sum to 1.
    When we normalize over the predicted class, the estimates in each
    column will sum to 1. When we normalize over all observations, the
    estimates in the entire matrix will sum to 1.


.. nbimport::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cells: 3

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
estimator is then fitted using the
:meth:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE.fit` method on the ``reference`` data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cells: 4

The fitted ``estimator`` can be used to estimate performance on other data, for which performance cannot be calculated.
Typically, this would be used on the latest production data where target is missing. In our example this is
the ``monitored_df`` data.

NannyML can then output a dataframe that contains all the results. Let's have a look at the results for monitored period
only.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cells: 5

.. nbtable::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cell: 6

Apart from chunk-related data, the results data have the following columns for each metric
that was estimated:

 - **value** - the estimate of a metric for a specific chunk.
 - **sampling_error** - the estimate of the :term:`Sampling Error`.
 - **realized** - when **target** values are available for a chunk, the realized performance metric will also
   be calculated and included within the results.
 - **upper_confidence_boundary** and **lower_confidence_boundary** - These values show the :term:`confidence band<Confidence Band>` of the relevant metric
   and are equal to estimated value +/- 3 times the estimated :term:`sampling error<Sampling Error>`.
 - **upper_threshold** and **lower_threshold** - crossing these thresholds will raise an alert on significant
   performance change. The thresholds are calculated based on the actual performance of the monitored model on chunks in
   the **reference** partition. The thresholds are 3 standard deviations away from the mean performance calculated on
   chunks.
   The thresholds are calculated during **fit** phase.
 - **alert** - flag indicating potentially significant performance change. ``True`` if estimated performance crosses
   upper or lower threshold.

These results can be also plotted. Our plot contains several key elements.

* *The purple step plot* shows the estimated performance in each chunk of the monitored period. Thick squared point
  markers indicate the middle of these chunks.

* *The low-saturated purple area* around the estimated performance in the monitored period corresponds to the :term:`confidence band<Confidence Band>` which is
  calculated as the estimated performance +/- 3 times the estimated :term:`Sampling Error`.

* *The gray vertical line* splits the reference and monitored periods.

* *The red horizontal dashed lines* show upper and lower thresholds for alerting purposes.

* *The red diamond-shaped point markers* in the middle of a chunk indicate that an alert has been raised. Alerts are caused by the estimated performance crossing the upper or lower threshold.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Estimating Confusion Matrix - Multiclass Classification.ipynb
    :cells: 7

.. image:: ../../../_static/tutorials/performance_estimation/multiclass/tutorial-confusion-matrix-estimation-multiclass-monitored-with-ref.svg

Additional information such as the chunk index range and chunk date range (if timestamps were provided) is shown in the hover for each chunk (these are
interactive plots, though only static views are included here).

Insights
--------

After reviewing the performance estimation results, we should be able to see any indications of performance change that
NannyML has detected based upon the model's inputs and outputs alone.


What's next
-----------

The :ref:`Data Drift<data-drift>` functionality can help us to understand whether data drift is causing the performance problem.
When the target values become available we can use
:ref:`realized performance calculation<multiclass-confusion-matrix-calculation>` to
:ref:`compare realized and estimated confusion matrix results<compare_estimated_and_realized_performance>`.
