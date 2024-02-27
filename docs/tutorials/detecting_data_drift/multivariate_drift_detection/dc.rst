.. _multivariate_drift_detection_dc:

=================
Domain Classifier
=================

The second multivariate drift detection method of NannyML is Domain Classifier.
It provides a measure of how easy it is to discriminate the reference data from the examined chunk data.
You can read more about on the :ref:`How it works: Domain Classifier<how-multiv-drift-dc>` section.
When there is no data drift the datasets can't discerned and we get a value of 0.5.
The more drift there is, the higher the returned measure will be, up to a value of 1.

Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cells: 1 3 4 6 8

.. admonition:: **Advanced configuration**
    :class: hint

    - To learn how :class:`~nannyml.chunk.Chunk` works and to set up custom chunkings check out
      the :ref:`chunking tutorial <chunking>`.
    - To learn how :class:`~nannyml.thresholds.ConstantThreshold` works and to set up custom threshold
      check out the :ref:`thresholds tutorial <thresholds>`.

Walkthrough
-----------

The method returns a single number, measuring the discrimination capability of the discriminator.
Any increase in the discrimination value above 0.5 reflects a change in the structure of the model inputs.

NannyML calculates the discrimination value for the monitored model's inputs, and raises an alert if the
values get outside the  pre-defined range of ``[0.45, 0.65]``. If needed this range can be adjusted by specifying
a threshold strategy more appropriate for the user's data.

In order to monitor a model, NannyML needs to learn about it from a reference dataset.
Then it can monitor the data subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package set it up as our reference and analysis dataframes.
This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cell: 2

The :class:`~nannyml.drift.multivariate.domain_classifier.calculator.DomainClassifierCalculator`
module implements this functionality. We need to instantiate it with appropriate parameters:

- **feature_column_names:** A list with the column names of the features we want to run drift detection on.
- **treat_as_categorical (Optional):** A list containing the names of features in the provided data set that
  should be treated as categorical. Needs not be exhaustive.
- **timestamp_column_name (Optional):** The name of the column in the reference data that
  contains timestamps.
- **chunk_size (Optional):** The number of observations in each chunk of data
  used. Only one chunking argument needs to be provided. For more information about
  :term:`chunking<Data Chunk>` configurations check out the :ref:`chunking tutorial<chunking>`.
- **chunk_number (Optional):** The number of chunks to be created out of data provided for each
  :ref:`period<data-drift-periods>`.
- **chunk_period (Optional):** The time period based on which we aggregate the provided data in
  order to create chunks.
- **chunker (Optional):** A NannyML :class:`~nannyml.chunk.Chunker` object that will handle the aggregation
  provided data in order to create chunks.
- **cv_folds_num (Optional):** Number of cross-validation folds to use when calculating DC discrimination value.
- **hyperparameters (Optional):** A dictionary used to provide your own custom hyperparameters when training the
  discrimination model. Check out the available hyperparameter options in the `LightGBM docs`_.
- **tune_hyperparameters (Optional):** A boolean controlling whether hypertuning should be performed on the internal
  regressor models whilst fitting on reference data.
- **hyperparameter_tuning_config (Optional):** A dictionary that allows you to provide a custom hyperparameter
  tuning configuration when `tune_hyperparameters` has been set to `True`. Available options are available
  in the `AutoML FLAML documentation`_.
- **threshold (Optional):** The threshold strategy used to calculate the alert threshold limits.
  For more information about thresholds, check out the :ref:`thresholds tutorial<thresholds>`.

Next, the :meth:`~nannyml.base.AbstractCalculator.fit` method needs to be called on the reference data,
which the results will be based on. Then the
:meth:`~nannyml.base.AbstractCalculator.calculate` method will
calculate the multivariate drift results on the provided data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cells: 3

We can see these results of the data provided to the
:meth:`~nannyml.base.AbstractCalculator.calculate`
method as a dataframe.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cell: 5

The drift results from the reference data are accessible from the properties of the results object:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cell: 7


NannyML can also visualize the multivariate drift results in a plot. Our plot contains several key elements.

* The purple step plot shows the reconstruction error in each chunk of the analysis period. Thick squared point
  markers indicate the middle of these chunks.
* The red horizontal dashed lines show upper and lower thresholds for alerting purposes.
* If discrimination value crosses the upper or lower threshold an alert is raised.
  A red, diamond-shaped point marker additionally indicates this in the middle of the chunk.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate - Domain Classifier.ipynb
    :cells: 8

.. image:: /_static/tutorials/detecting_data_drift/multivariate_drift_detection/classifier-for-drift-detection.svg

The multivariate drift results provide a concise summary of where data drift
is happening in our input data.

Insights
--------

Using this method of detecting drift, we can identify changes that we may not have seen using solely univariate methods.

What Next
---------

After reviewing the results, we want to look at the :ref:`drift results of individual features<univariate_drift_detection>`
to see what changed in the model's features individually.

The :ref:`Performance Estimation<performance-estimation>` functionality can be used to
estimate the impact of the observed changes.


.. _`AutoML FLAML documentation`: https://microsoft.github.io/FLAML/docs/reference/automl/automl
.. _`LightGBM docs`: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
