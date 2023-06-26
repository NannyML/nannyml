.. _multivariate_drift_detection:

============================
Multivariate Drift Detection
============================

Why Perform Multivariate Drift Detection
----------------------------------------

Multivariate data drift detection addresses the shortcomings of :ref:`univariate data detection methods<univariate_drift_detection>`.
It provides one summary number reducing the risk of false alerts, and detects more subtle changes
in the data structure that cannot be detected with univariate approaches.

Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 1 3 4 6 8

.. admonition:: **Advanced configuration**
    :class: hint

    - To learn how :class:`~nannyml.chunk.Chunk` works and to set up custom chunkings check out the :ref:`chunking tutorial <chunking>`
    - To learn how :class:`~nannyml.thresholds.ConstantThreshold` works and to set up custom threshold check out the :ref:`thresholds tutorial <thresholds>`

Walkthrough
-------------------------------------------

NannyML uses Data Reconstruction with PCA to detect such changes. For a detailed explanation of
the method see :ref:`Data Reconstruction with PCA Deep Dive<data-reconstruction-pca>`.

The method returns a single number, measuring the :term:`Reconstruction Error`. The changes in this value
reflect a change in the structure of the model inputs.

NannyML calculates the reconstruction error over time for the monitored model, and raises an alert if the
values get outside a range defined by the variance in the reference :ref:`data period<data-drift-periods>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package and setting it up as our reference and analysis dataframes. This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cell: 2

The :class:`~nannyml.drift.multivariate.data_reconstruction.calculator.DataReconstructionDriftCalculator`
module implements this functionality.  We need to instantiate it with appropriate parameters - the column names of the features we want to run drift detection on,
and the timestamp column name. The features can be passed in as a simple list of strings. Alternatively, we can create a list by excluding the columns in the dataframe that are not features,
and pass them into the argument.

Next, the :meth:`~nannyml.base.AbstractCalculator.fit` method needs to be called on the reference data, which the results will be based on.
Then the
:meth:`~nannyml.base.AbstractCalculator.calculate` method will
calculate the multivariate drift results on the provided data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 3

Any missing values in our data need to be imputed. The default :term:`Imputation` implemented by NannyML imputes
the most frequent value for categorical features and the mean for continuous features. These defaults can be
overridden with an instance of `SimpleImputer`_ class, in which case NannyML will perform the imputation as instructed.

An example of where custom imputation strategies are used can be seen below.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 10

Because our synthetic dataset does not have missing values, the results are the same in both cases.
We can see these results of the data provided to the
:meth:`~nannyml.base.AbstractCalculator.calculate`
method as a dataframe.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cell: 5

The drift results from the reference data are accessible from the properties of the results object:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cell: 7


NannyML can also visualize the multivariate drift results in a plot. Our plot contains several key elements.

* The purple step plot shows the reconstruction error in each chunk of the analysis period. Thick squared point
  markers indicate the middle of these chunks.

* The low-saturated purple area around the reconstruction error indicates the :ref:`sampling error<estimation_of_standard_error>`.

* The red horizontal dashed lines show upper and lower thresholds for alerting purposes.

* If the reconstruction error crosses the upper or lower threshold an alert is raised which is indicated with a red,
  low-saturated background across the whole width of the relevant chunk. A red, diamond-shaped point marker additionally indicates this in the middle of the chunk.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Multivariate.ipynb
    :cells: 8

.. image:: /_static/tutorials/detecting_data_drift/multivariate_drift_detection/pca-reconstruction-error.svg

The multivariate drift results provide a concise summary of where data drift
is happening in our input data.

.. _SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html


Insights
--------

Using this method of detecting drift, we can identify changes that we may not have seen using solely univariate methods.

What Next
---------

After reviewing the results, we want to look at the :ref:`drift results of individual features<univariate_drift_detection>`
to see what changed in the model's features individually.

The :ref:`Performance Estimation<performance-estimation>` functionality can be used to
estimate the impact of the observed changes.

For more information on how multivariate drift detection works, the
:ref:`Data Reconstruction with PCA<data-reconstruction-pca>` explanation page gives more details.
