.. _drift_detection_for_regression_model_outputs:

=======================================================
Drift Detection for Regression Model Outputs
=======================================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of the model outputs tells us the model's evaluation of the expected
outcome across the model's population.
If the model's population changes, then the outcome will be different.
The difference in actions is very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.


Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 1 3 4 6 8 10


Walkthrough
-----------

NannyML detects data drift for :term:`Model Outputs` using the
:ref:`Univariate Drift Detection methodology<univariate_drift_detection_walkthrough>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset.
Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference
and analysis dataframes. This synthetic data is for a regression model predicting used car prices. You can find more
details about it :ref:`here<dataset-synthetic-regression>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cell: 2

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator`
class implements the functionality needed for drift detection in model outputs. First, the class is instantiated with appropriate parameters.
To check the model outputs for data drift, we need to pass the name of the predictions column, the name of the timestamp column and the
type of the machine learning problem our model is addressing. In our case the problem type is regression.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.fit` method
is called on the reference data, so that the data baseline can be established.
Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.calculate` method
calculates the drift results on the data provided. An example using it can be seen below.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 3

We can then display the results in a table.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cell: 5

The drift results from the reference data are accessible though the ``previous_reference_results`` property of the drift calculator who is also accessible from the results object.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cell: 7

NannyML can show the statistical properties of the drift in model outputs as a plot.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 8

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/regression/drift_guide_prediction_drift.svg


NannyML can also visualise how the distributions of the model predictions evolved over time.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Regression.ipynb
    :cells: 10

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/regression/drift_guide_prediction_distribution.svg


Insights
--------

We can see that in the middle of the analysis period the model output distribution has changed significantly and
there is a good possiblity that the performance of our model has been impacted.

What Next
---------

If required, the :ref:`performance estimation<regression-performance-estimation>` functionality of NannyML can help
provide estimates of the impact of the observed changes to Model Outputs without having to wait for Model Targets to
become available.
