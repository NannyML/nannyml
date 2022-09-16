.. _drift_detection_for_multiclass_classification_model_outputs:

============================================================
Drift Detection for Multiclass Classification Model Outputs
============================================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of the model outputs tells us the model's evaluation of how likely
the predicted outcome is to happen across the model's population.
If the model's population changes, then our populations' actions will be different.
The difference in actions is very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.

.. note::
    The following example uses :term:`timestamps<Timestamp>`.
    These are optional but have an impact on the way data is chunked and results are plotted.
    You can read more about them in the :ref:`data requirements<data_requirements_columns_timestamp>`.



Just The Code
------------------------------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 1 3 4 6 8 10 12


Walkthrough
------------------------------------------------

NannyML detects data drift for :term:`Model Outputs` using the
:ref:`Univariate Drift Detection methodology<univariate_drift_detection_walkthrough>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference and analysis dataframes. This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cell: 2

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator`
class implements the functionality needed for drift detection in model outputs. First, the class is instantiated with appropriate parameters.
To check the model outputs for data drift, we only need to pass in the column header of the outputs as `y_pred` and `y_pred_proba`.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.fit` method
is called on the reference data, so that the data baseline can be established.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.calculate` method
calculates the drift results on the data provided. An example using it can be seen below.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 3

We can then display the results in a table, or as plots.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cell: 5

NannyML can show the statistical properties of the drift in model scores as a plot.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 6

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-drift-prepaid_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-drift-upmarket_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-drift-highstreet_card.svg

NannyML can also visualise how the distributions of the model scores evolved over time.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 8

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-distribution-prepaid_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-distribution-upmarket_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-distribution-highstreet_card.svg

NannyML can show the statistical properties of the drift in the predicted labels as a plot.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 10

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-prediction-drift.svg

NannyML can also visualise how the distributions of the predicted labels evolved over time.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Outputs - Multiclass Classification.ipynb
    :cells: 12

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-prediction-distribution.svg



What Next
-----------------------

If required, the :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Outputs.
