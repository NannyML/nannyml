.. _data_requirements:

==================
Data requirements
==================

In this guide we'll present an overview of the different kinds of data NannyML requires to run its various features. The specifics for each feature are also covered in the various :ref:`Tutorials`, but an overview of all the different requirements is presented here for reference.

.. _data-drift-periods:

Data Periods
------------

NannyML works with two :term:`Data Periods<data period>`. The first one is called the :ref:`reference period<data-drift-periods-reference>`,
is represented by the **reference dataset**, and is used to establish the expectations of the model's performance.
The second is called the :ref:`analysis period<data-drift-periods-analysis>`, is represented by the **analysis
dataset** which is, as the name suggests, analyzed by NannyML to check whether model performance meets the
expectations set based on **reference dataset**.

.. _data-drift-periods-reference:

Reference Period
^^^^^^^^^^^^^^^^

The reference period's purpose is to establish a baseline of expectations for the machine
learning model being monitored. It needs to include the model inputs, model outputs and
the performance results of the monitored model. The performance of the model for this period is assumed
to be stable and acceptable.

The reference dataset contains observations for which target values
are available, so the model performance can be calculated for this set.
The ranges and distribution of model inputs, outputs and targets need to be well-known and validated for this set.

.. warning::
    Don't use model training data as a reference dataset. Machine learning models tend to overfit on their training data.
    Therefore expectations for model performance will be unrealistic.

.. _data-drift-periods-analysis:

Analysis Period
^^^^^^^^^^^^^^^

The analysis period is where NannyML analyzes the data drift and the performance of the monitored
model using the knowledge gained from analyzing the reference period. In the average use case, it will
consist of the latest production data up to a desired point in the past, which should be after
the reference period ends. The analysis period is not required to have targets available.

When performing drift analysis, NannyML compares each :term:`Data Chunk` of the analysis period
with the reference data. NannyML will flag any meaningful changes to data distributions as data drift.

The analysis data does not need to contain any target values, so performance can only be estimated for it.
If target data is provided for the analysis period, it can be used for calculating performance, but will be ignored
when estimating the performance.


Columns
-------

The following sections describe the different data columns that are required by NannyML. These will differ based on
the type of the model being monitored, and the function being used. There will be columns that are common across model types, where others will
be specific to a given model type. Also note that there is an expectation that the columns have the same name between reference and
analysis datasets when they describe the same thing.

We will illustrate this using the fictional ``car_loan`` model included with the library,
a binary classifier trying to predict whether a prospective customer will pay off a car loan.

Below we see the columns contained in our dataset.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Data Requirements.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Data Requirements.ipynb
    :cell: 2

.. nbimport::
    :path: ./example_notebooks/Tutorial - Data Requirements.ipynb
    :cells: 3

.. nbtable::
    :path: ./example_notebooks/Tutorial - Data Requirements.ipynb
    :cell: 4

In the following sections we will explain their purpose.

.. _data_requirements_columns_timestamp:

Timestamp
^^^^^^^^^

The column containing the timestamp at which the observation occurred, i.e. when the model was invoked
using the given inputs and yielding the resulting prediction. See :term:`Timestamp`.

In the sample data this is the ``timestamp`` column.

.. note::
    **Format**
        Any format supported by Pandas, most likely:

        - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
        - *Unix-epoch* in units of seconds, e.g. ``1513393355``


.. warning::
    This column is optional. When a timestamp column is not provided, plots will no longer make use of a time based x-axis
    but will use the index of the chunks instead. The following plots illustrate this:

    .. figure:: /_static/tutorials/data_requirements/data-requirements-time-based-x-axis.svg

        Plot using a time based X-axis


    .. figure:: /_static/tutorials/data_requirements/data-requirements-index-based-x-axis.svg

        Plot using an index based X-axis


    Some :class:`~nannyml.chunk.Chunker` classes might require the presence of a timestamp, such as the
    :class:`~nannyml.chunk.PeriodBasedChunker`.


Target
^^^^^^

The actual outcome of the event the machine learning model is trying to predict.

In the sample data this is the ``repaid`` column.

Required as part of the reference data for :ref:`performance estimation<performance-estimation>`,
and as part of both reference and analysis data to :ref:`calculate performance<performance-calculation>`.

Features
^^^^^^^^

The features of your model. These can be categorical or continuous and NannyML identifies this based on their
declared pandas data types.

In the sample data, the features are ``car_value``, ``salary_range``, ``debt_to_income_ratio``, ``loan_length``,
``repaid_loan_on_prev_car``, ``size_of_downpayment`` and ``driver_tenure``.

Required to :ref:`estimate performance for regression models<regression-performance-estimation>` and :ref:`detect data drift<data-drift>` on features.


Model Output columns
--------------------

Predicted class probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :term:`score<Predicted scores>` or :term:`probability<Predicted probabilities>` that is emitted by the model, most likely a float.

In the sample data this is the ``y_pred_proba`` column.

Required for running :ref:`performance estimation<performance-estimation>` on binary classification models.

In multiclass classification problems there are expected to be one column of
:term:`score<Predicted scores>` or :term:`probability<Predicted probabilities>`
for each class. They are required for running :ref:`performance estimation<performance-estimation>` on multi-class models.

Prediction class labels
^^^^^^^^^^^^^^^^^^^^^^^

The :term:`predicted label<Predicted labels>`, retrieved by interpreting (thresholding) the prediction scores or probabilities.

In the sample data this is the ``y_pred`` column.

Required for running :ref:`performance estimation<performance-estimation>` or :ref:`performance calculation<performance-calculation>` on binary classification, multi-class, and regression models.


NannyML Functionality Requirements
----------------------------------

After version 0.5 NannyML has relaxed the column requirements so that each functionality only requires what it needs.
You can see those requirements in the table below:

+--------------+---------------------------------------------------------------------------+-------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| Data         | Performance Estimation                                                    | Realized Performance                | Feature Drift                                                         | Target Drift                      | Output Drift                      |
|              +-------------------------------------+-------------------------------------+                                     +-----------------------------------+-----------------------------------+                                   |                                   |
|              | Classification models               | Regression models                   |                                     | Univariate                        | Multivariate                      |                                   |                                   |
+==============+=====================================+=====================================+=====================================+===================================+===================================+===================================+===================================+
| timestamp    |                                     |                                     |                                     |                                   |                                   |                                   |                                   |
+--------------+-------------------------------------+-------------------------------------+-------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| features     |                                     | Required (reference and analysis)   |                                     | Required (reference and analysis) | Required (reference and analysis) |                                   |                                   |
+--------------+-------------------------------------+-------------------------------------+-------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| y_pred_proba | Required (reference and analysis)   |                                     |                                     |                                   |                                   |                                   | Required (reference and analysis) |
+--------------+-------------------------------------+-------------------------------------+-------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| y_pred       | | Required (reference and analysis) | Required (reference and analysis)   | | Required (reference and analysis) |                                   |                                   |                                   | Required (reference and analysis) |
|              | | Not needed for ROC_AUC metric     |                                     | | Not needed for ROC_AUC metric     |                                   |                                   |                                   |                                   |
+--------------+-------------------------------------+-------------------------------------+-------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+
| y_true       | Required (reference only)           |  Required (reference only)          | Required (reference and analysis)   |                                   |                                   | Required (reference and analysis) |                                   |
+--------------+-------------------------------------+-------------------------------------+-------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+


What next
---------

You can check out our tutorials on how to :ref:`estimate performance<performance-estimation>`,
:ref:`calculate performance<performance-calculation>`, and :ref:`detect data drift<data-drift>`.
