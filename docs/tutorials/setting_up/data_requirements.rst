.. _data_requirements:

==================
Data requirements
==================

In this guide we'll present an overview of the data NannyML requires to function.
It serves as a starting point for the guide on :ref:`providing metadata<import-data>`.

.. _data-drift-periods:

Data Periods
------------

In order to monitor a model, NannyML needs to learn about it from a reference dataset (*reference period*).
Then it can monitor the data that is subject to actual analysis (*analysis period*).

Reference Period
^^^^^^^^^^^^^^^^

The reference period's purpose is to establish a baseline of expectations for the machine
learning model being monitored. It needs to include the model inputs, model outputs and
the performance results of the monitored model. The performance of the model for this period is assumed
to be stable and acceptable.

The *reference* dataset contains observations for which target values
are available, hence the model performance can be *calculated* for this set.
The ranges and distribution of model inputs, outputs and targets is well-known and validated for this set.

.. warning::
    Don't use *training* data as a *reference* data set to prevent overfitting, e.g during model score calibration.


Analysis Period
^^^^^^^^^^^^^^^

The *analysis* period is where NannyML analyzes the data drift and the performance of the monitored
model using the knowledge gained from analyzing the *reference* period.
The *analysis* period will usually consist of the latest production data up to a desired point in
the past, which should be after the point where the reference period ends.
The analysis period is not required to have targets available.

When performing drift analysis, NannyML compares each :term:`Data Chunk` of the analysis period
with the reference data. NannyML will flag any meaningful changes to data distributions as data drift.

The *analysis* data does not contain any target values, so performance can only be *estimated* for it.


Common columns
--------------

The following sections describe the different data columns that are required by NannyML. These will differ based on
the type of the model being monitored. There will be columns that are common across model types, where others will
be specific to a given model type.

We will illustrate this using the fictional **work_from_home** model,
a binary classifier trying to predict whether someone will be working from home on a given day or not.


.. code-block:: python

    >>> import nannyml as nml
    >>> reference, _, _ = nml.load_synthetic_binary_classification_dataset()
    >>> reference[['identifier', 'work_home_actual', 'timestamp', 'y_pred_proba',
       'partition', 'y_pred']].head()

+----+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+==============+====================+=====================+================+=============+==========+
|  0 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+--------------+--------------------+---------------------+----------------+-------------+----------+
|  3 |            3 |                  1 | 2014-05-10 01:12:09 |           0.98 | reference   |        1 |
+----+--------------+--------------------+---------------------+----------------+-------------+----------+
|  4 |            4 |                  1 | 2014-05-10 02:21:34 |           0.99 | reference   |        1 |
+----+--------------+--------------------+---------------------+----------------+-------------+----------+

.. code-block:: python

    >>> reference, _, _ = nml.load_synthetic_binary_classification_dataset()
    >>> reference[['distance_from_office', 'salary_range', 'gas_price_per_litre',
       'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure']].head()

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |
+====+========================+================+=======================+==============================+====================+===========+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+
|  3 |               2.53041  | 20K - 40K €    |               2.31872 |                      7.94425 | False              | Tuesday   | 0.453649 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+
|  4 |               2.25364  | 60K+ €         |               2.22127 |                      8.88448 | True               | Thursday  | 5.69526  |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+


Timestamp
^^^^^^^^^^^^

The column containing the timestamp at which the observation occurred, i.e. when the model was invoked
using the given inputs and yielding the resulting prediction. See :term:`Timestamp`.

In the sample data this is the ``timestamp`` column.

.. note::
    **Format**
        Any format supported by Pandas, most likely:

        - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
        - *Unix-epoch* in units of seconds, e.g. ``1513393355``

Target
^^^^^^

The actual outcome of the event the machine learning model is trying to predict.

In the sample data this is the ``work_home_actual`` column.

.. note::
    **Target** values are only required in the reference data.
    Performance in will be *calculated* using them.
    In the *analysis data* where they are not required, performance can be *estimated*. This :ref:`performance-estimation`
    will use the targets in reference period and the :term:`Model Outputs`
    in the analysis period to estimate performance in the analysis dataset.

Period
^^^^^^

The period each observation belongs to. An indicator for NannyML on whether to use this observation as
*reference* data or *analysis* data.

In the sample data this is the ``partition`` column.

.. note::
    We are aware that the term ``partition`` can be confusing. Preparations are in the work to phase out this name
    and eventually the need for this column entirely.

Binary classification columns
-----------------------------

Predicted probability
^^^^^^^^^^^^^^^^^^^^^

The :term:`score<Predicted scores>` or :term:`probability<Predicted probabilities>` that is emitted by the model, most likely a float.

In the sample data this is the ``y_pred`` column.


Prediction
^^^^^^^^^^

The :term:`predicted label<Predicted labels>`, retrieved by interpreting (thresholding) the prediction scores or probabilities.

In the sample data this is the ``y_pred_proba`` column.


Multiclass classification columns
---------------------------------

Predicted class probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :term:`scores<Predicted scores>` or :term:`probabilities<Predicted probabilities>` emitted by the model, a single
column for each class.

.. warning::
    Either this or the *prediction* should be set for the metadata to be complete.


Prediction
^^^^^^^^^^

The :term:`predicted label<Predicted labels>`, retrieved by interpreting (thresholding) the prediction scores or probabilities.


Insights and Follow Ups
-----------------------

Read more on how to describe your dataset to NannyML by :ref:`providing model metadata<import-data>`.
