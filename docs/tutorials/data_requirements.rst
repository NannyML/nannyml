.. _data_requirements:

==================
Data requirements
==================

In this guide we'll present an overview of the different kinds of data NannyML requires to run it's various features. The specifics for each feature are also covered in the various :ref:`Tutorials`, but an overview of all the different requirements is presented here for reference.

.. _data-drift-periods:

Data Periods
------------

NannyML works with two :term:`Data Periods<data period>`. The first one is called the :ref:`reference period<data-drift-periods-reference>`,
is represented by the **reference dataset**, and is used to establish the expectations of the model's performance.
The second is called the :ref:`analysis period<data-drift-periods-analysis>`, is represented by the **analysis period**,
and is used to whether the model is still performing according to expectations.

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
--------------

The following sections describe the different data columns that are required by NannyML. These will differ based on
the type of the model being monitored, and the function being used. There will be columns that are common across model types, where others will
be specific to a given model type.

We will illustrate this using the fictional ``work_from_home`` model included with the library,
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

Currently required for all features of NannyML, though we are looking to drop this requirement in a future release.

Target
^^^^^^

The actual outcome of the event the machine learning model is trying to predict.

In the sample data this is the ``work_home_actual`` column.

Required as part of the reference data for :ref:`performance estimation<performance-estimation>`, and as part of both reference and analysis data to :ref:`calculate performance<performance-calculation>`.

Features
^^^^^^^^^

The features of your model. These can be categorical or continuous, and NannyML identified this based on their declared pandas data types. 

In the sample data, the features are ``distance_from_office``, ``salary_range``, ``gas_price_per_litre``, ``public_transportation_cost``, ``wfh_prev_workday``, ``workday`` and ``tenure``.

Required to :ref:`detect data drift<data-drift>` on features.



Binary classification columns
-----------------------------

Predicted probability
^^^^^^^^^^^^^^^^^^^^^

The :term:`score<Predicted scores>` or :term:`probability<Predicted probabilities>` that is emitted by the model, most likely a float. 

In the sample data this is the ``y_pred_proba`` column.

Required for running :ref:`performance estimation<performance-estimation>` on binary classification models.


Predicted class probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :term:`scores<Predicted scores>` or :term:`probabilities<Predicted probabilities>` emitted by the model, a single
column for each class.

.. warning::
    Either this or the prediction should be set for the metadata to be complete.

Required for running :ref:`performance estimation<performance-estimation>` on multi-class models.


Prediction
^^^^^^^^^^

The :term:`predicted label<Predicted labels>`, retrieved by interpreting (thresholding) the prediction scores or probabilities.

In the sample data this is the ``y_pred`` column.

Required for running :ref:`performance estimation<performance-estimation>` or :ref:`calculate performance<performance-calculation>` on binary classification, multi-class, and regression models.




What next
-----------------------

You can check out our tutorials on how to :ref:`estimate performance<performance-estimation>`, 
:ref:`calculate performance<performance-calculation>`, and :ref:`detect data drift<data-drift>`.
