.. _dataset-synthetic-binary:

=======================================
Synthetic Binary Classification Dataset
=======================================

NannyML provides a synthetic dataset describing a binary classification problem in
order to make it easier to test and document its features.

Problem Description
===================

The dataset describes a machine learning model that tries to predict whether an employee will
work from home on the next day.

Dataset Description
===================

A sample of the dataset can be seen below.


.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_binary_classification_dataset()
    >>> display(reference.head(3))

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+


The model uses 7 features:

- `distance_from_office`: A numerical feature. The distance in kilometers from the employee's house to the workplace.
- `salary_range`: A categorical feature with 4 categories that bin the employee's yearly income.
- `gas_price_per_litre`: A numerical feature. The price of gas per litre close to the employee's residence.
- `public_transportation_cost`: A numerical feature. The price, in euros, of public transportation from
  the employee's residence to the workplace.
- `wfh_prev_workday`: A categorical feature with 2 categories, stating whether the employee worked from home
  the previous workday.
- `workday`: A categorical feature with 5 categories. The day of the week where we want to predict whether the employee
  will work from home.
- `tenure`: A numerical feature describing how many years the employee has been at the company.

The model predicts both a probability of the employee working from home that is available from the `y_pred_proba` column.
A binary prediction is also available from the `y_pred` column. The `work_home_actual` is the :term:`Target` column describing
what actually happened.


There are also three auxiliarry columns that are helpful but not used by the monitored model:

- `identifier`: A unique number referencing each employee. This is very useful for joining the target
  results on the analysis dataset when we want to compare estimated with realized performace.
- `timestamp`: A date column informing us of the date the prediction was made.
- `partition`: The partition column tells us which :term:`Data Period` each row comes from.


Metadata Extraction
===================

The dataset's columns are name such that the heuristics NannyML uses to extract metadata can
identify them. We can see below how to extract metadata


.. code-block:: python

    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor', model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.is_complete()
    (False, ['target_column_name'])

We see that the `target_column_name` has not been correctly idenfied. We need to manually specify it.

.. code-block:: python

    >>> metadata.target_column_name = 'work_home_actual'
    >>> metadata.is_complete()
    (True, [])

Let's now see the metadata that NannyML has inferred about the model.

.. code-block:: python

    >>> metadata.to_df()

+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|    | label                             | column_name                | type        | description                                   |
+====+===================================+============================+=============+===============================================+
|  0 | timestamp_column_name             | timestamp                  | continuous  | timestamp                                     |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  1 | partition_column_name             | partition                  | categorical | partition                                     |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  2 | target_column_name                | work_home_actual           | categorical | target                                        |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  3 | distance_from_office              | distance_from_office       | continuous  | extracted feature: distance_from_office       |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  4 | salary_range                      | salary_range               | categorical | extracted feature: salary_range               |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  5 | gas_price_per_litre               | gas_price_per_litre        | continuous  | extracted feature: gas_price_per_litre        |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  6 | public_transportation_cost        | public_transportation_cost | continuous  | extracted feature: public_transportation_cost |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  7 | wfh_prev_workday                  | wfh_prev_workday           | categorical | extracted feature: wfh_prev_workday           |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  8 | workday                           | workday                    | categorical | extracted feature: workday                    |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
|  9 | tenure                            | tenure                     | continuous  | extracted feature: tenure                     |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
| 10 | prediction_column_name            | y_pred                     | continuous  | predicted label                               |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+
| 11 | predicted_probability_column_name | y_pred_proba               | continuous  | predicted score/probability                   |
+----+-----------------------------------+----------------------------+-------------+-----------------------------------------------+

For more information about specifying metadata look at :ref:`Providing Metadata<import-data>`.
