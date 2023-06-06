.. _dataset-synthetic-binary:

=======================================
Synthetic Binary Classification Dataset
=======================================

NannyML provides a synthetic dataset describing a binary classification problem,
to make it easier to test and document its features.

To find out what requirements NannyML has for datasets, check out :ref:`Data Requirements<data_requirements>`.

Problem Description
===================

The dataset describes a machine learning model that tries to predict whether an employee will
work from home on the next day.

Dataset Description
===================

A sample of the dataset can be seen below.


.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.datasets.load_synthetic_binary_classification_dataset()
    >>> display(reference.head(3))

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+----------+

The model uses 7 features:

- **distance_from_office** - a numerical feature. The distance in kilometers from the employee's house to the workplace.
- **salary_range** - a categorical feature with 4 categories that identify the range the employee's yearly income falls within.
- **gas_price_per_litre** - a numerical feature. The price of gas per litre close to the employee's residence.
- **public_transportation_cost** - a numerical feature. The price, in euros, of public transportation from
  the employee's residence to the workplace.
- **wfh_prev_workday** - a categorical feature with 2 categories, stating whether the employee worked from home
  the previous workday.
- **workday** - a categorical feature with 5 categories. The day of the week where we want to predict whether the employee
  will work from home.
- **tenure** - a numerical feature describing how many years the employee has been at the company.

The model predicts the probability of the employee working from home, recorded in the **y_pred_proba** column.
A binary prediction is also available from the **y_pred** column. The **work_home_actual** is the :term:`Target` column describing
what actually happened.


There are also two auxiliary columns that are helpful but not used by the monitored model:

- **identifier** - a unique number referencing each employee. This is very useful for joining the target
  results on the analysis dataset, when we want to :ref:`compare estimated with realized performace.<compare_estimated_and_realized_performance>`.
- **timestamp** - a date column informing us of the date the prediction was made.
