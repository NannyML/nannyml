.. _dataset-synthetic-binary-car-loan:

================================================
Synthetic Binary Classification Car Loan Dataset
================================================

NannyML provides a synthetic dataset describing a binary classification problem,
to make it easier to test and document its features.

To find out what requirements NannyML has for datasets, check out :ref:`Data Requirements<data_requirements>`.

Problem Description
===================

The dataset describes a machine learning model that predicts whether a customer
will repay a loan to buy a car.

Dataset Description
===================

A sample of the dataset can be seen below.

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_car_loan_dataset()
    >>> display(reference.head(3))


+----+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+----------------+----------+----------+-------------------------+
|    |   car_value | salary_range   |   debt_to_income_ratio |   loan_length | repaid_loan_on_prev_car   | size_of_downpayment   |   driver_tenure |   y_pred_proba |   y_pred |   repaid | timestamp               |
+====+=============+================+========================+===============+===========================+=======================+=================+================+==========+==========+=========================+
|  0 |       39811 | 40K - 60K €    |               0.63295  |            19 | False                     | 40%                   |        0.212653 |           0.99 |        1 |        1 | 2018-01-01 00:00:00.000 |
+----+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+----------------+----------+----------+-------------------------+
|  1 |       12679 | 40K - 60K €    |               0.718627 |             7 | True                      | 10%                   |        4.92755  |           0.07 |        0 |        0 | 2018-01-01 00:08:43.152 |
+----+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+----------------+----------+----------+-------------------------+
|  2 |       19847 | 40K - 60K €    |               0.721724 |            17 | False                     | 0%                    |        0.520817 |           1    |        1 |        1 | 2018-01-01 00:17:26.304 |
+----+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+----------------+----------+----------+-------------------------+

The model uses 7 features:

- **car_value** - a numerical feature representing the price of the car.

- **salary_range** - a categorical feature with 4 categories that identify the range
  the employee's yearly income falls within.

- **debt_to_income_ratio** - a numerical feature representing the ratio of debt to income from the customer.

- **loan_length** - a numerical feature representing in how many months the customer wants to repay the loan.

- **repaid_loan_on_prev_car** - a categorical feature with 2 categories, stating whether the customer
  repaid or not a previous loan.

- **size_of_downpayment** - a categorical feature with 10 categories, representing the percentage in increments of 10%
  of the size of the downpayment of the car value.

- **tenure** - a numerical feature describing how many years the costumer has been driving.


There are 3 columns that reference the output of the model:

- **y_pred_proba** - the model predicted probability of the customer repaying the loan.
- **y_pred** - the model prediction in binary form.
- **repaid** - the :term:`Target` column describing if the customer actually repaid the loan.


There is also an auxiliary column that is helpful but not used by the monitored model:

- **timestamp** - a date column informing us of the date the prediction was made.


Data Quality Version
======================

NannyML also provides a version of the car loan dataset that includes missing and unseen values in order to
demonstrate the data quality modules provided by NannyML. The problem modeled and the columns included are the
same. You can access this dataset with:

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_synthetic_car_loan_data_quality_dataset()
    >>> # let's show an instance where new and missing values are present.
    >>> display(analysis.iloc[41515:41520])

+-------+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+-------------------------+----------------+----------+----------+
|       |   car_value | salary_range   |   debt_to_income_ratio |   loan_length | repaid_loan_on_prev_car   | size_of_downpayment   |   driver_tenure | timestamp               |   y_pred_proba | period   |   y_pred |
+=======+=============+================+========================+===============+===========================+=======================+=================+=========================+================+==========+==========+
| 41515 |       58071 | 40K - 60K €    |               0.694352 |            20 | True                      | 30%                   |        0.44644  | 2019-07-09 02:57:35.280 |           0.9  | analysis |        1 |
+-------+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+-------------------------+----------------+----------+----------+
| 41516 |       40317 | 20K - 20K €    |               0.581372 |             8 | True                      | 50%                   |      nan        | 2019-07-09 03:06:18.432 |           0.16 | analysis |        0 |
+-------+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+-------------------------+----------------+----------+----------+
| 41517 |       57487 | 40K - 60K €    |               0.703041 |             7 | True                      | 30%                   |        5.2826   | 2019-07-09 03:15:01.584 |           0.07 | analysis |        0 |
+-------+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+-------------------------+----------------+----------+----------+
| 41518 |       21555 | 0 - 20K €      |               0.268774 |            16 | False                     | 0%                    |        4.04887  | 2019-07-09 03:23:44.736 |           0.01 | analysis |        0 |
+-------+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+-------------------------+----------------+----------+----------+
| 41519 |       78265 | 40K - 60K €    |               0.71856  |            19 | True                      | 40%                   |        0.208278 | 2019-07-09 03:32:27.888 |           0.85 | analysis |        1 |
+-------+-------------+----------------+------------------------+---------------+---------------------------+-----------------------+-----------------+-------------------------+----------------+----------+----------+

The dataset has induced missing values at **salary_range** and **driver_tenure** features. And it has a new value, ``50%`` at **size_of_downpayment** feature.
You can see how the dataset is used on the :ref:`data quality tutorials<data-quality>`.
