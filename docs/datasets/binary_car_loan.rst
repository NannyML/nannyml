.. _dataset-synthetic-binary-car-loan:

===================================================
Synthetic Binary Classification Dataset - Car Loan
===================================================

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

- `car_value`: A numerical feature representing the price of the car.

- `salary_range`: A categorical feature with 4 categories that identify the range
  the employee's yearly income falls within.

- `debt_to_income_ratio`: A numerical feature representing the ratio of debt to income from the customer.

- `loan_length`: A numerical feature representing in how many months the customer wants to repay the loan.

- `repaid_loan_on_prev_car`: A categorical feature with 2 categories, stating whether the customer
  repaid or not a previous loan.

- `size_of_downpayment`: A categorical feature with 10 categories, representing the percentage in increments of 10%
  of the size of the downpayment of the car value.

- `tenure`: A numerical feature describing how many years the costumer has been driving.


There are 3 columns that reference the output of the model:

- `y_pred_proba`: The model predicted probability of the customer repaying the loan.
- `y_pred`: The model prediction in binary form.
- `repaid`: The :term:`Target` column describing if the customer actually repaid the loan.


There is also an auxiliary column that is helpful but not used by the monitored model:

- `timestamp`: A date column informing us of the date the prediction was made.

