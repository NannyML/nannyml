.. _dataset-synthetic-regression:

=======================================
Synthetic Regression Dataset
=======================================

NannyML provides a synthetic dataset describing a regression problem,
to make it easier to test and document its features.

To find out what requirements NannyML has for datasets, check out :ref:`Data Requirements<data_requirements>`.

Problem Description
===================

The dataset describes a machine learning model that tries to predict the price of a used car.

Dataset Description
===================

A sample of the dataset can be seen below.


.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.datasets.load_synthetic_car_price_dataset()
    >>> display(reference.head())

+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|    |   car_age |   km_driven |   price_new |   accident_count |   door_count | fuel     | transmission   |   y_true |   y_pred | timestamp               |
+====+===========+=============+=============+==================+==============+==========+================+==========+==========+=========================+
|  0 |        15 |      144020 |       42810 |                4 |            3 | diesel   | automatic      |      569 |     1246 | 2017-01-24 08:00:00.000 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  1 |        12 |       57078 |       31835 |                3 |            3 | electric | automatic      |     4277 |     4924 | 2017-01-24 08:00:33.600 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  2 |         2 |       76288 |       31851 |                3 |            5 | diesel   | automatic      |     7011 |     5744 | 2017-01-24 08:01:07.200 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  3 |         7 |       97593 |       29288 |                2 |            3 | electric | manual         |     5576 |     6781 | 2017-01-24 08:01:40.800 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  4 |        13 |        9985 |       41350 |                1 |            5 | diesel   | automatic      |     6456 |     6822 | 2017-01-24 08:02:14.400 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+

The model uses 7 features:

- **car_age** - a numerical feature. The age of the car in years.
- **km_driven** - a numerical feature. The number of kilometers a car has drived.
- **price_new** - a numerical feature. The price of the car in Euros when it was new.
- **accident_count** - a numerical feature. The number of accidents the car has been involved in.
- **door_count** - a numerical feature. The number of doors the car has. If it is a hatchback, the door count is increased by 1.
- **fuel** - a categorical feature describing whether the car uses gas, diesel or electricity as fuel.
- **transmission** - a categorical feature describing whether the car uses manual or automatic transmission.


The model predicts the predicted price of the car at the **y_pred** column.
The **y_true** is the :term:`Target` column describing the actual value of the car.


There is also an auxiliary column that is helpful but not used by the monitored model:

- **timestamp** - a date column informing us of the date the prediction was made.
