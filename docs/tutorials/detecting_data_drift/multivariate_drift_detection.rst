.. _multivariate_drift_detection:

======================================
Multivariate Feature Drift Detection
======================================

Why Perform Multivariate Drift Detection
----------------------------------------

The univariate approach to data drift detection is simple and interpretable but
is not sensitive to all changes that can happen in a dataset. Multivariate drift detection
focuses on the structure of our data within the data input space and
detects changes in that structure. This allows capturing changes in our data that may not be
visible from the univariate drift detection methods.

Just The Code
-------------

If you just want the code to experiment yourself withinb a Jupyter Notebook, here you go:

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>> reference, analysis, analysis_target = nml.load_synthetic_sample()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(reference.head())
    >>>
    >>> # Let's initialize the object that will perform Data Reconstruction with PCA
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=metadata, chunk_size=5000).fit(reference_data=reference)
    >>> # let's see RC error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)
    >>>
    >>> from sklearn.impute import SimpleImputer
    >>>
    >>> # Let's initialize the object that will perform Data Reconstruction with PCA
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(
    >>>     model_metadata=metadata,
    >>>     chunk_size=5000,
    >>>     imputer_categorical=SimpleImputer(strategy='constant', fill_value='missing'),
    >>>     imputer_continuous=SimpleImputer(strategy='median')
    >>> )
    >>> # NannyML compares drift versus the full reference dataset.
    >>> rcerror_calculator.fit(reference_data=reference)
    >>> # let's see RC error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)
    >>>
    >>> # We use the data property of the results class to view the relevant data.
    >>> display(rcerror_results.data)
    >>>
    >>> figure = rcerror_results.plot(kind='drift')
    >>> figure.show()

Walkthrough on multivariate drift detection
-------------------------------------------

NannyML uses Data Reconstruction with PCA to detect such changes. For a detailed explanation of
the method see
:ref:`Data Reconstruction with PCA Deep Dive<data-reconstruction-pca>`.
The method returns a single number, :term:`Reconstruction Error`. The changes in this value
reflect a change in the structure of the model inputs. NannyML monitors the
reconstruction error over time for the monitored model and raises an alert if the
values get outside of a range defined by the variance in the reference partition.

Let's start by loading some synthetic data provided by the NannyML package.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> reference, analysis, analysis_target = nml.load_synthetic_sample()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    >>> metadata.target_column_name = 'work_home_actual'
    >>> reference.head()


+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  3 |               2.53041  | 20K - 20K €    |               2.31872 |                      7.94425 | False              | Tuesday   | 0.453649 |            3 |                  1 | 2014-05-10 01:12:09 |           0.98 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  4 |               2.25364  | 60K+ €         |               2.22127 |                      8.88448 | True               | Thursday  | 5.69526  |            4 |                  1 | 2014-05-10 02:21:34 |           0.99 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+


The :class:`~nannyml.drift.model_inputs.multivariate.data_reconstruction.calculator.DataReconstructionDriftCalculator`
module implements this functionality. One way to use it can be seen below:


.. code-block:: python

    >>> # Let's initialize the object that will perform Data Reconstruction with PCA
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=metadata, chunk_size=5000).fit(reference_data=reference)
    >>> # let's see RC error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)


An important detail is that missing values in our data need to be imputed. The default :term:`Imputation` implemented by NannyML imputes
the most frequent value for categorical features and the mean for continuous features. It takes place if the relevant optional
arguments are not specified. If needed they can be specified with an instannce of `SimpleImputer`_ class
in which cases NannyML will perform the imputation as instructed. An example where custom imputation strategies are used can be seen below:


.. code-block:: python

    >>> from sklearn.impute import SimpleImputer
    >>>
    >>> # Let's initialize the object that will perform Data Reconstruction with PCA
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(
    >>>     model_metadata=metadata,
    >>>     chunk_size=5000,
    >>>     imputer_categorical=SimpleImputer(strategy='constant', fill_value='missing'),
    >>>     imputer_continuous=SimpleImputer(strategy='median')
    >>> )
    >>> # NannyML compares drift versus the full reference dataset.
    >>> rcerror_calculator.fit(reference_data=reference)
    >>> # let's see RC error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)


Because our synthetic dataset does not have missing values, the results are the same in both cases:

.. code-block:: python

    >>> # We use the data property of the results class to view the relevant data.
    >>> rcerror_results.data

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   reconstruction_error |   lower_threshold |   upper_threshold | alert   |
+====+===============+===============+=============+=====================+=====================+=============+========================+===================+===================+=========+
|  0 | [0:4999]      |             0 |        4999 | 2014-05-09 00:00:00 | 2014-09-09 23:59:59 | reference   |                1.12096 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  1 | [5000:9999]   |          5000 |        9999 | 2014-09-09 00:00:00 | 2015-01-09 23:59:59 | reference   |                1.11807 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  2 | [10000:14999] |         10000 |       14999 | 2015-01-09 00:00:00 | 2015-05-09 23:59:59 | reference   |                1.11724 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  3 | [15000:19999] |         15000 |       19999 | 2015-05-09 00:00:00 | 2015-09-07 23:59:59 | reference   |                1.12551 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  4 | [20000:24999] |         20000 |       24999 | 2015-09-07 00:00:00 | 2016-01-08 23:59:59 | reference   |                1.10945 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  5 | [25000:29999] |         25000 |       29999 | 2016-01-08 00:00:00 | 2016-05-09 23:59:59 | reference   |                1.12276 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  6 | [30000:34999] |         30000 |       34999 | 2016-05-09 00:00:00 | 2016-09-04 23:59:59 | reference   |                1.10714 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  7 | [35000:39999] |         35000 |       39999 | 2016-09-04 00:00:00 | 2017-01-03 23:59:59 | reference   |                1.12713 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  8 | [40000:44999] |         40000 |       44999 | 2017-01-03 00:00:00 | 2017-05-03 23:59:59 | reference   |                1.11424 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
|  9 | [45000:49999] |         45000 |       49999 | 2017-05-03 00:00:00 | 2017-08-31 23:59:59 | reference   |                1.11045 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 10 | [50000:54999] |         50000 |       54999 | 2017-08-31 00:00:00 | 2018-01-02 23:59:59 | analysis    |                1.11854 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 11 | [55000:59999] |         55000 |       59999 | 2018-01-02 00:00:00 | 2018-05-01 23:59:59 | analysis    |                1.11504 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 12 | [60000:64999] |         60000 |       64999 | 2018-05-01 00:00:00 | 2018-09-01 23:59:59 | analysis    |                1.12546 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 13 | [65000:69999] |         65000 |       69999 | 2018-09-01 00:00:00 | 2018-12-31 23:59:59 | analysis    |                1.12845 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 14 | [70000:74999] |         70000 |       74999 | 2018-12-31 00:00:00 | 2019-04-30 23:59:59 | analysis    |                1.12289 |           1.09658 |           1.13801 | False   |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 15 | [75000:79999] |         75000 |       79999 | 2019-04-30 00:00:00 | 2019-09-01 23:59:59 | analysis    |                1.22839 |           1.09658 |           1.13801 | True    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 16 | [80000:84999] |         80000 |       84999 | 2019-09-01 00:00:00 | 2019-12-31 23:59:59 | analysis    |                1.22003 |           1.09658 |           1.13801 | True    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 17 | [85000:89999] |         85000 |       89999 | 2019-12-31 00:00:00 | 2020-04-30 23:59:59 | analysis    |                1.23739 |           1.09658 |           1.13801 | True    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 18 | [90000:94999] |         90000 |       94999 | 2020-04-30 00:00:00 | 2020-09-01 23:59:59 | analysis    |                1.20605 |           1.09658 |           1.13801 | True    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+
| 19 | [95000:99999] |         95000 |       99999 | 2020-09-01 00:00:00 | 2021-01-01 23:59:59 | analysis    |                1.24258 |           1.09658 |           1.13801 | True    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-------------------+-------------------+---------+

NannyML can also visualize multivariate drift results with the following code:

.. code-block:: python

    >>> figure = rcerror_results.plot(kind='drift')
    >>> figure.show()

.. image:: /_static/drift-guide-multivariate.svg

The multivariate drift results provide a consice summary of where data drift
is happening in our input data.

.. _SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html


Insights and Follow Ups
-----------------------

After reviewing the results we may want to look at the
:ref:`drift results of individual features<univariate_drift_detection>`
to see what changed in the model's feature's individually.
Moreover the :ref:`Performance Estimation<performance-estimation>` functionality can be used to
estimate the impact of the observed changes.


For more information on how multivariate drift works the
:ref:`Data Reconstruction with PCA<data-reconstruction-pca>` explanation page gives more details.
