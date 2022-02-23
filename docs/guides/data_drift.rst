.. _data-drift:

====================
Data Drift Detection
====================

What is data drift and why is it important?
===========================================

Suppose we have a machine learning model that uses some multidimensional input data
:math:`\mathbf{X}` and makes predictions :math:`y`.

Our model has likely been trained on some data distribution :math:`P(\mathbf{X})`.
We have data drift when our production data comes from a different distribution
:math:`P(\mathbf{X'}) \neq P(\mathbf{X})`.

A machine learning model operating on an input distribution different than
the one it has been trained on will probably underperform. It is therefore crucial to detect
data drift, in a timely manner, when we have a model in production. By further investigating the
characteristics of the observed drift, we will be able to estimate the impact
of the drift on the model's performance.

Let us note here that data drift is not the only change that can happen when we have a
machine learning model in production. Another change we can have is concept drift.
Here, the distribution of our input data stays the same, but the relationship between our outcome
changes. In this case we have: :math:`P(y'|\mathbf{X'}) \neq P(y|\mathbf{X})` while
:math:`P(\mathbf{X'}) = P(\mathbf{X})`.

There is also a special case of data drift called label shift. In this case, the outcome
distributions between our training and production data are different, meaning
:math:`P(y') \neq P(y)`. However, the relationship between our population characteristics and
a specific outcome does not change, namely :math:`P(\mathbf{X'}|y') = P(\mathbf{X}|y)`.


Data Partitions
===============

As can be seen from our data drift discussion earlier, before we can start talking about data drift,
we need to have two different datasets to compare. NannyML uses the reference partition and the
analysis partition for this purpose.

Reference Partition
-------------------

The reference partition's purpose is to serve as a dataset suitable for our machine learning model.
We also assume that the performance results of our model are available for this dataset and that they
are acceptable.

The reference dataset can be the test set we used when evaluating our model before
we deploy it to production. Alternatively, it can be a reference (or benchmark) period when our
model is in production and its performance results where satisfactory.

Analysis Partition
------------------

The analysis partition's represents the dataset where we want to examine the performance of our
model. This will usually consist of the latest production data up to a desired point in the past,
which will be after the point where our reference partition ends. The analysis partition is not
required to have ground truth and associated performance results available.

As part of our data drift analysis, we will compare periods of the analysis partition, which we
call chunks internally in NannyML, with the reference data. NannyML will flag any meaningful
differences as data drift.


Data Drift in practice
======================

NannyML uses two approaches to detect and investigate data drift. A Univariate approach and a
Multivariate approach.

Data Preparation
----------------

We use the dataset we imported from :ref:`the import data guide<import-data>`.
Hence we assume that we have the following objects set up:

.. code-block:: python

    >>> type(md)
    <class 'nannyml.metadata.ModelMetadata'>
    >>> type(reference)
    <class 'pandas.core.frame.DataFrame'>
    >>> type(analysis)
    <class 'pandas.core.frame.DataFrame'>


.. _data-drift-univariate:

Univariate Drift Detection
--------------------------

NannyML's Univariate approach for data drift looks at each variable individually and conducts 
statistical tests comparing the chunks created from the data provided with the reference dataset.
For continuous features we use the KS Test and for categorical features we use the 2 sample
Chi squared test. Both tests provide a statistic where they measure the observed drift
and a p-value that shows how likely we are to get the observed sample if there was no drift.

The :py:class:`nannyml.drift.univariate_statistical_drift_calculator.UnivariateStatisticalDriftCalculator`
class implements the functionality needed for Univariate Drift Detection.
An example of us using it can be seen below:

.. code-block:: python

    # Let's initialize the object that will perform the Univariate Drift calculations
    # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=md, chunk_size=5000)
    # NannyML compares drift versus the full reference dataset.
    >>> univariate_calculator.fit(reference_data=reference)
    # let's see drift statistics for all available data
    >>> data = pd.concat([reference, analysis])
    >>> univariate_results = univariate_calculator.calculate(data=data)
    # let's view a small subset of our results:

    >>> univariate_results.iloc[:5, :9]
        key             start_index     end_index   start_date  end_date                partition   salary_range_chi2   salary_range_p_value    salary_range_alert
    0 	[0:4999]        0               4999        2014-05-09  2014-09-09 23:59:59     reference   2.898781            0.407                   False
    1 	[5000:9999] 	5000 	        9999 	    2014-09-09 	2015-01-09 23:59:59 	reference   3.144391 	        0.370                   False
    2 	[10000:14999] 	10000 	        14999 	    2015-01-09 	2015-05-09 23:59:59 	reference   2.451881 	        0.484 	                False
    3 	[15000:19999] 	15000 	        19999 	    2015-05-09 	2015-09-07 23:59:59 	reference   4.062620 	        0.255 	                False
    4 	[20000:24999] 	20000 	        24999 	    2015-09-07 	2016-01-08 23:59:59 	reference   2.413988 	        0.491 	                False
    >>> univariate_results.iloc[-5:, :9]
        key             start_index     end_index   start_date  end_date                partition   salary_range_chi2   salary_range_p_value    salary_range_alert
    15 	[75000:79999] 	75000           79999       2019-04-30  2019-09-01 23:59:59     analysis    455.622094          0.0                     True
    16 	[80000:84999] 	80000           84999       2019-09-01  2019-12-31 23:59:59     analysis    428.633384          0.0                     True
    17 	[85000:89999] 	85000           89999       2019-12-31  2020-04-30 23:59:59     analysis    453.247444          0.0                     True
    18 	[90000:94999] 	90000           94999       2020-04-30  2020-09-01 23:59:59     analysis    438.259970          0.0                     True
    19 	[95000:99999] 	95000           99999       2020-09-01  2021-01-01 23:59:59     analysis    474.891775          0.0                     True

We see that for each feature we have 3 columns with results. The first column contains the corresponding test
statistic. The second column contains the corresponding p-value and the third value contains whether we have
a drift alert for that feature and the relevant chunk.

TODO: Show visualizations
-------------------------
 - What about alerts?

.. _data-drift-multivariate:

Multivariate Drift Detection
----------------------------

The univariate approach to data drift detection is very useful. But unfortunately it does not
tell us the full story. Data living in multidimensional spaces can have complex structures
whose change may not be visible by just viewing the distributions of each features. We go
into more detail on this issue at :ref:`Data Reconstruction with PCA Deep Dive<data-reconstruction-pca>`.

For drift detection purposes the key thing we need to know is that a change in reconstruction error
values reflects a change in the structure we have learnt for our data. We therefore monitor
reconstruction error over time for our machine learning models and raise an alert if the
values get outside the range of what we are accustomed to.

The :py:class:`nannyml.drift.data_reconstruction_drift_calcutor.DataReconstructionDriftCalculator`
module implements this functionality. An example of us using it can be seen below:


.. code-block:: python

    # Let's initialize the object that will perform Data Reconstruction with PCA
    # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=md, chunk_size=5000)
    # NannyML compares drift versus the full reference dataset.
    >>> rcerror_calculator.fit(reference_data=reference)
    # let's see RC error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)
    >>> rcerror_results

        key             start_index end_index   start_date  end_date                partition 	reconstruction_error    alert
    0   [0:4999]        0           4999        2014-05-09  2014-09-09 23:59:59     reference   1.120961                False
    1   [5000:9999]     5000        9999        2014-09-09  2015-01-09 23:59:59     reference   1.118071                False
    2   [10000:14999]   10000       14999       2015-01-09  2015-05-09 23:59:59     reference   1.117237                False
    3   [15000:19999]   15000       19999       2015-05-09  2015-09-07 23:59:59     reference   1.125514                False
    4   [20000:24999]   20000       24999       2015-09-07  2016-01-08 23:59:59     reference   1.109446                False
    5   [25000:29999]   25000       29999       2016-01-08  2016-05-09 23:59:59     reference   1.122759                False
    6   [30000:34999]   30000       34999       2016-05-09  2016-09-04 23:59:59     reference   1.107138                False
    7   [35000:39999]   35000       39999       2016-09-04  2017-01-03 23:59:59     reference   1.127134                False
    8   [40000:44999]   40000       44999       2017-01-03  2017-05-03 23:59:59     reference   1.114237                False
    9   [45000:49999]   45000       49999       2017-05-03  2017-08-31 23:59:59     reference   1.110450                False
    10  [50000:54999]   50000       54999       2017-08-31  2018-01-02 23:59:59     analysis    1.118536                False
    11  [55000:59999]   55000       59999       2018-01-02  2018-05-01 23:59:59     analysis    1.115044                False
    12  [60000:64999]   60000       64999       2018-05-01  2018-09-01 23:59:59     analysis    1.125460                False
    13  [65000:69999]   65000       69999       2018-09-01  2018-12-31 23:59:59     analysis    1.128453                False
    14  [70000:74999]   70000       74999       2018-12-31  2019-04-30 23:59:59     analysis    1.122892                False
    15  [75000:79999]   75000       79999       2019-04-30  2019-09-01 23:59:59     analysis    1.228393                True
    16  [80000:84999]   80000       84999       2019-09-01  2019-12-31 23:59:59     analysis    1.220028                True
    17  [85000:89999]   85000       89999       2019-12-31  2020-04-30 23:59:59     analysis    1.237394                True
    18  [90000:94999]   90000       94999       2020-04-30  2020-09-01 23:59:59     analysis    1.206051                True
    19  [95000:99999]   95000       99999       2020-09-01  2021-01-01 23:59:59     analysis    1.242579                True
    # TODO: Show visualizations of results

