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
================

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

The analysis partition's purpose is the dataset where we want to examine the performance of our
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


Univariate Drift Detection
--------------------------

The Univariate approach looks at each variable individually and conducts statistical tests comparing
the chunks created from the datasets with the reference dataset. For continuous features we use the
KS Test and for categorical features we use the 2 sample Chi squared test. Both tests provide a
statistic where they measure the observed drift and a p-value that shows how likely we are to
get the observed sample if there was no drift.

The :meth:`nannyml.drift.statistical_drift_calculator` module implements this functionality.
An example of us using it can be seen below:

.. code-block:: python

    # Let's initialize the object that will perform the Univariate Drift calculations
    >>> univariate_calculator = nml.StatisticalDriftCalculator(model_metadata=md)
    # We want to compare drift versus the full reference dataset.
    >>> univariate_calculator.fit(reference_data=reference)
    # let's see drift statistics for all available data (and let's use a chunk size of 5000 data points)
    >>> data = pd.concat([reference, analysis])
    >>> univariate_results = univariate_calculator.calculate(data=data, chunk_size=5000)
    >>> univariate_results
        key 	        start_index     end_index   start_date 	    end_date 	                partition 	wfh_prev_workday_chi2 	wfh_prev_workday_p_value    wfh_prev_workday_alert 	salary_range_chi2 	... 	distance_from_office_alert  public_transportation_cost_dstat 	public_transportation_cost_p_value  public_transportation_cost_alert 	    gas_price_per_litre_dstat 	gas_price_per_litre_p_value 	gas_price_per_litre_alert   tenure_dstat   tenure_p_value 	tenure_alert
    0 	[0:4999]        0               4999 	    2011-01-02 	    2020-12-31 23:59:59 	reference 	0.414606 	        0.520 	                    False 	                2.898781 	        ... 	False 	                    0.00998 	                        0.752 	                            False 	                            0.01122 	                0.612 	                        False 	                    0.00978 	    0.774 	        False
    1 	[5000:9999] 	5000 	        9999 	    2011-01-01 	    2020-12-31 23:59:59 	reference 	0.033486 	        0.855 	                    False 	                3.144391 	        ... 	False	                    0.01046 	                        0.698 	                            False 	                            0.01222 	                0.502 	                        False 	                    0.01192 	    0.534       	False
    2 	[10000:14999] 	10000 	        14999 	    2011-01-01 	    2020-12-30 23:59:59 	reference 	0.168656 	        0.681 	                    False 	                2.451881 	        ... 	False 	                    0.01706 	                        0.140 	                            False 	                            0.00886 	                0.865 	                        False 	                    0.01268 	    0.454 	        False
    3 	[15000:19999] 	15000 	        19999 	    2011-01-01 	    2020-12-31 23:59:59 	reference 	0.056270 	        0.812 	                    False 	                4.062620 	        ... 	False 	                    0.01220 	                        0.504 	                            False 	                            0.00956 	                0.797 	                        False 	                    0.01074 	    0.667 	        False
    4 	[20000:24999] 	20000 	        24999 	    2011-01-03 	    2020-12-31 23:59:59 	reference 	0.242059 	        0.623 	                    False 	                2.413988 	        ... 	False 	                    0.00662 	                        0.988 	                            False 	                            0.00758 	                0.955 	                        False 	                    0.00924 	    0.829 	        False
    5 	[25000:29999] 	25000 	        29999 	    2011-01-01 	    2020-12-30 23:59:59 	reference 	3.614573 	        0.057 	                    False 	                3.796063 	        ... 	False 	                    0.01186 	                        0.541 	                            False 	                            0.01032 	                0.714 	                        False 	                    0.00794 	    0.935 	        False
    6 	[30000:34999] 	30000 	        34999 	    2011-01-02 	    2020-12-28 23:59:59 	reference 	0.075705 	        0.783 	                    False 	                3.228836 	        ... 	False 	                    0.00636 	                        0.992 	                            False 	                            0.01094 	                0.644 	                        False 	                    0.01120 	    0.615 	        False
    7 	[35000:39999] 	35000 	        39999 	    2011-01-02 	    2021-01-01 23:59:59 	reference 	0.414606 	        0.520 	                    False 	                1.393300 	        ... 	False 	                    0.00832 	                        0.909 	                            False 	                            0.01736 	                0.128 	                        False 	                    0.00740 	    0.963 	        False
    8 	[40000:44999] 	40000 	        44999 	    2011-01-01 	    2020-12-31 23:59:59 	reference 	0.012656 	        0.910 	                    False 	                0.304785 	        ... 	False 	                    0.01176 	                        0.552 	                            False 	                            0.00842 	                0.901 	                        False 	                    0.01464 	    0.281 	        False
    9 	[45000:49999] 	45000 	        49999 	    2011-01-01 	    2020-12-31 23:59:59 	reference 	2.203832 	        0.138 	                    False 	                2.987581 	        ... 	False 	                    0.00820 	                        0.917 	                            False                                   0.00786 	                0.939 	                        False 	                    0.01306 	    0.417 	        False
    10 	[50000:54999] 	50000 	        54999 	    2011-01-02 	    2020-12-31 23:59:59 	analysis 	1.703195 	        0.192 	                    False 	                1.033683 	        ... 	False 	                    0.00956 	                        0.797 	                            False 	                            0.01576 	                0.207 	                        False 	                    0.02124 	    0.033 	        True
    11 	[55000:59999] 	55000 	        59999 	    2011-01-01 	    2020-12-31 23:59:59 	analysis 	0.242059 	        0.623 	                    False 	                5.762412 	        ... 	False 	                    0.01488 	                        0.264 	                            False 	                            0.01272 	                0.450 	                        False 	                    0.01006 	    0.743 	        False
    12 	[60000:64999] 	60000 	        64999 	    2011-01-01 	    2020-12-29 23:59:59 	analysis 	3.178618 	        0.075 	                    False 	                2.653961 	        ... 	False 	                    0.01290 	                        0.432 	                            False 	                            0.01746 	                0.124 	                        False 	                    0.02370 	    0.012       	True
    13 	[65000:69999] 	65000 	        69999 	    2011-01-02 	    2020-12-30 23:59:59 	analysis 	0.024299 	        0.876 	                    False 	                0.070843 	        ... 	False 	                    0.01598 	                        0.194 	                            False 	                            0.01282 	                0.440 	                        False 	                    0.01446 	    0.295 	        False
    14 	[70000:74999] 	70000 	        74999 	    2011-01-02 	    2020-12-31 23:59:59 	analysis 	0.487381 	        0.485 	                    False 	                1.005422 	        ... 	False 	                    0.01136 	                        0.596 	                            False 	                            0.01922 	                0.069 	                        False 	                    0.00912 	    0.841 	        False
    15 	[75000:79999] 	75000 	        79999 	    2011-01-01 	    2020-12-31 23:59:59 	analysis 	1179.903143             0.000                       True 	                455.622094 	        ... 	True 	                    0.18346 	                        0.000 	                            True 	                            0.00824 	                0.915 	                        False 	                    0.00702 	    0.977 	        False
    16 	[80000:84999] 	80000 	        84999 	    2011-01-03 	    2020-12-31 23:59:59 	analysis 	1162.989441 	        0.000 	                    True 	                428.633384 	        ... 	True 	                    0.18334 	                        0.000 	                            True 	                            0.01068 	                0.674 	                        False 	                    0.00826 	    0.913 	        False
    17 	[85000:89999] 	85000 	        89999 	    2011-01-01 	    2020-12-30 23:59:59 	analysis 	1170.491329 	        0.000 	                    True 	                453.247444 	        ... 	True 	                    0.20062 	                        0.000 	                            True 	                            0.01002 	                0.748 	                        False 	                    0.01398 	    0.334 	        False
    18 	[90000:94999] 	90000 	        94999 	    2011-01-02 	    2021-01-01 23:59:59 	analysis 	1023.347641 	        0.000 	                    True 	                438.259970 	        ... 	True 	                    0.18740 	                        0.000 	                            True 	                            0.01070 	                0.671 	                        False 	                    0.00896 	    0.856 	        False
    19 	[95000:99999] 	95000 	        99999 	    2011-01-02 	    2020-12-31 23:59:59 	analysis 	1227.536732 	        0.000 	                    True 	                474.891775 	        ... 	True 	                    0.20018 	                        0.000 	                            True 	                            0.00700 	                0.978 	                        False 	                    0.00632 	    0.993 	        False


We see that for each feature we have 3 columns with results. The first column contains the corresponding test
statistic. The second column contains the corresponding p-value and the third value contains whether we have
a drift alert for that feature and the relevant chunk.


Multivariate Drift Detection
----------------------------

- Univariate Changes in the data distributions
    - We use statistical tests to detect and measure changes

- Multivariate changes …
    - Multidimensional data can change in ways that are not obvious from univariate views
    - We use reconstruction error to detect them
