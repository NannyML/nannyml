.. _data-drift:

====================
Data Drift Detection
====================

What is data drift and why is it important?
===========================================

Let's study a machine learning model that uses some multidimensional input data
:math:`\mathbf{X}` and makes predictions :math:`y`.

The model has been trained on some data distribution :math:`P(\mathbf{X})`.
There is data drift when the production data comes from a different distribution
:math:`P(\mathbf{X'}) \neq P(\mathbf{X})`.

A machine learning model operating on an input distribution different than
the one it has been trained on will probably underperform. It is therefore crucial to detect
data drift, in a timely manner, when a model is in production. By further investigating the
characteristics of the observed drift, the data scientists operating the model
will be able to estimate the impact of the drift on the model's performance.

There is also a special case of data drift called label shift. In this case, the outcome
distributions between the training and production data are different, meaning
:math:`P(y') \neq P(y)`. However, the relationship between the population characteristics and
a specific outcome does not change, namely :math:`P(\mathbf{X'}|y') = P(\mathbf{X}|y)`.

It is important to note that data drift is not the only change that can happen when there is a
machine learning model in production. Another important change is concept drift, where the
distribution of the model's input data stays the same, but their relationship with the outcome
changes. In this case we have: :math:`P(y'|\mathbf{X'}) \neq P(y|\mathbf{X})` while
:math:`P(\mathbf{X'}) = P(\mathbf{X})`.


Data Partitions
===============

As described earlier, in order to look for data drift one needs to have two different datasets
to compare. NannyML uses the reference partition and the analysis partition for this purpose.

Reference Partition
-------------------

The reference partition's purpose is to establish a baseline of expectations for the machine
learning model being monitored. The assumption here is that the model inputs, model outputs as well
as the performance results of the monitored model are available for the reference partition, have
been reviewed and are acceptable.

The reference dataset can be a reference (or benchmark) period when the
monitored model has been in production and its performance results where satisfactory.
Alternatively, it can be the test set used when evaluating the monitored model before
deploying it to production.

Analysis Partition
------------------

The analysis partition is the partition of interest where the data science practitioner wants
to examine the performance of the monitored model. The analysis partition will usually consist
of the latest production data up to a desired point in
the past, which will be after the point where the reference partition ends.
The analysis partition is not required to have ground truth and associated performance results
available.

NannyML when performing drift analysis compares :term:`Data Chunk` of the analysis partition
with the reference data. NannyML will flag any meaningful differences as data drift.

Drift alerts
============

NannyML uses a statistical approach to issuing an :term:`Alert`. It establishes an expected baseline from
the reference data and when the drift results for a chunk are unlikely, given the expectations
from the baseline, then it issues a drift alert. Given this statistical approach, there can be
cases where the alert is a false positive. However when reviewing the data drift vizualizations
they will be easy to spot and discard. An example of that will be presented later.

Data Drift in practice
======================

NannyML uses two approaches to detect and investigate data drift. A Univariate approach and a
Multivariate approach.

Data Preparation
----------------

For demonstration purposes the synthetic sample data provided with the NannyML package
are shown here.

.. code-block:: python

    import nannyml as nml

    reference, analysis, analysis_gt = nml.load_synthetic_sample()
    md = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    md.timestamp_column_name = 'timestamp'
    md.ground_truth_column_name = 'work_home_actual'


.. _data-drift-univariate:

Univariate Drift Detection
--------------------------

NannyML's Univariate approach for data drift looks at each variable individually and conducts
statistical tests comparing the chunks created from the data provided with the reference dataset.
NannyML uses the KS Test for continuous features and the 2 sample
Chi squared test for categorical features. Both tests provide a statistic where they measure the
observed drift and a p-value that shows how likely we are to get the observed sample
under the assumption that there was no drift. If the p-value is less than 0.05 NannyML considers
the result unlikely and issues an alert for the associated chunk and feature.

The :py:class:`nannyml.drift.univariate_statistical_drift_calculator.UnivariateStatisticalDriftCalculator`
class implements the functionality needed for Univariate Drift Detection.
An example using it can be seen below:

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

NannyML produces 3 columns with results for each feature. The first column contains the corresponding test
statistic. The second column contains the corresponding p-value and the third column says whether there
is a drift alert for that feature and the relevant chunk.

NannyML can also visualize those results with the following code:

.. code-block:: python

    # Let's initialize the plotting class:
    plots = nml.DriftPlots(model_metadata=univariate_calculator.model_metadata, chunker=univariate_calculator.chunker)

    for itm in md.features:
        fig = plots.plot_univariate_statistical_drift(univariate_results, metric='statistic', feature_label=itm.label)
        fig.show()

.. image:: ../_static/drift-guide-distance_from_office.svg

.. image:: ../_static/drift-guide-gas_price_per_litre.svg

.. image:: ../_static/drift-guide-tenure.svg

.. image:: ../_static/drift-guide-wfh_prev_workday.svg

.. image:: ../_static/drift-guide-workday.svg

.. image:: ../_static/drift-guide-public_transportation_cost.svg

.. image:: ../_static/drift-guide-salary_range.svg


NannyML also shows details about the distributions of continuous variables and
stacked bar charts for categorical variables. It does so with the following code:


.. code-block:: python

    for itm in md.continuous_features:
        fig = plots.plot_continuous_feature_distribution_over_time(
            data=pd.concat([reference, analysis], ignore_index=True),
            drift_results=univariate_results,
            feature_label=itm.label
        )
        fig.show()

.. image:: ../_static/drift-guide-joyplot-distance_from_office.svg

.. image:: ../_static/drift-guide-joyplot-gas_price_per_litre.svg

.. image:: ../_static/drift-guide-joyplot-public_transportation_cost.svg

.. image:: ../_static/drift-guide-joyplot-tenure.svg

.. code-block:: python

    for itm in md.categorical_features:
        fig = plots.plot_categorical_feature_distribution_over_time(
            data=pd.concat([reference, analysis], ignore_index=True),
            drift_results=univariate_results,
            feature_label=itm.label
        )
        fig.show()

.. image:: ../_static/drift-guide-stacked-salary_range.svg

.. image:: ../_static/drift-guide-stacked-wfh_prev_workday.svg

.. image:: ../_static/drift-guide-stacked-workday.svg

NannyML highlights with red the areas with possible data drift.
The ``tenure`` feature has two alerts that are false positives.
The features ``distance_from_office``, ``salary_range``, ``public_transportation_cost``,
``wfh_prev_workday`` have been rightly identified as exhibiting drift.

.. _data-drift-multivariate:

Multivariate Drift Detection
----------------------------

The univariate approach to data drift detection is powerful, interpretable but limited.
Data living in multidimensional spaces can have complex structures
whose change may not be visible by just viewing the distributions of each feature.

One solution for this problem is to user the reconstruction error for a dataset.
In general reconstruction error is the error resulting from re-creating
a dataset after a dimensionality reduction transformation followed by its
inverse transformation. The error is computed to be the mean of the Euclidean distance
of all the points in our dataset. We explain the problem and the reasoning behind
recosntruction error in more detail at
:ref:`Data Reconstruction with PCA Deep Dive<data-reconstruction-pca>`.

For drift detection purposes the key thing to know is that a change in reconstruction error
values reflects a change in the structure of the model inputs. NannyML enables monitoring the
reconstruction error over time for the monitored model and raises an alert if the
values get outside the range observed in the reference partition.

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

NannyML can also visualize multivariate drift results with the following code:

.. code-block:: python

    fig = plots.plot_data_reconstruction_drift(rcerror_results)
    fig.show()

.. image:: ../_static/drift-guide-multivariate.svg

The mutlrivariate drift results provide a consice summary of where data drift
is happening in our input data.
