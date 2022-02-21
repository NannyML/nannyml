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

In general a machine learning model operating on an input distribution different than
the one it has been trained on will underperform. It is therefore crucial to detect the
presense of data drift when we have a model in production. By further investigating the
characteristics of the observed drift we will be in a position to estimate the impact
of the drift to the model's performance.

Let us note here that data drift is not the only change that can happen when we have a
machine learning model in production. Another change we can have is concept drift. In this
case the distribution of our input data stays the same but the relationship between our outcome
changes. In this case we have: :math:`P(y'|\mathbf{X'}) \neq P(y|\mathbf{X})` while
:math:`P(\mathbf{X'}) = P(\mathbf{X})`.

There is also a special case of data drift called label shift. In this case the outcome
distributions between our training and production data are different, meaning
:math:`P(y') \neq P(y)`. However, the relationship between our population characteristics and
a specific outcome does not change, namely :math:`P(\mathbf{X'}|y') = P(\mathbf{X}|y)`.


Data Partitions
================

As can be seen from our data drift discussion earlier before we can start talking about data drift
we need to have two different datasets to compare. NannyML uses the reference partition and the
analysis partition for this purpose.

Reference Partition
-------------------

The reference partition's purpose is to serve as a dataset suitable for our machine learning model.
We also assume that the performance results of our model are available for this dataset and that it
has acceptable performance.

The reference dataset can be the test set we used when evaluating our model before
we deploy it to production. Alternatively it can be a reference (or benchmark) period when our
model is in production and its performance results where satisfactory.

Analysis Partition
------------------

The analysis partition's purpose is the dataset where we want to examine the performance of our
model. This will usually consist of the latest production data up to a desired point in the past
after the point where our reference partition ends. The analysis partition is not required to have
ground truth and associated performance results available.

As part of our data drift analysis will will compare periods of the analysis partition, which we
call chunks internaly in NannyML, with the reference data. Any meaningful differences will be
flagged as data drift.


Data Drift in practice
======================

NannyML uses two approaches to detect and investigate data drift. A Univariate approach and a
Multivariate approach.

Univariate Drift Detection
--------------------------

The Univariate approach looks at each variable individually and conducts statistical tests comparing
the chunks created from the datasets with the reference dataset. For continuous features we use the
KS Test and for categorical features we use the 2 sample Chi squared test.

    Add references to code docs ...


Multivariate Drift Detection
----------------------------

- Univariate Changes in the data distributions
    - We use statistical tests to detect and measure changes

- Multivariate changes …
    - Multidimensional data can change in ways that are not obvious from univariate views
    - We use reconstruction error to detect them
