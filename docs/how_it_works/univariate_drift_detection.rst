.. _how-it-works-univariate-drift-detection:

Univariate Drift Detection
==========================

Univariate Drift Detection looks at each feature individually and checks whether its
distribution has changed compared to reference data. There are many ways to compare two samples of data and measure
their *similarity*. NannyML provides several methods so that users can choose the one that suits
their data best, the one they are familiar with (and trust) or just use a couple or even all of them to look at
distribution change from all the different perspectives. This page explains on which aspect of the distribution change
each method is focused and in which situations a specific method could be a good choice. Methods are grouped
by the ones applicable to categorical (discrete) and continuous variables. Even if a method can be used for both,
usually the implementation between categorical and continuous is different so it mentioned in both places.

TBD describe why are we looking at statistic/distance first, still having pvalue though.

.. _univariate-drift-detection-continuous-methods:

Methods for Continuous Variables
--------------------------------


Kolmogorov-Smirnov Test
.......................

A two-sample, non-parametric statistical test that compares empirical (i.e. build from the data) Cumulative
Distribution Functions. Test statistic (called D-statistic) is the maximum absolute difference between the two CDFs.
D-statistic is robust, easy to interpret, falls into range 0-1 and is sensitive to changes in both - shape and
location of the empirical distributions. This makes KS test a number one choice for many data distribution monitoring
practitioners. See the image below to get intuition on how value of D-statistic changes with the change of data
distribution.


.. image:: ../_static/how-it-works-univariate-drift-detection-ks.svg
    :width: 400pt


Jensen-Shannon Distance
........................
A square root of Jensen-Shannon Divergence which measures similarity between two probability distributions. It is a
modified version of KL



.. _univariate-drift-detection-categorical-methods:

Methods for Categorical Variables
---------------------------------


Chi-squared Test
................

Jensen-Shanon Divergence
........................
