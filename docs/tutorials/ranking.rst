.. _tutorial-ranking:

========
Ranking
========

NannyML uses ranking to order columns in univariate drift results. The resulting order can help prioritize what to investigate further to fully address any issues with the model being monitored.

There are currently two ranking methods in NannyML: alert count and correlation ranking.

Just The Code
=============

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 1 3 5 7 9 11

Walkthrough
===========

Ranking methods use univariate drift calculation results and performance estimation or realized performance
results to rank features.

.. note::
    The univariate drift calculation results need to be created or filtered
    so that there is only one drift method used for each feature. Similarly, the performance estimation
    or realized performance results need to be created or filtered so that only one performance metric
    is present in them.

Below we can see in more details how to use each ranking method.

.. _tutorial-ranking-alert:

Alert Count Ranking
-------------------

Let's look deeper into our first ranking method.
Alert count ranking ranks features according to the number of alerts generated within the ranking period.
It is based on the univariate drift results of the features or data columns considered.

The first thing we need before using the alert count ranker is to create the univariate drift results.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 2

To illustrate the results, we filter and display the analysis period results for **debt_to_income_ratio** feature.
The next step is to instantiate the ranker and instruct it to :meth:`~nannyml.drift.ranker.AlertCountRanker.rank`
the provided results. Notice that the univariate results are filtered to ensure they only have one drift method
per categorical and continuous feature as required.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 3

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 4

The alert count ranker results give a simple and concise view of features that tend to break univariate drift
thresholds more than others.

.. _tutorial-ranking-correlation:

Correlation Ranking
-------------------

Let's continue to the second ranking method. Correlation ranking ranks features according to how much they correlate
to absolute changes in the performance metric selected.

Therefore we first need to create the performance results we will use in our ranking. The estimated
performance results are created below.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 5

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 6

The analysis period estimations are shown.

The realized performance results are also created
since both can be used according to the use case.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 7

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 8

The analysis period results are shown.

We can now proceed to correlation ranking. First, let's correlate drift results with the estimated ``roc_auc``.
A key difference here is that after instantiation, we need to :meth:`~nannyml.drift.ranker.CorrelationRanker.fit`
the ranker with the related results from the reference period and only contain the performance metric we want
the correlation ranker to use. You can read more about why this is needed on the
:ref:`Correlation Ranking, How it Works<how-ranking-correlation>` page.
Then, after fitting, we can :meth:`~nannyml.drift.ranker.CorrelationRanker.rank` providing appropriately
filtered univariate and performance results.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 9

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 10

Depending on circumstances, it may be appropriate to consider the correlation
of drift results on just the analysis dataset or for different metrics.
Below we can see the correlation between the same drift and the ``recall``
results.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cells: 11

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 12

Insights
========

The intended use of ranking results is to suggest prioritization of further investigation of drift results.

If other information is available, such as feature importance, they can also prioritize
which drifted features can be investigated.

What's Next
===========

More information about the specifics of how ranking works can be found on the
:ref:`How it Works, Ranking<how-ranking>` page.
