.. _how-ranking:

========
Ranking
========

As mentioned in the :ref:`ranking tutorial<tutorial-ranking>` NannyML uses ranking to sort columns in
univariate drift results. The resulting order can be helpful in prioritizing what to further investigate
to fully address any issues with the model. Let's deep dive into how ranking options work.

Alert Count Ranking
===================

The alert count ranker is quite simple. First, it uses the provided univariate drift results to count all the alerts a
feature has for the period in question. And then, it ranks the features starting with the one with the highest alert count
and continues in decreasing order.

.. _how-ranking-correlation:

Correlation Ranking
===================

The correlation ranker is a bit more complex. Instead of just looking at the univariate drift results, it also
uses performance results. Those can be either :term:`Estimated Performance` results or :term:`Realized Performance` results.
It then looks for correlation between the univariate drift results and the absolute performance change and ranks
higher the features with higher correlation.

Let's look into detail how correlation ranking works to get a better understanding.
We'll use an example from the :ref:`Correlation Ranking Tutorial<tutorial-ranking-correlation>`:

.. nbimport::
    :path: ./example_notebooks/How it Works - Ranking.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/How it Works - Ranking.ipynb
    :cell: 2


We see that after initializing the :class:`~nannyml.drift.ranker.CorrelationRanker` correlation ranker, the next step is to
:meth:`~nannyml.drift.ranking.CorrelationRanking.fit` it by providing performance results
from the reference :term:`period<Data Period>`. From those results, the ranker calculates
the average performance during the reference period. This value is saved at the ``mean_reference_performance`` property of the ranker.

Then we proceed with the :meth:`~nannyml.drift.ranking.CorrelationRanking.rank` method where we provide
the chosen univariate drift and performance results. The performance results are preprocessed
in order to caclulate the absolute difference of observed performance values with the mean performance
on reference. We can see how this transformation affects the performance values below:

.. nbimport::
    :path: ./example_notebooks/How it Works - Ranking.ipynb
    :cells: 3

.. image:: /_static/how-it-works/ranking-abs-perf.svg

The next step is to calculate the `pearson correlation`_ between the drift results and the calculated
absolute performance changes.

In order to build an intuition about how the pearson correlation ranks
features this way we can compare the drift values of two features,
**wfh_prev_workday** and **gas_price_per_litre** with the absolute performance difference
as shown in this plot below.

.. nbimport::
    :path: ./example_notebooks/How it Works - Ranking.ipynb
    :cells: 4

.. image:: /_static/how-it-works/ranking-abs-perf-features-compare.svg

In the results, the correlation ranker outputs not only the pearson correlation coefficient but
also the associated p-value for testing non-correlation. This is done to help interpret the
results if needed.

.. _`pearson correlation`: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
