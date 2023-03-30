.. _quick-start:

==========
Quickstart
==========

----------------
What is NannyML?
----------------

.. include:: ./common/quickstart_what_is_nannyml.rst


.. _walk_through_the_quickstart:



This Quickstart presents some of the core functionalities of NannyML on an example real-world of binary classification
model.

-------------------------------
Exemplary Workflow with NannyML
-------------------------------


Loading data
------------

We will use real-world dataset that contains inputs and predictions of a binary classification model that
predicts whether an individual is employed. Details about the dataset can be found
:ref:`here <dataset-real-world-ma-employment>`.

The data are split into two periods: :ref:`reference <data-drift-periods-reference>` and
:ref:`analysis<data-drift-periods-analysis>`. The reference data is used by
NannyML to establish a baseline for model performance and variable distributions. Your model's test dataset
can serve as the reference data. The analysis data is simply the data you want to analyze i.e. check whether the model
maintains its performance or if feature distributions have shifted etc. This would usually be your latest production data.

Let's load libraries and the data:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 1

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 3

.. nbtable::
    :path: ./example_notebooks/Quickstart.ipynb
    :cell: 4

.. nbtable::
    :path: ./example_notebooks/Quickstart.ipynb
    :cell: 5

Dataframes contain:

- model inputs like ``AGEP`` (person age), ``SCHL`` (education level) etc.
- ``year`` - the year data was gathered, the ``df_reference`` data covers 2015 while ``df_analysis`` ranges 2016-2018.
- ``y_true`` - classification :term:`target<Target>`, **notice that target is not available in** ``df_analysis``.
- ``y_pred_proba`` - analyzed model predicted probability scores.
- ``y_pred`` - analyzed model predictions.


Estimating Performance without Targets
--------------------------------------

ML models are deployed in production on the condition that they produce accurate enough predictions to provide
business value. This condition is evaluated based on unseen data during model development phase.
The main goal of ML model monitoring is thus to continuously verify whether the model maintains its anticipated
performance in production (which is not the case most of the time [1]_).

Monitoring performance is relatively straightforward when :term:`targets<Target>` are available but this is often not
the case.
Labels can be delayed, costly or impossible to get. In such cases, estimating
performance is a good start of the monitoring workflow. NannyML can estimate the performance of an ML model in
production
without access to targets.

Before proceeding, we need to introduce the notion of :term:`data chunks<Data Chunk>`. The natural way of
thinking
about
monitoring model in production is time-related. Predictions are always made at some point in time and thus have a natural order.
However assessing the model's condition based on a single observation is unreliable
- some performance metrics cannot be even calculated for a single data point. Therefore, we will group observations into
chunks based on their order of occurrence. Let's define the size of the chunk that we will use throughout the
whole analysis:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 6

For :ref:`binary classification model performance estimation<binary-performance-estimation>` we will use
:class:`~nannyml.performance_estimation
.confidence_based
.cbpe.CBPE` class (:ref:`Confidence-based Performance Estimation
<how-it-works-cbpe>`) to estimate ``roc_auc`` metric. Let's initialize the estimator and provide the required
parameters:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 7

Now we will fit it on ``df_reference`` and estimate on ``df_analysis``:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 8

Let's visualize the results:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 9

.. image:: ./_static/quick-start-perf-est.svg

We should take note of the significant drop in estimated performance during the latter part of the analysis period.
Let's investigate this to determine whether we can rely on the estimation.

Investigating Data Distribution Shifts
--------------------------------------

While :ref:`developing the monitored model<dataset-real-world-ma-employment>` we discovered that the primary predictors
for this problem are the first two features, namely `AGEP` (person's age) and `SCHL` (education level). Focusing on
these features, we will employ :ref:`the univariate drift detection module<_univariate_drift_detection>` to examine the distribution behavior of
these two variables. We will instantiate the :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator`
class with
required parameters, fit on ``df_reference`` and calculate on
``df_analysis``.

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 11, 12

.. image:: ./_static/quick-start-drift.svg

Plots show JS-distance calculated between the chunk of interest and the reference data for each feature. For `AGEP`
one can notice mild shift starting in around one-third of the analysis period and a high peak that likely corresponds
to performance drop. Around the same time a similar peak can be notice for `SCHL`. Let's check whether the shift
happens at the same time as the performance drop by :ref:`showing both results in single plot<compare_estimated_and_realized_performance>`:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 14

.. image:: ./_static/quick-start-drift-n-performance.svg

Plot confirms our supposition: the main drift peak coincides with the strongest performance drop. It is interesting
to see that there is a noticeable shift magnitude increase right before the estimated drop happens. That could looks
like an early sign of incoming issues. Now let's see what
actually happened with the distributions by visualizing their change in the analysis period:

.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 16

.. image:: ./_static/quick-start-univariate-distribution.svg

The age distribution has strongly shifted towards younger people (around 18 years old). In the education level
feature one of the categories has doubled its relative frequency in the performance drop chunks. Since plots in the
notebook are interactive (thery're not in the docs though) they allow for value checking by hovering over the corresponding
sections in the stacked-bar plot. The category of interest is encoded :ref:`with value 19<dataset-real-world-ma-employment>`, which
corresponds to people with
*1 or more years of college credit, no degree*. It is likely that during the investigated period, there was a
significant survey conducted at colleges and universities.


Comparing Estimation with Realized Performance when Targets Arrive
------------------------------------------------------------------

The above findings enhance trust in the estimation. Once the labels are in place, we can :ref:`calculate performance<performance-calculation>`
and
compare with the estimation to verify its accuracy. We will use :class:`~nannyml.performance_calculation.calculator
.PerformanceCalculator`
and follow the familiar pattern: initialize, fit and calculate. Then we will plot the comparison:


.. nbimport::
    :path: ./example_notebooks/Quickstart.ipynb
    :cells: 21

.. image:: ./_static/quick-start-estimated-and-realized.svg

Even though the estimation is somewhat off, we see that the realized performance has indeed sharply dropped in the
two indicated chunks. It is also interesting to notice that the performance was relatively stable in the preceding
period even though ``AGEP`` was already slightly shifted at that time. This confirms the need to monitor
performance/estimated performance as not every shift impacts performance.


------------
What's next?
------------

This Quickstart presents some of the core functionalities of NannyML on an example of real-world binary classification
data. The walk through is concise to help you getting familiar with fundamental concepts and structure of the
library. NannyML provides other useful functionalities (like well-received :ref:`multivariate drift
detection<multivariate_drift_detection>`) that
can help you monitor your production models comprehensively. All :ref:`our tutorials<tutorials>` are a good place to start exploring them.

If you want to know what is implemented under the hood - visit :ref:`how it works<how_it_works>`. Finally, if you just look for examples
on other datasets or ML problems look through our :ref:`examples<examples>`.


**References**

.. [1] https://www.nature.com/articles/s41598-022-15245-z
