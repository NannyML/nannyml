.. _drift_detection_for_regression_model_targets:

=======================================================
Drift Detection for Regression Model Targets
=======================================================

Why Perform Drift Detection for Model Targets
---------------------------------------------

The performance of a machine learning model can be affected if the distribution of targets changes.
The target distribution can change both because of data drift but also because of label shift.

A change in the target distribution may mean that business assumptions on which the model is
used may need to be revisited.

NannyML uses :class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
in order to monitor drift in the :term:`Target` distribution. It can calculate the KS
statistic (from the :term:`Kolmogorov-Smirnov test`) for aggregated drift results
but also show the target distribution results per chunk with joyploys.

.. note::
    The Target Drift detection process can handle missing target values across all :term:`data periods<Data Period>`.


Just The Code
-------------

.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 1 3 4 6 8 10


Walkthrough
-----------

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some :ref:`synthetic car pricing data<dataset-synthetic-regression>` provided by the NannyML package, and setting it up as our reference and analysis dataframes.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation<performance-estimation>`. But it is required to detect drift for the targets, so the first thing we need to in this case is set up the right data in the right dataframes.
The analysis target values are expected to be ordered correctly, just like in sklearn.

.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 1

.. nbtable::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cell: 2

Now that the data is in place we'll create a new
:class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
instantiating it with the appropriate parameters. We need the name for the target, ``y_true``, and the timestamp columns.
We also need to specify the machine learning problem we are working on.

.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 3

Afterwards, the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.fit`
method gets called on the reference :term:`period<Data Period>`, which represent an accepted target distribution
which we will compare against the analysis :term:`period<Data Period>`.

Then the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.calculate` method is
called to calculate the target drift results on the data provided. We use the previously assembled data as an argument.

We can display the results of this calculation in a dataframe.

.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 4

.. nbtable::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cell: 5

We can also display the results from the reference dataframe.

.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 6

.. nbtable::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cell: 7

The results can be also easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.
We first plot the KS Statistic drift results for each chunk.

.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 8

Note that a dashed line, instead of a solid line, will be used for chunks that have missing target values.

.. image:: /_static/tutorials/detecting_data_drift/model_targets/regression/target-drift.svg

And then we create the joyplot to visualize the target distribution values for each chunk.


.. nbimport::
    :path: ./_build/notebooks/Tutorial - Model Targets - Regression.ipynb
    :cells: 10

.. image:: /_static/tutorials/detecting_data_drift/model_targets/regression/target-distribution.svg


Insights
--------

Looking at the results we can see that there has been some target drift towards lower car prices.
We should also check to see if the performance of our model has been affected through
:ref:`realized performance monitoring<regression-performance-calculation>`.
Lastly we would need to check with the business stakeholders to see if the changes observed can affect the company's
sales and marketing policies.


What Next
---------

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes. Moreover the :ref:`Univariate Drift Detection<univariate_drift_detection>`
as well as the :ref:`Multivariate Drift Detection<multivariate_drift_detection>` can add further context if needed.
