.. _sum_stats_std:

==================
Standard Deviation
==================


Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cells: 1 3 4 6

Walkthrough
-----------

The Standard Deviation value calculation is straightforward.
For each :term:`chunk<Data Chunk>` NannyML calculates the standard deviation for all selected numerical columns.
The resulting
values from the reference :ref:`data<data-drift-periods>` chunks are used to calculate the
alert :term:`thresholds<Threshold>`. The standard deviation value results from the analysis chunks are
compared against those thresholds and generate alerts if applicable.

We begin by loading the :ref:`synthetic car loan dataset<dataset-synthetic-binary-car-loan>` provided by the NannyML package.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cell: 2

The :class:`~nannyml.stats.std.calculator.SummaryStatsStdCalculator` class implements
the functionality needed for standard deviation values calculations.
We need to instantiate it with appropriate parameters:

- The names of the columns to be evaluated.
- Optionally, the name of the column containing the observation timestamps.
- Optionally, a chunking approach or a predefined chunker. If neither is provided, the default
  chunker creating 10 chunks will be used.
- Optionally, a threshold strategy to modify the default one. See available threshold options
  :ref:`here<thresholds>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cells: 3

Next, the :meth:`~nannyml.base.AbstractCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be
compared with for :term:`alert<Alert>` generation. Then the
:meth:`~nannyml.base.AbstractCalculator.calculate`.calculate` method will
calculate the data quality results on the data provided to it.

The results can be filtered to only include a certain data period, method or column by using the ``filter`` method.
You can evaluate the result data by converting the results into a `DataFrame`,
by calling the :meth:`~nannyml.base.AbstractResult.to_df` method.
By default this will return a `DataFrame` with a multi-level index. The first level represents the column, the second level
represents resulting information such as the data quality metric values, the alert thresholds or the associated sampling error.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cell: 5

More information on accessing the information contained in the
:class:`~nannyml.stats.std.result.Result`
can be found on the :ref:`working_with_results` page.

The next step is visualizing the results, which is done using the
:meth:`~nannyml.stats.std.result.Result.plot` method.
It is recommended to filter results for each column and plot separately.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Std.ipynb
    :cells: 6

.. image:: /_static/tutorials/stats/std-car_value.svg
.. image:: /_static/tutorials/stats/std-debt_to_income_ratio.svg
.. image:: /_static/tutorials/stats/std-driver_tenure.svg

Insights
--------
We see that only the **car_value** column exhibits a permanent change in standard deviation values.
However both **car_value** and **driver_tenure** appear to have one off events where they have some
slightly abnormal values.


What Next
---------

We can also inspect the dataset for other Summary Statistics such as :ref:`sum_stats_avg`.
We can also look for any :term:`Data Drift` present in the dataset using :ref:`data-drift` functionality of
NannyML.
