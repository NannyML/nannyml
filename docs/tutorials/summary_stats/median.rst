.. _sum_stats_median:

======
Median
======


Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cells: 1 3 4 6

Walkthrough
-----------

The Median value calculation is straightforward.
For each :term:`chunk<Data Chunk>` NannyML calculates the median for all selected numerical columns.
The resulting
values from the reference :ref:`data<data-drift-periods>` chunks are used to calculate the
alert :term:`thresholds<Threshold>`. The median value results from the monitored chunks are
compared against those thresholds and generate alerts if applicable.

We begin by loading the :ref:`synthetic car loan dataset<dataset-synthetic-binary-car-loan>` provided by the NannyML package.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cell: 2

The :class:`~nannyml.stats.avg.calculator.SummaryStatsMedianCalculator` class implements
the functionality needed for median values calculations.
We need to instantiate it with appropriate parameters:

- **column_names:** A list with the names of columns to be evaluated.
- **timestamp_column_name (Optional):** The name of the column in the reference data that
  contains timestamps.
- **chunk_size (Optional):** The number of observations in each chunk of data
  used. Only one chunking argument needs to be provided. For more information about
  :term:`chunking<Data Chunk>` configurations check out the :ref:`chunking tutorial<chunking>`.
- **chunk_number (Optional):** The number of chunks to be created out of data provided for each
  :ref:`period<data-drift-periods>`.
- **chunk_period (Optional):** The time period based on which we aggregate the provided data in
  order to create chunks.
- **chunker (Optional):** A NannyML :class:`~nannyml.chunk.Chunker` object that will handle the aggregation
  provided data in order to create chunks.
- **threshold (Optional):** The threshold strategy used to calculate the alert threshold limits.
  For more information about thresholds, check out the :ref:`thresholds tutorial<thresholds>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cells: 3

Next, the :meth:`~nannyml.base.AbstractCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the monitored data will be
compared with for :term:`alert<Alert>` generation. Then the
:meth:`~nannyml.base.AbstractCalculator.calculate` method will
calculate the data quality results on the data provided to it.

The results can be filtered to only include a certain data period, method or column by using the ``filter`` method.
You can evaluate the result data by converting the results into a `DataFrame`,
by calling the :meth:`~nannyml.base.AbstractResult.to_df` method.
By default this will return a `DataFrame` with a multi-level index. The first level represents the column, the second level
represents resulting information such as the data quality metric values, the alert thresholds or the associated sampling error.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cell: 5

More information on accessing the information contained in the
:class:`~nannyml.stats.median.result.Result`
can be found on the :ref:`working_with_results` page.

The next step is visualizing the results, which is done using the
:meth:`~nannyml.stats.median.result.Result.plot` method.
It is recommended to filter results for each column and plot separately.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Stats - Median.ipynb
    :cells: 6

.. image:: /_static/tutorials/stats/median-car_value.svg
.. image:: /_static/tutorials/stats/median-debt_to_income_ratio.svg
.. image:: /_static/tutorials/stats/median-driver_tenure.svg

Insights
--------
We see that only the **car_value** column exhibits a change in median value.


What Next
---------

We can also inspect the dataset for other Summary Statistics such as :ref:`sum_stats_std`.
We can also look for any :term:`Data Drift` present in the dataset using :ref:`data-drift` functionality of
NannyML.
