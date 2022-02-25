.. _chunk-data:
====
Chunking data
====

.. note::
    Not sure what data chunk is in the first place? Read about :term:`Data Chunk`.

Why we need chunks?
====
NannyML monitors ML model performance and input data changes. Both can be reliably evaluated only on samples
of data containing a number of observations. These samples are called chunks. All the results generated are
calculated and presented on the level of chunk i.e. a chunk is a single data point. Go to
:ref:`Data Drift guide<data-drift>` or :ref:`Performance Estimation guide<performance-estimation>` to see example
results.


Creating chunks
====

The examples below will explain how chunks are created depending on the instructions provided. The examples below
will be run based on performance estimation flow on NannyML synthetic dataset. Set up first with:

.. code-block:: python

    >>> import nannyml as nml
    >>> idf_ref, df_ana, _ = nml.datasets.load_synthetic_sample()
    >>> imd = nml.extract_metadata(df_ref)
    >>> md.ground_truth_column_name = 'work_home_actual'



Time-based chunking
~~~~~
Time-based chunking is simply creating chunks based on time intervals. One chunk can contain all the observations
from single hour, day, week, month etc. In most cases such chunks will vary in length. Specify ``chunk_period`` argument
to get required split. See the
example with
quarters:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=md, chunk_period="Q")
    >>> cbpe.fit(reference_data=df_ref)
    >>> est_perf = cbpe.estimate(df_ana)
    >>> est_perf.iloc[:3,:5]

+----+--------+---------------+-------------+---------------------+---------------------+
|    | key    |   start_index |   end_index | start_date          | end_date            |
+====+========+===============+=============+=====================+=====================+
|  0 | 2017Q3 |             0 |        1261 | 2017-08-31 00:00:00 | 2017-09-30 23:59:59 |
+----+--------+---------------+-------------+---------------------+---------------------+
|  1 | 2017Q4 |          1262 |        4951 | 2017-10-01 00:00:00 | 2017-12-31 23:59:59 |
+----+--------+---------------+-------------+---------------------+---------------------+
|  2 | 2018Q1 |          4952 |        8702 | 2018-01-01 00:00:00 | 2018-03-31 23:59:59 |
+----+--------+---------------+-------------+---------------------+---------------------+

.. note::
    Be aware that each calendar quarter will be taken into account, even if it is not full of records. Make sure to control
    for that while preparing the data. See example below.

    .. code-block:: python

        >>> est_perf.iloc[-2:,:5]

    +----+--------+---------------+-------------+---------------------+---------------------+
    |    | key    |   start_index |   end_index | start_date          | end_date            |
    +====+========+===============+=============+=====================+=====================+
    | 13 | 2020Q4 |         46219 |       49989 | 2020-10-01 00:00:00 | 2020-12-31 23:59:59 |
    +----+--------+---------------+-------------+---------------------+---------------------+
    | 14 | 2021Q1 |         49990 |       49999 | 2021-01-01 00:00:00 | 2021-01-01 23:59:59 |
    +----+--------+---------------+-------------+---------------------+---------------------+

Possible time offsets are listed in the table below:

+------------+------------+
| Header 1   | Header 2   |
+============+============+
| S          | second     |
+------------+------------+
| T, min     | minute     |
+------------+------------+
| H          | hour       |
+------------+------------+
| D          | day        |
+------------+------------+
| W          | week       |
+------------+------------+
| M          | month      |
+------------+------------+
| Q          | quarter    |
+------------+------------+
| A, y       | year       |
+------------+------------+


Size-based chunking
~~~~~
Chunks can be of fixed size i.e. each chunk contains the same number of observations. Set this up by specifying
``chunk_size`` parameter:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=md, chunk_size=3500)
    >>> cbpe.fit(reference_data=df_ref)
    >>> est_perf = cbpe.estimate(df_ana)
    >>> est_perf.iloc[:3,:5]

+----+--------------+---------------+-------------+---------------------+---------------------+
|    | key          |   start_index |   end_index | start_date          | end_date            |
+====+==============+===============+=============+=====================+=====================+
|  0 | [0:3499]     |             0 |        3499 | 2017-08-31 00:00:00 | 2017-11-26 23:59:59 |
+----+--------------+---------------+-------------+---------------------+---------------------+
|  1 | [3500:6999]  |          3500 |        6999 | 2017-11-26 00:00:00 | 2018-02-18 23:59:59 |
+----+--------------+---------------+-------------+---------------------+---------------------+
|  2 | [7000:10499] |          7000 |       10499 | 2018-02-18 00:00:00 | 2018-05-14 23:59:59 |
+----+--------------+---------------+-------------+---------------------+---------------------+


.. note::
    If the number observations is not divisible by the chunk size required, the number of observation equal to the
    reminder of a division will be dropped. This ensures that each chunk has the same size, but in worst case
    scenario it results in dropping ``chunk_size-1`` rows. See:

    .. code-block:: python

        >>> est_perf.iloc[-2:,:5]

    +----+---------------+---------------+-------------+---------------------+---------------------+
    |    | key           |   start_index |   end_index | start_date          | end_date            |
    +====+===============+===============+=============+=====================+=====================+
    | 12 | [42000:45499] |         42000 |       45499 | 2020-06-18 00:00:00 | 2020-09-13 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+
    | 13 | [45500:48999] |         45500 |       48999 | 2020-09-13 00:00:00 | 2020-12-08 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+

    .. code-block:: python

        >>> df_ana.index.max()
        49999


Number-based chunking
~~~~~
The total number of chunks can be fixed by ``chunk_number`` parameter:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=md, chunk_number=9)
    >>> cbpe.fit(reference_data=df_ref)
    >>> est_perf = cbpe.estimate(df_ana)
    >>> len(est_perf)
    >>> 9

.. note::
    Created chunks will be equal in size. If number of observations is not divisible by ``chunk_number`` then the
    number of observations equal to the residual of the division will be dropped. See:

    .. code-block:: python

        >>>> est_perf.iloc[-2:,:5]

    +----+---------------+---------------+-------------+---------------------+---------------------+
    |    | key           |   start_index |   end_index | start_date          | end_date            |
    +====+===============+===============+=============+=====================+=====================+
    |  7 | [38885:44439] |         38885 |       44439 | 2020-04-03 00:00:00 | 2020-08-18 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+
    |  8 | [44440:49994] |         44440 |       49994 | 2020-08-18 00:00:00 | 2021-01-01 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+

.. note::
    The same splitting rule is always applied to the dataset used to fitting (``reference``) and the dataset of
    interest (in the presented case - ``analysis``). Unless these two data sets are of the same size, the chunk sizes
    will be different. Additionally, if the data drift or performance estimation is calculated on concatenated
    ``reference`` and ``analysis`` the results presented for ``reference`` will be calculated on different chunks
    than they were fitted.

Showing chunks on the plots
====
Finally, once the chunking method is selected, the full performance estimation can be run:

    .. code-block:: python

        >>>> cbpe = nml.CBPE(model_metadata=md, chunk_size=5_000)
        >>>> cbpe.fit(reference_data=df_ref)
        >>>> est_perf = cbpe.estimate(df_ana)
        >>>> plots = nml.PerformancePlots(model_metadata=md, chunker=cbpe.chunker)
        >>>> plots.plot_cbpe_performance_estimation(est_perf).show()

.. image:: ../_static/guide-chunking_your_data-pe_plot.svg

# TODO describe chunk boundaries on the plot

Additional considerations
====
Different partitions within one chunk
~~~~~
If you want to get performance estimation or data drift results for a dataset that contains two
partitions - ``reference`` and ``analysis``, most likely there will be a chunk that contains  observations from both of
them. Such chunk will be considered as ``analysis`` chunk, even if only one observation belongs to ``analysis``
observations. See the example:

    .. code-block:: python

        >>>> cbpe = nml.CBPE(model_metadata=md, chunk_number=9)
        >>>> cbpe.fit(reference_data=df_ref)
        >>>> # Estimate on concatenated reference and analysis
        >>>> est_perf = cbpe.estimate(pd.concat([df_ref, df_ana]))
        >>>> est_perf.iloc[3:5,:7]


+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   estimated_roc_auc |
+====+===============+===============+=============+=====================+=====================+=============+=====================+
|  3 | [33333:44443] |         33333 |       44443 | 2016-07-25 00:00:00 | 2017-04-19 23:59:59 | reference   |            0.968876 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+
|  4 | [44444:55554] |         44444 |       55554 | 2017-04-19 00:00:00 | 2018-01-15 23:59:59 | analysis    |            0.968921 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+

    .. code-block:: python

        >>>> df_ref.index.max()
        49999

.. note::
    This is especially important for Performance Estimation where ``reference`` period should be treated like a train
    set is treated when developing ML model whereas ``analysis`` is like test. Performance Estimation on
    ``reference`` will be in most cases much more accurate then on ``analysis``. First chunk of ``analysis`` which
    contains some of the ``reference`` observations will be affected by this. Be aware when interepreting the
    results.


Underpopulated chunks
~~~~~
Depending on the selected chunking method and the provided datasets, some chunks may be very small. In fact, they
might so small that results obtained are governed by noise rather than actual signal. NannyML estimates minimum chunk
size for the monitored data and model provided (see how in :ref:`deep dive<minimum-chunk-size>`). If some of the chunks
created are smaller than the minimum chunk size, a warning will be raised. For example:

    .. code-block:: python

        >>>> cbpe = nml.CBPE(model_metadata=md, chunk_period="Q")
        >>>> cbpe.fit(reference_data=df_ref)
        >>>> est_perf = cbpe.estimate(df_ana)
        UserWarning: The resulting list of chunks contains 1 underpopulated chunks.They contain too few records to be
        statistically relevant and might negatively influence the quality of calculations. Please consider splitting
        your data in a different way or continue at your own risk.

When the warning is about 1 chunk, it is usually the last chunk and this is due to the reasons described in above
sections. When there are more chunks mentioned - the selected splitting method is most likely not suitable.
Investigate that and be aware when analyzing results. See :ref:`deep dive<minimum-chunk-size>` to get a better
understanding.

Not enough chunks
~~~~~
Sometimes selected chunking method may result in not enough chunks being generated in the ``reference``
period. NannyML calculates thresholds based on variability of metrics on ``reference`` chunks (# TODO links here to
either deep dives or guides - depending where we describe thresholds for PE and DD). Having 6 chunks is
absolute minimum (which is still far from being comfortable). If there is less than 6 chunks, warning will be raised:

    .. code-block:: python
        >>>> cbpe = nml.CBPE(model_metadata=md, chunk_number=5)
        >>>> cbpe.fit(reference_data=df_ref)
        >>>> est_perf = cbpe.estimate(df_ana)
        UserWarning: The resulting number of chunks is too low.Please consider splitting your data in a different way or
        continue at your own risk.


