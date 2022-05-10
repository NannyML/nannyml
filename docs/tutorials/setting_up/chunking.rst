.. _chunking:

======================================
Chunking
======================================

Why do we need chunks?
----------------------

NannyML monitors ML models in production by doing data drift detection and performance estimation or monitoring.
This functionality relies on aggregate metrics that are evaluated on samples of production data.
These samples are called :term:`chunks<Data Chunk>`. All the results generated are
calculated and presented per chunk i.e. a chunk is a single data point on the monitoring results. You
can refer to :ref:`Data Drift guide<data-drift>` or :ref:`Performance Estimation guide<performance-estimation>`
to review example results.



Walkthrough on creating chunks
------------------------------

To allow for flexibility there are many ways to create chunks. The examples provided will explain how chunks are
created depending on the instructions provided. The examples will be run based on the performance estimation flow on
synthetic dataset provided by NannyML. Set up first with:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> reference, analysis, _ = nml.datasets.load_synthetic_sample()
    >>> metadata = nml.extract_metadata(reference, model_type=nml.ModelType.CLASSIFICATION_BINARY)
    >>> metadata.target_column_name = 'work_home_actual'


Time-based chunking
~~~~~~~~~~~~~~~~~~~

Time-based chunking creates chunks based on time intervals. One chunk can contain all the observations
from a single hour, day, week, month etc. In most cases, such chunks will vary in the number of 
observations they contain. Specify the ``chunk_period``
argument to get an appropriate split. See the example below that chunks data quarterly:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_period="Q").fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(analysis)
    >>> est_perf.data.iloc[:3,:5]

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
    Notice that each calendar quarter was taken into account, even if it was not fully covered with records.
    This means some chunks contain fewer observations (usually the last and the first). See the first row above - Q3 is July-September,
    but the first record in the data is from the last day of August. First chunk has ~1.2k of observations while the 2nd
    and 3rd contain above 3k.

Possible time offsets are listed in the table below:

+------------+------------+
| Alias      | Description|
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
~~~~~~~~~~~~~~~~~~~

Chunks can be of fixed size, i.e. each chunk contains the same number of observations. Set this up by specifying
``chunk_size`` parameter:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_size=3500).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(analysis)
    >>> est_perf.data.iloc[:3,:5]

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
    If the number of observations is not divisible by the chunk size required, the number of rows equal to the
    remainder of a division will be dropped. This ensures that each chunk has the same size, but in worst case
    scenario it results in dropping ``chunk_size-1`` rows. Notice that the last index in last chunk is 48999 while
    the last index in raw data is 49999:

    .. code-block:: python

        >>> est_perf.data.iloc[-2:,:5]

    +----+---------------+---------------+-------------+---------------------+---------------------+
    |    | key           |   start_index |   end_index | start_date          | end_date            |
    +====+===============+===============+=============+=====================+=====================+
    | 12 | [42000:45499] |         42000 |       45499 | 2020-06-18 00:00:00 | 2020-09-13 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+
    | 13 | [45500:48999] |         45500 |       48999 | 2020-09-13 00:00:00 | 2020-12-08 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+

    .. code-block:: python

        >>> analysis.index.max()
        49999


Number-based chunking
~~~~~~~~~~~~~~~~~~~~~

The total number of chunks can be set by the ``chunk_number`` parameter:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_number=9).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(analysis)
    >>> len(est_perf.data)
    9

.. note::
    Chunks created this way will be equal in size. If the number of observations is not divisible by the ``chunk_number`` then
    the number of observations equal to the residual of the division will be dropped. See:

    .. code-block:: python

        >>> est_perf.data.iloc[-2:,:5]

    +----+---------------+---------------+-------------+---------------------+---------------------+
    |    | key           |   start_index |   end_index | start_date          | end_date            |
    +====+===============+===============+=============+=====================+=====================+
    |  7 | [38885:44439] |         38885 |       44439 | 2020-04-03 00:00:00 | 2020-08-18 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+
    |  8 | [44440:49994] |         44440 |       49994 | 2020-08-18 00:00:00 | 2021-01-01 23:59:59 |
    +----+---------------+---------------+-------------+---------------------+---------------------+

    .. code-block:: python

        >>> analysis.index.max()
        49999

.. note::
    The same splitting rule is always applied to the dataset used for fitting (``reference``) and the dataset of
    interest (in the presented case - ``analysis``). Unless these two datasets are of the same size, the chunk sizes
    can be considerably different. E.g. if the ``reference`` dataset has 10,000 observations and the ``analysis`` dataset has
    80,000, and chunking is number-based, chunks in ``reference`` will be much small than in ``analysis``.
    Additionally, if the data drift or performance estimation is calculated on
    combined ``reference`` and ``analysis`` the results presented for ``reference`` will be calculated on different
    chunks than they were fitted.

Automatic chunking
~~~~~~~~~~~~~~~~~~

The default chunking method is size-based, with the size being three times the
estimated minimum size for the monitored data and model (see how NannyML estimates minimum chunk size in :ref:`deep
dive<minimum-chunk-size>`):

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(pd.concat([reference, analysis]))
    >>> est_perf.data.iloc[:3,:5]

+----+-------------+---------------+-------------+---------------------+---------------------+
|    | key         |   start_index |   end_index | start_date          | end_date            |
+====+=============+===============+=============+=====================+=====================+
|  0 | [0:899]     |             0 |         899 | 2014-05-09 00:00:00 | 2014-06-01 23:59:59 |
+----+-------------+---------------+-------------+---------------------+---------------------+
|  1 | [900:1799]  |           900 |        1799 | 2014-06-01 00:00:00 | 2014-06-23 23:59:59 |
+----+-------------+---------------+-------------+---------------------+---------------------+
|  2 | [1800:2699] |          1800 |        2699 | 2014-06-23 00:00:00 | 2014-07-15 23:59:59 |
+----+-------------+---------------+-------------+---------------------+---------------------+

Chunks on plots with results
----------------------------

Finally, once the chunking method is selected, the full performance estimation can be run:

    .. code-block:: python

        >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_size=5_000).fit(reference_data=reference)
        >>> est_perf = cbpe.estimate(analysis)
        >>> est_perf.plot(kind='performance').show()

.. image:: /_static/guide-chunking_your_data-pe_plot.svg

Each marker on the plot represents estimated performance (y axis) for a single chunk.
Markers are placed at the end of the period covered by the chunk i.e. they indicate the last timestamp in the
chunk (x axis). Plots are interactive - hovering over the marker will display the information about the period.
