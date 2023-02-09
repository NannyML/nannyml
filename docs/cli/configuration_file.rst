.. _cli_configuration:

=======================================================
Configuration file
=======================================================


.. _cli_configuration_location:

Locations
---------------------------------------------

The ``nml`` CLI will look for configuration files called either ``nannyml.yaml`` or ``nann.yml`` in a number of
preset locations. These presets can be overridden by telling NannyML where to look for your configuration.
You can do this by using an environment variable or a command line argument.

The ``nml`` CLI will go over the possible options in the following order:

#. Evaluate the ``-c`` or ``--configuration-path`` command line argument

    When providing an explicit location to the ``nml`` CLI, the configuration file living at that location will always
    be prioritised above any preset location.

    .. code-block:: bash

        nml -c /path/to/nann.yml run

#. Evaluate the ``NML_CONFIG_PATH`` environment variable

    If the ``NML_CONFIG_PATH`` environment variable was found, its value will be interpreted as a path pointing to
    your config file.

    .. code-block:: bash

        export NML_CONFIG_PATH /path/to/nann.yml

#. Look for ``nannyml.yaml`` or ``nann.yml`` in the ``/config`` directory

    This directory is unlikely to exist on your local system, but is easy to use when mounting your configuration
    files into the `NannyML docker container <https://hub.docker.com/repository/docker/nannyml/nannyml>`_.

    .. code-block:: bash

        docker run -v /path/to/config/dir/:/config/ nannyml/nannyml nml run

#. Look for ``nannyml.yaml`` or ``nann.yml`` in the current working ``$PWD`` directory

    When working on your local system you can just run the ``nml`` CLI in the same location as your ``nannyml.yaml``
    or ``nann.yml`` file. Make sure you've activated your virtual environment when using one!

.. _cli_configuration_format:

Format
---------------------------------------------

The configuration file format is broken down into multiple sections.

Input section
*************

This section describes the input data for NannyML, i.e. the ``reference`` and ``analysis`` datasets.

The following snippet shows the basic form of pointing towards two local CSV files as reference and analysis data.

.. code-block:: yaml

    input:
      reference_data:
        path: /data/synthetic_sample_reference.csv

      analysis_data:
        path: /data/synthetic_sample_analysis.csv

You can also work with data living in cloud storage. We currently support reading data from S3 buckets
(Amazon Web Services), GCS buckets (Google Cloud Platform) and ADLS or Azure Blob Storage (Microsoft Azure).
We use the awesome `fsspec <https://github.com/fsspec>`_ project for this.

You can provide credentials to access these locations by using cloud-vendor specific way (e.g. setting some environment
variables or providing config files like ``.aws/credentials``) or provide them in the configuration.

.. code-block:: yaml

    input:
      reference_data:
        path: s3://nml-data/synthetic_sample_reference.pq
        credentials:  # providing example AWS credentials
          client_kwargs:
            aws_access_key_id: 'ACCESS_KEY_ID'
            aws_secret_access_key: 'SECRET_ACCESS_KEY'

      analysis_data:
        path: gs://nml-data/synthetic_sample_analysis.pq
        credentials:  # providing example GCP credentials
            token: ml6-workshop-fa83b3d60b5d.json  # path to service account key file


Any ``pandas.read_csv`` or ``pandas.read_parquet`` options can be passed along by providing them in the configuration
using the ``read_args`` parameter.

.. code-block:: yaml

    input:
      reference_data:
        path: /data/synthetic_sample_reference.csv
        read_args:
            delimiter: ;
            chunksize: 100000


When target values are delivered separately you can specify these as an input as well. You must also provide a column
used to join your target values with your analysis data.

.. code-block:: yaml

    input:
      reference_data:
        path: /data/synthetic_sample_reference.csv

      analysis_data:
        path: /data/synthetic_sample_analysis.csv

      target_data:
        path: /data/synthetic_sample_analysis_gt.csv
        join_column: identifier



Output section
**************

The output section allows you to instruct NannyML on how and where to write the outputs of the calculations.
We currently support writing data and plots to a local or cloud filesystem or exporting data to a relational database.


.. warning::

    This is a very early release and additional ways of outputting data are on their way.
    This configuration section will be prone to big changes in the future.


Writing to filesystem
""""""""""""""""""""""

You can specify the folder to write outputs to using the ``path`` parameter.
The optional ``format`` parameter allows you to choose the format to export the results DataFrames in.
Allowed values are ``csv`` and ``parquet``, with ``parquet`` being the default.


.. code-block:: yaml

    output:
      raw_files:
        path: /data/out/
        format: parquet

The output section supports the use of credentials:

.. code-block:: yaml

    output:
      raw_files:
        path: s3://nml-data/synthetic_sample_reference.pq
        credentials:  # providing example AWS credentials
          client_kwargs:
            aws_access_key_id: 'ACCESS_KEY_ID'
            aws_secret_access_key: 'SECRET_ACCESS_KEY'


The output format supports passing along any ``pandas.to_csv`` or ``pandas.to_parquet`` using the ``write_args``
parameter.

.. code-block:: yaml

    output:
      raw_files:
        path: /data/out/
        format: csv
        write_args:
          headers: False


Writing to a pickle file
"""""""""""""""""""""""""

NannyML supports directly pickling the ``Result`` objects returned by calculators and estimators.
Use the following configuration to enable this:

.. code-block:: yaml

    output:
      pickle:
        path: /data/out/  # a *.pkl file will be written here by each calculator/estimator


Writing to a relational database
""""""""""""""""""""""""""""""""

NannyML can also export its data to a relational database. When provided with a connection string NannyML will create
the required table structure and insert calculator and estimator results in there.

.. warning::

    Your data must contain a :term:`timestamp<Timestamp>` column in order to use this functionality.


There is a separate table for each calculator and estimator. The following sample from the `cbpe_performance_metrics`
table illustrates their overall structure:

+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| id  | model_id  | run_id  | timestamp                   | metric_name  | value               | alert  |
+=====+===========+=========+=============================+==============+=====================+========+
| 1   | 2         | 4       | 2014-05-09 12:00:00.000000  | ROC AUC      | 0.9395984406102346  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 2   | 2         | 4       | 2014-05-10 12:00:00.000000  | ROC AUC      | 0.9669333004887973  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 3   | 2         | 4       | 2014-05-11 12:00:00.000000  | ROC AUC      | 0.9616566861394408  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 4   | 2         | 4       | 2014-05-12 12:00:00.000000  | ROC AUC      | 0.9631921191605108  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 5   | 2         | 4       | 2014-05-13 12:00:00.000000  | ROC AUC      | 0.9679918198658687  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 6   | 2         | 4       | 2014-05-14 12:00:00.000000  | ROC AUC      | 0.9680751598579069  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 7   | 2         | 4       | 2014-05-15 12:00:00.000000  | ROC AUC      | 0.9593668335222013  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 8   | 2         | 4       | 2014-05-16 12:00:00.000000  | ROC AUC      | 0.964513389926401   | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+
| 9   | 2         | 4       | 2014-05-17 12:00:00.000000  | ROC AUC      | 0.9674120045991212  | false  |
+-----+-----------+---------+-----------------------------+--------------+---------------------+--------+

- **id** is the database primary (technical) key, uniquely identifying each row.

- **model_id** is a foreign key to the `model` table. It currently only contains a name for a model
  but having this allows you to filter on a model when performing queries or visualizing in dashboards.

- **run_id** is a foreign key to the `run` table. It contains information about how and when NannyML was run.
  It also serves to filter metrics that were inserted during a given run, allowing you to easily remove these in case of errors.

- **timestamp** is a :term:`timestamp<Timestamp>` created by finding the middle point of the start and
  end timestamps for each :term:`chunk<Data Chunk>`. E.g. for a chunk starting at midnight and ending just before
  midnight of that day, the generated timestamp will be at noon.

- **metric_name** is a column specific to some calculators and estimators. It contains the name of the metric
  that's being calculated or estimated.

- **value** contains the actual value that was being calculated. This might be a realized or estimated performance
  metric or a drift metric.

- **alert** contains a boolean value (``true`` or ``false``) indicating whether the metric crossed a threshold,
  thus raising an alert.

- **upper_threshold** contains the value of the upper threshold for the metric.
  Exceeding this value results in an alert.

- **lower_threshold** contains the value of the lower threshold for the metric.
  Diving under this value results in an alert.

- **feature_name** is not listed here but is present in univariate calculator results. It contains the name of the
  feature the metric value belongs to.

We currently support all databases supported by SQLAlchemy. You can find more information on the required
connection strings in their `Engine Configuration <https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls>`_.
The following snippet illustrates how to configure the database export to a Postgres database running locally.

.. code-block:: yaml

    output:
      database:
        connection_string: postgresql://postgres:mysecretpassword@localhost:5432/postgres
        model_name: my regression model


Note the presence of the ``model_name`` value. It will ensure an entry for the given name is present in the `model`
table (by either retrieving or creating it) and link it to the metrics using the ``model.id`` value as a foreign key.
This configuration is optional but recommended. Dropping this parameter results in the metrics being written without
a ``model_id`` value, which makes them harder to link to a single given model.


Column mapping section
***********************

This section is responsible for teaching NannyML about your specific model: what are its features, predictions, ...
You do this by providing a column mapping that associates a NannyML specific meaning to your input data.
For more information on this, check out the :ref:`data_requirements` documentation.

The following snippet lists the column mapping for the :ref:`dataset-synthetic-binary-car-loan`.

.. code-block:: yaml

    column_mapping:
      features:
        - car_value
        - salary_range
        - debt_to_income_ratio
        - loan_length
        - repaid_loan_on_prev_car
        - size_of_downpayment
        - tenure
      timestamp: timestamp
      y_pred: y_pred
      y_pred_proba: y_pred_proba
      y_true: repaid

This snippet shows how to setup the column mapping for the :ref:`dataset-synthetic-multiclass`.

.. code-block:: yaml

    column_mapping:
      features:
        - acq_channel
        - app_behavioral_score
        - requested_credit_limit
        - app_channel
        - credit_bureau_score
        - stated_income
        - is_customer
      timestamp: timestamp
      y_pred: y_pred
      y_pred_proba:
        prepaid_card: y_pred_proba_prepaid_card
        highstreet_card: y_pred_proba_highstreet_card
        upmarket_card: y_pred_proba_upmarket_card
      y_true: y_true

.. _cli_configuration_store:

Store section
*****************

This section lets you set up a :class:`~nannyml.io.store.file_store.FilesystemStore` for caching purposes.

When a :class:`~nannyml.io.store.file_store.FilesystemStore` is configured it will be used to store and load fitted
calculators during the run. NannyML will use the store to try to load pre-fitted calculators. If none can be found
a new calculator will be created, fitted and persisted using the store.
The next time NannyML is run using the same configuration file it will find the stored calculator and use it subsequently.

Check out the :ref:`tutorial on storing and loading calculators<storing_and_loading_calculators>` to learn more.

This snippet shows how to setup the store in configuration using the local filesystem:

.. code-block:: yaml

    store:
      file:
        path: /out/nml-cache/calculators

This snippet shows how use S3:

.. code-block:: yaml

    store:
      file:
        path: s3://my-bucket/nml/cache/
        credentials:
          client_kwargs:
            aws_access_key_id: '<ACCESS_KEY_ID>'
            aws_secret_access_key: '<SECRET_ACCESS_KEY>'

This snippet shows how to use Google Cloud Storage:

.. code-block:: yaml

    store:
      file:
        path: gs://my-bucket/nml/cache/
        credentials:
            token: service-account-access-key.json

This snippet shows how to use Azure Blob Storage:

.. code-block:: yaml

    store:
      file:
        path: abfs://my-bucket/nml/cache/
        credentials:
            account_name: '<ACCOUNT_NAME>'
            account_key: '<ACCOUNT_KEY>'


Chunker section
*****************

The chunker section allows you to set the chunking behavior for all of the calculators and estimators that will be run.
Check the :ref:`chunking` documentation for more information on the practice of chunking and the available ``Chunkers``.

This section is optional and when it is absent NannyML will use a :class:`~nannyml.chunking.DefaultChunker` instead.

.. code-block:: yaml

    chunker:
      chunk_size: 5000  # chunks of fixed size


.. code-block:: yaml

    chunker:
      chunk_period: W  # chunks grouping observations by week


Scheduling section
*******************

The scheduling section allows you to configure the schedule NannyML is to run on. This section is optional and if none
is found NannyML will just run a single time, unscheduled.

There are currently two ways of scheduling in NannyML.

- **Interval** scheduling allows you to set the interval between NannyML runs, such as *every 6 hours* or *every 3 days*.
  The available time increments are ``weeks``, ``days``, ``hours`` and ``minutes``.
- **Cron** scheduling allows you to leverage the widely known ``crontab`` expressions to control scheduling.

.. code-block:: yaml
    :caption: Interval based scheduling configuration

    scheduling:
      interval:
        days: 1  # wait one day from the timestamp at which the command is run


.. code-block:: yaml
    :caption: ``cron`` based scheduling configuration

    scheduling:
      cron:
        crontab: "*/5 * * * *" # every 5 minutes, so on 00:05, 00:10, 00:15, ...


Standalone parameters section
*****************************

This section contains some standalone parameters that mostly serve as an alternative to CLI arguments.

The required `problem_type` variable allows you to pass along a :class:`~nannyml._typing.ProblemType` value.
NannyML uses this information to better understand the provided model inputs and outputs.

.. code-block:: yaml

    problem_type: regression  # pass the problem type (one of 'classification_binary', 'classification_multiclass' or 'regression')


.. code-block:: yaml

    ignore_errors: True  # continue execution if a calculator/estimator fails


Templating paths
--------------------------------------------

To use NannyML as a scheduled job we provide some support for path templating. This allows you to read data from and
write data to locations that are based on timestamps.

The following example illustrates writing outputs to a 3-tiered directory structure for years, months and days.
When NannyML is run as a daily scheduled job the results will be written to a different folder each day, preserving
the outputs of previous runs.

.. code-block:: yaml

    output:
      path: /data/out/{{year}}/{{month}}/{{day}}


The following placeholders are currently supported:

- ``minute``
- ``hour``
- ``day``
- ``weeknumber``
- ``month``
- ``year``


Examples
--------------------------------------------

The following example contains the configuration required to run the ``nml`` CLI
for the :ref:`dataset-synthetic-binary-car-loan`.

All data is read and written to the local filesystem.

.. code-block:: yaml

    input:
      reference_data:
        path: data/synthetic_sample_reference.csv

      analysis_data:
        path: data/synthetic_sample_analysis.csv

    output:
      raw_files:
        path: out/
        format: parquet

    column_mapping:
      features:
        - car_value
        - salary_range
        - debt_to_income_ratio
        - loan_length
        - repaid_loan_on_prev_car
        - size_of_downpayment
        - tenure
      timestamp: timestamp
      y_pred: y_pred
      y_pred_proba: y_pred_proba
      y_true: work_home_actual

    problem_type: classification_binary

    ignore_errors: True


The following example contains the configuration used to run the ``nml`` CLI on the :ref:`dataset-synthetic-multiclass`.

Input data is read from one S3 bucket using templated paths.
Targets have been provided separately - they are not present in the analysis data.
The results are written to another S3 bucket, also using a templated path.


.. code-block:: yaml

    input:
      reference_data:
        path: s3://nml-data/{{year}}/{{month}}/{{day}}/mc_reference.csv
        credentials:
          client_kwargs:
            aws_access_key_id: 'DATA_ACCESS_KEY_ID'
            aws_secret_access_key: 'DATA_SECRET_ACCESS_KEY'

      analysis_data:
        path: s3://nml-data/{{year}}/{{month}}/{{day}}/mc_analysis.csv
        credentials:
          client_kwargs:
            aws_access_key_id: 'DATA_ACCESS_KEY_ID'
            aws_secret_access_key: 'DATA_SECRET_ACCESS_KEY'

      target_data:
        path: s3://nml-data/{{year}}/{{month}}/{{day}}/mc_analysis.csv
        join_column: identifier
        credentials:
          client_kwargs:
            aws_access_key_id: 'DATA_ACCESS_KEY_ID'
            aws_secret_access_key: 'DATA_SECRET_ACCESS_KEY'

    output:
      raw_files:
        path: s3://nml-results/{{year}}/{{month}}/{{day}}
        format: parquet
        credentials:  # different credentials
            client_kwargs:
              aws_access_key_id: 'RESULTS_ACCESS_KEY_ID'
              aws_secret_access_key: 'RESULTS_SECRET_ACCESS_KEY'

    chunker:
      chunk_size: 5000

    column_mapping:
      features:
        - acq_channel
        - app_behavioral_score
        - requested_credit_limit
        - app_channel
        - credit_bureau_score
        - stated_income
        - is_customer
      timestamp: timestamp
      y_pred: y_pred
      y_pred_proba:
        prepaid_card: y_pred_proba_prepaid_card
        highstreet_card: y_pred_proba_highstreet_card
        upmarket_card: y_pred_proba_upmarket_card
      y_true: y_true

    problem_type: classification_multiclass

    ignore_errors: False


The following example contains the configuration required to run the ``nml`` CLI
for the :ref:`dataset-synthetic-regression`.

The data is read from the local filesystem but written to an external database.

.. code-block:: yaml

    input:
      reference_data:
        path: data/regression_synthetic_reference.csv

      analysis_data:
        path: data/regression_synthetic_analysis.csv

      target_data:
        path: data/regression_synthetic_analysis_targets.csv

    output:
      database:
        connection_string: postgresql://postgres:mysecretpassword@localhost:5432/postgres
        model_name: regression_car_price

    problem_type: regression

    chunker:
      chunk_period: D

    column_mapping:
      features:
        - car_age
        - km_driven
        - price_new
        - accident_count
        - door_count
        - transmission
        - fuel
      timestamp: timestamp
      y_pred: y_pred
      y_true: y_true
