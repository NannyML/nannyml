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
We currently only support writing data and plots to a local or cloud filesystem.


.. warning::

    This is a very early release and additional ways of outputting data are on their way.
    This configuration section will be prone to big changes in the future.


You can specify the folder to write outputs to using the ``path`` parameter.
The optional ``format`` parameter allows you to choose the format to export the results DataFrames in.
Allowed values are ``csv`` and ``parquet``, with ``parquet`` being the default.


.. code-block:: yaml

    output:
      path: /data/out/
      format: parquet

The output section supports the use of credentials:

.. code-block:: yaml

    output:
        path: s3://nml-data/synthetic_sample_reference.pq
        credentials:  # providing example AWS credentials
          client_kwargs:
            aws_access_key_id: 'ACCESS_KEY_ID'
            aws_secret_access_key: 'SECRET_ACCESS_KEY'


The output format supports passing along any ``pandas.to_csv`` or ``pandas.to_parquet`` using the ``write_args``
parameter.

.. code-block:: yaml

    output:
      path: /data/out/
      format: csv
      write_args:
        headers: False


Column mapping section
***********************

This section is responsible for teaching NannyML about your specific model: what are its features, predictions, ...
You do this by providing a column mapping that associates a NannyML specific meaning to your input data.
For more information on this, check out the :ref:`data_requirements` documentation.

The following snippet lists the column mapping for the :ref:`dataset-synthetic-binary`.

.. code-block:: yaml

    column_mapping:
      features:
        - distance_from_office
        - salary_range
        - gas_price_per_litre
        - public_transportation_cost
        - wfh_prev_workday
        - workday
        - tenure
      timestamp: timestamp
      y_pred: y_pred
      y_pred_proba: y_pred_proba
      y_true: work_home_actual

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

Standalone parameters section
*****************************

This section contains some standalone parameters that mostly serve as an alternative to CLI arguments.

The required `problem_type` variable allows you to pass along a :class:`~nannyml._typing.ProblemType` value.
NannyML uses this information to better understand the provided model inputs and outputs.

.. code-block:: yaml

    problem_type: regression  # pass the problem type (one of 'classification_binary', 'classification_multiclass' or 'regression')


.. code-block:: yaml

    ignore_errors: True  # continue execution of a calculator/estimator fails


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
for the :ref:`dataset-synthetic-binary`. All data is read and written to the local filesystem.

.. code-block:: yaml

    input:
      reference_data:
        path: data/synthetic_sample_reference.csv

      analysis_data:
        path: data/synthetic_sample_analysis.csv

    output:
      path: out/
      format: parquet

    column_mapping:
      features:
        - distance_from_office
        - salary_range
        - gas_price_per_litre
        - public_transportation_cost
        - wfh_prev_workday
        - workday
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
