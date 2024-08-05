.. _usage_logging:

==========================
Usage logging in NannyML
==========================

TLDR
----

- We collect anonymous statistics about what functions you use in NannyML
- We won’t collect any information about your identity or your datasets
- We will use this information to improve NannyML and as a way to illustrate traction for our (future) investors
- You can easily opt out, we provide multiple ways to do so
- You can always reach out to us and request any data related to you be removed

What do we mean by usage statistics?
-------------------------------------

We first need to explain what we consider to be**usage statistics**. We collect statistics about the general
usage of the NannyML library. Every time one of our **essential functions** is used a data package will be
shipped to an external service.

The essential functions are the following:

- Fitting any calculator or estimator
- Calculating/estimating using a calculator/estimator
- Plotting results
- Writing results to filesystem, pickle, or database
- Running NannyML using the CLI

The data that is collected and shipped has three different parts:

- **Environment data**: tells us more about the computational environment NannyML is running in.
  This includes figuring out if NannyML is running in a Python application, a notebook, or a container.
  What version of Python is it being used with? What operating system? What version of NannyML is being used?
- **Execution data:** data that helps us understand what functionality of NannyML you're using and how it is performing.
  This is limited to the name of the key function you're running, the metrics or methods you're calculating and the
  time it took NannyML to finish that function call. We’ll also check if any errors occurred during that run.
- **Identification data:** A fingerprint is created based on the hardware present in your machine and used as a
  unique identifier. Running NannyML from the same machine twice means the same unique identifier will be used -
  in theory, this doesn't apply to **Docker**. This allows us to detect repeat usage patterns
  without the need for personal identification.

The following snippet is an actual usage statistic data package sent from NannyML running in a container:

.. code-block:: javascript

    {
      'event': 'DLE estimator plot',
      'exception_occurred': true,
      'exception_type': 'nannyml.exceptions.InvalidArgumentsException',
      'kind': 'performance',
      'nannyml_version': '0.6.3',
      'os_type': 'Linux',
      'process_run_time': 0.00001277399999999318,
      'python_version': '3.8.8',
      'run_time': 0.000012636184692382812,
      'runtime_environment': 'docker',
      'userId': '00000000-0000-0000-0000-0242ac110002'
    }

What about personal data
########################

Apart from the hardware ID, there is nothing to link back to your machine, let alone to your identity.
You have our word on this: we will never collect any Personally Identifiable Information.
And don't just take our word: verify it! We invite you to review the implementation at
https://github.com/NannyML/nannyml/blob/main/nannyml/usage_logging.py.

What about my dataset?
######################

We deliberately avoid logging any arguments when running a key function to minimize the risk of leaking unwanted
and unnecessary information. One exception: the names of **metrics** and **methods** being used in the calculators or estimators.

We collect no information about the structure, size, or contents of your datasets. Your datasets remain yours only.


Why are we doing this?
----------------------

We have good reasons for wanting to collect these usage statistics.

Improving NannyML and prioritizing new features
###############################################

It is an easy claim to make. We are serious about it though. Looking at the aggregate usage statistics can teach us
what kind of functionality is used frequently and if there is functionality not used at all.

It can help us improve the user experience by looking at patterns within the usage events, tackle long processing times,
and help prevent feature-breaking exceptions.

By distributing NannyML as a library to run on your system as opposed to a service hosted by us,
we have no other way to gain these insights.

Surviving as a company
######################

We care about the impact of ML models performing sub-optimally at NannyML. It is our vision that the core functionality
we build, i.e. the algorithms distributed as the NannyML library should be available to everybody, for free, forever.
This was the main driver for building an open-source library. But the world of tech startups has always been a
tough one, and even more so in the last few years.

Because we work in open source, the NannyML library doesn't generate any revenue.
We're depending on external investors to provide us with the resources to continue our work, survive, and maybe even thrive.

We want to know what areas of NannyML we should focus on, and investors want to verify if it is worth
putting their resources into. Aggregate usage analytics provide the actual figures needed to secure funding, as well as motivation.

How usage logging works
-----------------------

We'll give a very brief overview of how we've implemented usage analytics.

1. We've created a `usage_logging` module within the library. It contains all the functionality related to usage analytics.
   Feel free to browse the source code at https://github.com/NannyML/nannyml/blob/main/nannyml/usage_logging.py.
2. We instrument our library by adding a `log_usage` decorator to our key functions, sometimes also providing some additional data (e.g. metric names).
3. Upon calling one of these key functions, the decorator will capture the required information. Our `usage_logging` module will then try to send it over
   to **Segment**, a third-party service provider specializing in customer data.
4. The usage events are aggregated and turned into insights in **Mixpanel**, another third-party service provider
   specializing in self-service product analytics.

.. image:: /_static/usage_logging_how_it_works.png


*To opt in* or *not to opt in*, that's the question
----------------------------------------------------

Whilst our team at NannyML saw the need for usage analytics, we did have some deeper discussions about how to present
this to you, the end user.

Do we disable usage analytics collection by default and have the end user explicitly opt in? 
Whilst it felt very intuitive and "correct” to do so, we asked ourselves the following question.
“Would I go through the trouble of explicitly enabling this every time I use NannyML?".
Our answer was no, we probably wouldn't bother. And if we wouldn't, it is only fair we don't expect you to.

We settled on opt-out behavior, so usage analytics will be enabled by default for the following reasons:

- We don't collect any information that can identify our users
- We don't collect any information about the data NannyML is used on
- We provide an easy way to turn usage analytics off, without any limitations on the product
- We believe that if you keep using NannyML, you probably want us to survive as a company

How to disable usage logging
----------------------------

It should be easy to disable logging. We provide three ways of doing so.
The first way - using environment variables - is universally applicable and easy to set up, so we recommend using that one.

Setting the environment variable
#################################

You can set this variable before running NannyML as a script, CLI command, or container.
Its value doesn’t matter, as long as the environment variable is present.

.. code-block:: bash
    :caption: Disable usage analytics when running a Python script

    NML_DISABLE_USAGE_LOGGING=1 python my_monitoring_script_with_nannyml.py


.. code-block:: bash
    :caption: Disable usage analytics when using the CLI

    NML_DISABLE_USAGE_LOGGING=1 nml -c nann.yml run


.. code-block:: bash
    :caption: Disable usage analytics when using our Docker container

    docker run -e NML_DISABLE_USAGE_LOGGING=1 nannyml/nannyml


Providing a ``.env`` file
#########################

NannyML will check for ``.env``  files, allowing you to provide environment variables without dealing with shells.
Just create a ``.env`` file in the directory of your script and NannyML will pick it up automatically.

.. code-block::
    :caption: A sample ``.env`` file that will disable usage logging

    NML_DISABLE_USAGE_LOGGING=1


Turning off user analytics in code
###################################

If you don't like toying with environment variables, you can just disable (or enable) the usage analytics within your
code before running anything. You can only do this when using NannyML as a library.


.. code-block:: python
    :caption: Disabling usage logging in code

    import nannyml as nml

    nml.disable_usage_logging()  # turn usage analytics off
    nml.enable_usage_logging()  # turn them back on
