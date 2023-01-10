.. _alerting_slack_notification:

==============================================
Setting up Slack notifications for alerts
==============================================

.. warning::
    NannyML currently uses incoming webhooks to integrate with Slack. They are easy to set up and configure:
    just provide a single incoming webhook URL.

    Please follow the appropriate guide in the
    `official Slack documentation <https://slack.com/help/articles/115005265063-Incoming-webhooks-for-Slack>`_
    on how to set up incoming webhooks and obtain the incoming webhook URL.


This page shows you how to set up alert notifications in Slack for NannyML. This can be done as a part of a one-off
script or notebook, but we believe you'll truly benefit from this functionality when you're running NannyML on an
hourly, daily or any scheduled fashion.

This functionality allows you to configure NannyML once and then have it run in the background, running on the most
recent model inputs and outputs, notifying you whenever an alert shows up.

.. note::
    If you want to use NannyML using the CLI or as a container, you'll have to set up alert handling using the
    NannyML configuration file. Check out the :ref:`configuration file documentation<configuration_file_alert_handling>`
    to learn more.


Just The Code
==============

.. nbimport::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cells: 1 3 5 7 8


Walkthrough
===============


This guide uses our own :ref:`synthetic data sets<dataset-synthetic-binary>` for simplicity.

Note that we're only using a single day of analysis data. This serves as a better example of running NannyML every day,
processing the new model inputs and outputs that occurred during the previous day.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cell: 2

All :class:`~nannyml.alerts.base.AlertHandler` instances use the results of a calculator or estimator. Our first step
is to create some calculators and get the results of their calculations.

We'll first create a :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator`, fit it on the
reference data and then calculate some drift metrics on the analysis data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cells: 3

.. nbtable::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cell: 4


We'll do the same for multivariate drift detection. Create a
:class:`~nannyml.drift.multivariate.data_reconstruction.calculator.DataReconstructionDriftCalculator`, fit it on the
reference data and then calculate the reconstruction errors.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cells: 5

.. nbtable::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cell: 6

Now we can create the alert handler. It takes a Slack Incoming Webhook URL as an argument, check out the
`official Slack documentation <https://slack.com/help/articles/115005265063-Incoming-webhooks-for-Slack>`_ to find
out how to get one.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cells: 7

Once the handler is created we can call the
:meth:`~nannyml.alerts.slack_notification_handler.SlackNotificationHandler.handle` method. It takes a list of
calculation results to process and will result in a message on the Slack channel associated with the
Slack Incoming Webhook.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Alert Handling.ipynb
    :cells: 8

The following image shows what the resulting NannyML notification on Slack looks like.

.. image:: /_static/tutorial_alert_handling_slack_notification.png

There is a section corresponding to each of the results: one for the univariate drift results and one for the
multivariate drift results.

A check mark indicates there are no alerts for the multivariate drift calculation.

The warning sign means there were one or multiple alerts for the univariate drift calculation. The notification contains
both a quick summary (in this case the number of drifting columns) and some more details (the list of all drifting
columns).

The notifications for univariate drift calculation alerts will show the columns where an alert occurred using any of
the specified methods. This means a column will be listed here if it triggered one or more alerts.

Since the other calculators and estimators don't work on the level of an individual feature they will show the list of
metrics that caused alerts in stead.

.. important::
    Slack notifications for a calculation result show the following:

    - `univariate drift`: the column names where an alert was triggered using any of the configured drift detection methods
    - `multivariate drift`: the name of the multivariate drift detection methods for which an alert was triggered
    - `realized performance`: the name of the performance metrics for which an alert was triggered
    - `estimated performance (CBPE)`: the name of the performance metrics for which an alert was triggered
    - `estimated performance (DLE)`: the name of the performance metrics for which an alert was triggered


What Next
==========

If you want to enable notifications when running NannyML from the CLI or as a container, you'll have to do the setup
using the configuration file. Check out the appropriate
:ref:`configuration file documentation<configuration_file_alert_handling>` to learn how to.

NannyML offers some other ways to handle alerts. Check out the relevant
:ref:`alert handling documentation<alert-handling>` to learn more.
