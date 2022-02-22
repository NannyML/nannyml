========
Glossary
========

.. glossary::

    feature
        A variable used by our machine learning model. A synonym for model input.

    Latent space
        A space of reduced dimensionality, compared to the model input space, that can
        represent our input data. This space is the result of a representation
        learning algorithm.

    Model
        Definition of a model

    Model inputs
        A variable used by our machine learning model.

    Model output
        The score or probability that our model provides.

    Multivariate Drift Detection
        Drift Detection steps that involve all the features of our model in order to
        create appropriate drift measures.

    PCA
        Principal Component Analysis. A method user for dimensionality reduction.

    Prediction
        A synonym for model output.

    Univariate Drift Detection
        Drift Detection steps that use each feature of our model individually
        in order to create appropriate drift measures.

..
    Feature input values
    --------------------

    Multiple columns of data, each one containing a different feature used by your model to make its predictions.
    Each column contains real (or realistically fake) data.
    NannyML can use a subset or all of them in its calculations.

    We need this so that NannyML can read the kind of data each feature uses,
    and monitor for changes to types, rates and distributions.

    Model predictions
    -----------------

    A single column that contains the prediction of your model, given the input values for that row.

    NannyML needs a sample of these to know what your model is producing as output.
    It can then monitor it in the future.

    Timestamp
    ---------

    Usually a single column, but can be multiple columns where necessary.
    This provides NannyML with the date and time that the prediction was made.

    NannyML need to understand when predictions were made, and how you record this, so it can bucket
    observations in time periods.

    .. note::
        **Format**
            Any format supported by Pandas, most likely:

            - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
            - *Unix-epoch* in units of seconds, e.g. ``1513393355``


    Identifier
    ----------

    Usually a single column, but can be multiple columns where necessary. It is used uniquely identify an observation.
    When providing ``target`` data at a later point in time, this value can help refer back to the original prediction.

    Being able to uniquely identify each row of data can help reference any particular issues NannyML might identify
    and make resolving issues easier for you. As we add functionality to provide ``target`` data afterwards your data
    will already be in the correct shape to support it!

    .. note::
        **Format**
            No specific format. Any str or int value is possible.

        **Candidates**
            An existing identifier from your business case.
            A technical identifier such as a globally unique identifier (GUID).
            A hash of some (or all) of your column values, using a hashing function with appropriate collision properties, e.g. the SHA-2 and SHA-3 families.
            A concatenation of your dataset name and a row number.


    Target
    ------

    The ``target`` (sometimes also called ``actual`` or ``ground truth``) is the actual outcome of the event you're trying
    to predict. This data might be available with some delay as it takes time to gather feedback in production systems or
    it might only be available in training data sets and not at all for production data.

    NannyML can use this information to analyze the performance of your model over time, and provide insights into the
    correlation between model performance and data or concept drift.
    When ground truth is available the performance will be *calculated*, otherwise it will be *estimated*.

    Partition
    ---------

    A column that tells us what partition the data is in. We will expect data be in one of two partitions.

    The first one is called the ``reference`` partition. It contains all the observations for a period with an *accepted*
    level of performance. It most likely also includes ``target`` data.

    The second partition is the ``analysis`` partition. It contains the observations you want NannyML to analyse.
    It is likely that performance here will be (partially) estimated.

    NannyML needs the partition information to understand which data it can use as a reference to compare other periods by.

    .. warning::
        We currently only support the following partition values: ``reference`` and ``analysis``.

        Please map your own values to them accordingly.
