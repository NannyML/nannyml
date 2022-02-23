========
Glossary
========

.. glossary::

    Butterfly dataset
        A dataset used in :ref:`<data-reconstruction-pca>` to give an example where univariate
        drift statistics are insufficient in detecting complex data drifts in multidimensional
        data.

    Feature
        A variable used by our machine learning model. A synonym for model input.

    Latent space
        A space of reduced dimensionality, compared to the model input space, that can
        represent our input data. This space is the result of a representation
        learning algorithm.
    
    Ground truth
        A synonym for :term:`Target`

    Identifier
        Usually a single column, but can be multiple columns where necessary. It is used uniquely identify an observation.
        When providing :term:`Target` data at a later point in time, this value can help refer back to the original prediction.

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

    Model
        Definition of a model

    Model inputs
        A variable used by our machine learning model.

    Model outputs
        The scores or probabilities that your model predicts for its target outcome.

    Model predictions
        A synonym for :term:`Model outputs`.


    Multivariate Drift Detection
        Drift Detection steps that involve all the features of our model in order to
        create appropriate drift measures.

    Partition
        A column that tells us what partition the data is in. We will expect data be in one of two partitions.

        The first one is called the ``reference`` partition. It contains all the observations for a period with an *accepted*
        level of performance. It most likely also includes ``target`` data.

        The second partition is the ``analysis`` partition. It contains the observations you want NannyML to analyse.
        It is likely that performance here will be (partially) estimated.

        NannyML needs the partition information to understand which data it can use as a reference to compare other periods by.

        .. warning::
            We currently only support the following partition values: ``reference`` and ``analysis``.

            Please map your own values to them accordingly.

    PCA
        Principal Component Analysis. A method user for dimensionality reduction.

    Predictions
        A synonym for :term:`Model outputs`.
    
    Target
        The actual outcome of the event the machine learning model is trying to predict.

    Timestamp
        Usually a single column, but can be multiple columns where necessary.
        This provides NannyML with the date and time that the prediction was made.

        NannyML need to understand when predictions were made, and how you record this,
        so it can bucket observations in time periods.

        .. note::
            **Format**
                Any format supported by Pandas, most likely:

                - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
                - *Unix-epoch* in units of seconds, e.g. ``1513393355``

    Univariate Drift Detection
        Drift Detection steps that use each feature of our model individually
        in order to create appropriate drift measures.

