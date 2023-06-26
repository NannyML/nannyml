.. _glossary:

########
Glossary
########

.. glossary::

    Alert
        An alert is an indication of whether a particular statistic calculated by NannyML is
        abnormal and possibly warrants further investigation. During data quality, drift or
        performance calculations lower and upper :term:`thresholds<Threshold>` can be specified to restrain the
        expected range of the metric being calculated or estimated.
        An alert is raised after NannyML finds the calculated metric outside of the specified range.

        Note that alerts are not raised during the reference :term:`Data Period`.

    Business Value Matrix
        A matrix that is used to calculate the business value of a model. For binary classification,
        the matrix is a 2x2 matrix with the following cells: true positive cost, true negative cost,
        false positive cost, false negative cost. The business value of a model is calculated as the
        sum of the products of the values in the matrix and the corresponding cells in the confusion
        matrix.

    Butterfly dataset
        A dataset used in :ref:`data-reconstruction-pca` to give an example where univariate
        drift statistics are insufficient in detecting complex data drifts in multidimensional
        data.

    CBPE (Confidence-Based Performance Estimation)
        A family of methods to estimate classification model performance in the absence of ground truth that takes
        advantage of the confidence which is expressed in the monitored model output probability/score prediction.
        To see how it works, check out our :ref:`CBPE deep dive<how-it-works-cbpe>`.

    Chi Squared test
        The Chi Squared test, or chi2 test as is sometimes called, is a non-parametric statistical test regarding
        discrete distributions. It is used to examine whether there is a statistically significant difference
        between expected and observed frequencies for one or more categories of a contingency table.
        In NannyML, we use the Chi Squared test to answer whether the two samples of a categorical variable
        come from a different distribution.

        The Chi Squared test results include the chi squared statistic and a p-value. The bigger the chi squared statistic,
        the more different the results between the two samples we are comparing. The p-value
        represents the chance that we would get the results we have provided if they came from the same
        distribution.

        You can find more information on the `wikipedia Chi-squared test page`_. At NannyML, we use the `scipy implementation of the
        Chi-square test of independence of variables in a contingency table`_.

    Child model
        Another name for the monitored model. It is used when describing solutions for which NannyML trains its own
        model called :term:`nanny model`.

    Concept Drift
        A change in the underlying pattern (or mapping) between the :term:`Model Inputs` and the :term:`Target` (P(y|X)).

    Confidence Band
        When we estimate a statistic from a sample, our estimation has to take into account the variance of that statistic
        from its sampled distribution. We do that by calculating :term:`Sampling Error`. When we visualize our results,
        we show a Confidence Band above and below our estimation. This confidence band comprises the values that have a distance
        less than the sampling error from our estimation. This helps us know when changes in the value of a statistic are
        statistically significant instead of happening due to the natural variance of the statistic.

        Note that the confidence band is also described as the sampling error range at the hover information that appears on
        the interactive plots.

    Confidence Score
        A score that is returned by the classification model together with class prediction. It expresses the confidence
        of the prediction i.e. the closer the score is to its minimum or maximum, the more confident the classifier is
        with its prediction. If the score is in the range between 0 and 1, it is called a *probability estimate*. It can also be
        the actual *probability*. Regardless of the algorithm type, all classification models calculate some form of
        confidence scores. These scores are then thresholded to return the predicted class. Confidence scores can be
        turned into calibrated probabilities and used to estimate the performance of classification models in the absence
        of ground truth, to learn more about this check out our :ref:`Confidence-based Performance Estimation Deep Dive<how-it-works-cbpe>`).

    Confusion Matrix
        A confusion matrix is a table that is often used to describe the performance of a classification model (or
        a set of classifiers). Each row of the matrix represents the instances in an actual class while each column
        represents the instances in a predicted class. In binary classification the matrix has 4 cells, that are
        commonly named as follows: true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN).
        For more information on the confusion matrix, see the `Wikipedia Confusion Matrix page`_.

    Covariate Shift
        A change in joint distribution of :term:`Model Inputs`, :math:`P(\mathbf{X})`.

    Data Drift
        A synonym for :term:`Covariate Shift`.

    Data Chunk
        A data chunk is simply a sample of data. All the results generated by NannyML are calculated and presented on the
        chunk level i.e. a chunk is a single data point on the monitoring results. Chunks are usually created based on time
        periods - they contain all the observations and predictions from a single hour, day, month etc. depending on
        the selected interval. They can also be size-based so that each chunk contains *n* observations or
        number-based so the whole data is split into *k* chunks. In each case chronology of data between chunks is
        maintained.

    Data Period
        A data period is a subset of the data used to monitor a model. NannyML expects the provided data to be in one of two data periods.

        The first data period is called the **reference** period. It contains all the observations for a period with an *accepted*
        level of performance. It most likely also includes **target** data. This period can be the test set for a model that
        only recently entered production or a selected benchmark dataset for a model that has been in production for some time.

        The second subset of the data is the **analysis** period. It contains the observations you want NannyML to analyse.
        In the absence of targets, performance in the analysis period can be estimated.

        You can read more about Data Periods in the :ref:`relevant data requirements section<data-drift-periods>`.

    Error
        The error of a statistic on a sample is defined as the difference between the value of the observation and the true value.
        The sample size can sometimes be 1 but it is usually bigger. When the error consists only of the effects
        of sampling, we call it :term:`sampling error<Sampling Error>`.

    Estimated Performance
        The performance the monitored model is expected to have as a result of the :term:`Performance Estimation` process.
        Estimated performance can be available immediately after predictions are made.

    Feature
        A variable used by our machine learning model. The model inputs consist of features.

    Label
        A synonym for :term:`Target`.

    Latent space
        A space of reduced dimensionality, compared to the model input space, that can
        represent our input data. This space is the result of a representation
        learning algorithm. Data points that are close together in the model input space
        are also close together in the latent space.

    Ground truth
        A synonym for :term:`Target`.

    Identifier
        Usually a single column, but can be multiple columns where necessary. It is used uniquely identify an observation.
        When providing :term:`Target` data at a later point in time, this value can help refer back to the original prediction.

        Being able to uniquely identify each row of data can help reference any particular issues NannyML might identify
        and make resolving issues easier for you. As we add functionality to provide **target** data afterwards your data
        will already be in the correct shape to support it!

        .. note::
            **Format**
                No specific format. Any str or int value is possible.

            **Candidates**
                An existing identifier from your business case.
                A technical identifier such as a globally unique identifier (GUID).
                A hash of some (or all) of your column values, using a hashing function with appropriate collision properties, e.g. the SHA-2 and SHA-3 families.
                A concatenation of your dataset name and a row number.

    Imputation
        The process of substituting missing values with actual values on a dataset.

    Kolmogorov-Smirnov test
        The Kolmogorov-Smirnov test, or KS test as it is more commonly called, is a non-parametric statistical test regarding
        the equality of continuous one-dimensional probability distributions. It can be used to compare a sample with a
        reference probability distribution, called one-sample KS test, or to compare two samples. In NannyML, we use the two-sample
        KS test looking to answer whether the two samples in question come from a different distribution.

        The KS test results include the KS statistic, or d-statistic as it is more commonly called, and a p-value.
        The d-statistic takes values between 0 and 1. The bigger the d-statistic,
        the more different the results between the two samples we are comparing are.
        The p value represents the chance that we would get the results we have provided if they come from the same
        distribution.

        You can find more information on the `wikipedia KS test page`_. At NannyML, we use the `scipy implementation of the
        two sample KS test`_.

    Loss
        Loss is a real number that quantifies the negative aspects associated with an event. It is defined by
        a :term:`Loss Function` that, for the purposes of Model Monitoring, comes from a specified performance metric.
        NannyML uses loss for :ref:`Performance Estimation for Regression<how-it-works-dle>`
        with the constraint that the :term:`Loss Function` is positive.

    Loss Function
        A `loss function`_ is a function that maps the :term:`residuals<Residual>` to a real number that
        represents a :term:`loss<Loss>` associated with the event.

    Model inputs
        Every :term:`Feature` used by the model.

    Model outputs
        The scores or probabilities that your model predicts for its target outcome.

    Model predictions
        A synonym for :term:`Model outputs`.

    Multivariate Drift Detection
        Drift Detection steps that involve all model features in order to
        create appropriate drift measures.

    Nanny model
        An extra model created by NannyML as part of its monitoring solution. The name is used to distinguish from
        the monitored model, which is sometimes referred to as :term:`child model`.

    Partition Column
        A column that tells us what :term:`Data Period` the data is in. A partition column is necessary for NannyML
        in order to produce model monitoring results.

    PCA
        Principal Component Analysis is a method used for dimensionality reduction. The method produces
        a linear transformation of the input data that results in a space with orthogonal components that maximise
        the available variance of the input data.

        More information is available on the `PCA Wikipedia page`_.

    Performance Estimation
        Estimating the performance of a deployed ML model without having access to :term:`Target`.

    Predictions
        A synonym for :term:`Model outputs`.

    Predicted labels
        The outcome a machine learning model predicts for the event it was called to predict.
        Predicted labels are a two value categorical variable. They can be represented by integers, usually
        0 and 1, booleans, meaning True or False, or strings. For NannyML, in a binary classification problem,
        it is ideal if predicted labels are presented as integers, with 1 representing the positive outcome.

    Predicted probabilities
        The probabilities assigned by a machine learning model regarding the chance that a positive event materializes
        for the binary outcome it was called to predict.

    Predicted scores
        Sometimes the prediction of a machine learning model is transformed into a continuous range of real numbers.
        Those scores take values outside the `[0,1]` range that is allowed for probabilities. The higher the score,
        the more likely the positive outcome should be.

    Ranking
        NannyML uses ranking to order columns in univariate drift results. The resulting order can be helpful
        in prioritizing what to further investigate if needed.
        More information can be found in the ranking :ref:`tutorial<tutorial-ranking>` and
        :ref:`how it works<how-ranking>` pages.

    Realized Performance
        The actual performance of the monitored model once :term:`Targets<Target>` become available.
        The term is used to differentiate between :term:`Estimated Performance` and actual results.

    Reconstruction Error
        The average Euclidean distance between the original and the reconstructed data points in a dataset.
        The reconstructed dataset is created by transforming our model inputs to a :term:`Latent space` and

        then transforming them back to the :term:`model input<Model Inputs>` space. Given that this process cannot be
        lossless, there will always be a difference between the original and the reconstructed data. This difference
        is captured by the reconstruction error.

    Residual
        The residual of a statistic on a sample is defined as the difference between the value of the observation and the expected value.
        The sample size can sometimes be 1 but it is usually bigger.
        For example the mean squared error regression metric could also be called mean squared residual because it uses the difference
        between the expected value (`y_pred`) and the observed results (`y_true`).

    Sampling Error
        Sampling errors are statistical errors that arise when a sample does not accurately represent the whole population.
        They are the difference between the real values of the population, which we don't always know, and the values
        derived by using samples from the population. In order to quantify the sampling error we use the :term:`Standard Error`.
        You can find more about how NannyML calculates sampling error at :ref:`estimation_of_standard_error`.

    Standard Error
        The Standard Error of a statistic is the standard deviation of the probability distribution we are sampling it from.
        It can also be an estimate of that standard deviation. If the statistic is the sample mean, then it is called
        Standard Error of the Mean and abbreviated as SEM.

        The exact value of standard error from an independent sample of :math:`n` observations taken from a statistical population with
        standard deviation :math:`\sigma` is:

        .. math::
            {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}

        Knowing the standard error of a statistic, we can calculate an appropriate range of values where the true value of the
        statistic lies with a given probability. More information can be read at the `Wikipedia Standard Error page`_.

    Target
        The actual outcome of the event the machine learning model is trying to predict. Also referred to as
        :term:`Ground truth` or :term:`Label`.

    Timestamp
        Usually a single column, but it can be multiple columns where necessary.
        This provides NannyML with the date and time that the prediction was made.

        NannyML needs to understand when predictions were made and how you record this,
        so it can bucket observations in time periods.

        .. note::
            **Format**
                Any format supported by Pandas, most likely:

                - *ISO 8601*, e.g. ``2021-10-13T08:47:23Z``
                - *Unix-epoch* in units of seconds, e.g. ``1513393355``

    Threshold
        A threshold is an upper or lower limit for the normally expected values of a drift method, data quality metric or performance metric.
        Outside of the range defined by the threshold values we classify the calculated value of the method or metric as abnormal in which case
        an :term:`Alert<alert>` is raised.

    Univariate Drift Detection
        Drift Detection methods that use each model feature individually
        in order to detect change.

    Unseen Values
        NannyML uses Unseen Values as a data quality check for categorical features. This is done in
        two steps. By looking at the reference :term:`Data Period` a set of expected is created for
        each categorical feature. The second step is looking at the values present in the analysis
        :term:`Data Period` for each categorical feature, any value not previously seen on the
        reference period is considered Unseen Value. You can find more information at the
        :ref:`unseen_values` tutorial.


.. _`wikipedia KS test page`: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
.. _`scipy implementation of the two sample KS test`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
.. _`contigency table`: https://en.wikipedia.org/wiki/Contingency_table
.. _`wikipedia Chi-squared test page`: https://en.wikipedia.org/wiki/Chi-squared_test
.. _`scipy implementation of the Chi-square test of independence of variables in a contingency table`:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
.. _`PCA Wikipedia page`: https://en.wikipedia.org/wiki/Principal_component_analysis
.. _`Wikipedia Standard Error page`: https://en.wikipedia.org/wiki/Standard_error
.. _`loss function`: https://en.wikipedia.org/wiki/Loss_function
.. _`Wikipedia Confusion Matrix page`: https://en.wikipedia.org/wiki/Confusion_matrix
