.. _performance-estimation-deep-dive:

=======================================
Confidence-based Performance Estimation
=======================================

This page describes the algorithms that NannyML uses to estimate monitored model performance. A high-level summary is
that NannyML takes advantage of the confidence accompanying the prediction. For
classifiers, this confidence is expressed by the probability that an observation belongs to a class.
These probabilities are then used to estimate a value of selected performance evaluation metric for a set of
observations. Provided that the monitored model returns well-calibrated probabilities (or probabilities that can
:ref:`get well-calibrated in post processing <cbpe_probability_calibration>`) and large enough set of observations,
:ref:`CBPE returns accurate estimation of performance metric value<performance-estimation-deep-dive-limitations>`.

--------------
CBPE algorithm
--------------

Binary classification
=====================

A binary classification model typically returns two outputs for each prediction - a class (binary) and a class
probability prediction (sometimes referred to as score). The score provides information about the confidence of the
prediction. A rule of thumb is that, the closer the score is to its lower or upper limit (usually 0 and 1), the higher
the probability that the classifier's prediction is correct. When this score is an actual probability, it can be
directly used to estimate the probability of making an error. For instance, imagine a high-performing model which,
for a large set of observations, returns a prediction of 1 (positive class) with a probability of 0.9. It means that
for approximately 90% of these observations, the model is correct, while for the other 10% the model is wrong.
Assuming properly calibrated probabilities, confusion matrix elements can be estimated and then used to calculate any
performance metric. Algorithms for estimating one of the simplest metrics (accuracy) and of the most complex (AUROC) are
described below. Common notation used in the sections below is the following:

    :math:`n` - number of analyzed observations/predictions,

    :math:`\hat{p} = Pr(y=1)` - monitored model probability estimate,

    :math:`y` - target label, in binary classification :math:`y\in{\{0,1\}}`,

    :math:`\hat{y}` - predicted label, in binary classification :math:`\hat{y}\in{\{0,1\}}`.

Let's start with accuracy as it should help us to quickly build the intuition. Accuracy is simply the ratio of correct
predictions to all predictions. It can be therefore expressed as:

.. math::
    Accuracy = \frac{TP+TN}{TP+FP+TN+FN} = \frac{TP+TN}{n}

Since the number of observations (:math:`n`) is known, the task comes down to estimating True Positives (TP) and
True Negatives (TN). The algorithm runs as follows:


    1. Get :math:`j`-*th* prediction from :math:`\mathbf{\hat{p}}`, denote :math:`\hat{p}_j`, and predicted label from
       :math:`\mathbf{\hat{y}}`, denote :math:`\hat{y}_j`.

    2. Calculate the estimated probability that the prediction is false:

    .. math::
        P(\hat{y} \neq y)_{j} = |\hat{y}_{j} -  \hat{p}_{j}|

    3. Calculate the estimated probability that the prediction is correct:

    .. math::
        P(\hat{y} = y)_{j}=1-P(\hat{y} \neq y)_{j}

    4. Calculate estimated confusion matrix element required for the metric:

    .. math::
        TP_{j}=\begin{cases}P(\hat{y} = y)_{j},\qquad  \ \hat{y}_{j}=1  \\  0, \qquad \qquad \qquad
        \hat{y}_{j}=0 \end{cases}

    .. math::
        TN_{j}=\begin{cases} 0,\qquad \qquad \qquad \hat{y}_{j}=1 \\ P(\hat{y} = y)_{j},\qquad \
        \hat{y}_{j}=0\end{cases}

    5. Get estimated confusion matrix elements for the whole set of predictions, e.g. for True Positives:

    .. math::
        {TP} = \sum_{j=1}^{n} {TP}_{j}

    6. Estimate accuracy:

    .. math::
        accuracy = \frac{TP+TN}{n}

Actually, first three steps are enough to estimate expected accuracy - once probabilities that the predictions are
correct are known, all that needs to be done is calculation of mean of these probabilities. The explanation above
contains the extra steps to show how confusion matrix elements are estimated which is a necessary step for other
confusion-matrix-based metrics like precision or recall. Notice, that for a model that returns positive class if
probability is larger than 50% CBPE cannot estimate accuracy lower than 0.5. This is because CBPE works well only for
models that estimate probabilities well and a model that is worse than random guess certainly does not do this. Read
more in :ref:`Limitations <performance-estimation-deep-dive-limitations>`.

A different type of metric is ROC AUC.
To estimate it one needs values of confusion matrix elements (True
Positives, False Positives, True Negatives, False Negatives)
for a set of all thresholds :math:`t`. This set is obtained by selecting a subset of :math:`m`
unique values from the set of probability predictions
:math:`\mathbf{\hat{p}}` and sorting them increasingly.
Therefore :math:`\mathbf{t}=\{\hat{p_1}, \hat{p_2}, ..., \hat{p_m}\}` and
:math:`\hat{p_1} < \hat{p_2} < ... < \hat{p_m}` (:math:`1 < m \leq n`).

The algorithm for estimating ROC AUC runs as follows:

    1. Get :math:`i`-*th* threshold from :math:`\mathbf{t}` (:math:`i` ranges from 1 to :math:`m`),  denote :math:`t_i`, .
    2. Get :math:`j`-*th* prediction from :math:`\mathbf{\hat{p}}` (:math:`j` ranges from 1 to :math:`n`), denote :math:`\hat{p}_j`.
    3. Get the predicted label by thresholding the probability estimate:

    .. math::
        \hat{y}_{i,j}=\begin{cases}1,\qquad  \hat{p}_j \geq t_i \\ 0,\qquad  \hat{p}_j < t_i \end{cases}

    4. Calculate the estimated probability that the prediction is false:

    .. math::
        P(\hat{y} \neq y)_{i,j} = |\hat{y}_{i,j} -  \hat{p}_{j}|

    5. Calculate the estimated probability that the prediction is correct:

    .. math::
        P(\hat{y} = y)_{i,j}=1-P(\hat{y} \neq y)_{i,j}

    6. Calculate the confusion matrix elements probability:

    .. math::
        TP_{i,j}=\begin{cases}P(\hat{y} = y)_{i,j},\qquad  \hat{y}_{i,j}=1  \\  0,\qquad \qquad \qquad \thinspace  \hat{y}_{i,j}=0 \end{cases}

    .. math::
        FP_{i,j}=\begin{cases}P(\hat{y} \neq y)_{i,j},\qquad  \hat{y}_{i,j}=1  \\  0,\qquad \qquad \qquad \thinspace  \hat{y}_{i,j}=0
        \end{cases}

    .. math::
        TN_{i,j}=\begin{cases} 0,\qquad \qquad \qquad \thinspace  \hat{y}_{i,j}=1 \\ P(\hat{y} = y)_{i,j},\qquad \hat{y}_{i,j}=0\end{cases}

    .. math::
        FN_{i,j}=\begin{cases} 0,\qquad \qquad \qquad \thinspace  \hat{y}_{i,j}=1 \\ P(\hat{y} \neq y)_{i,j},\qquad \hat{y}_{i,j}=0\end{cases}

    7. Calculate steps 2-6 for all predictions in :math:`\hat{\mathbf{p}}`
       (i.e. for all :math:`j` from 1 to :math:`n`) so
       that confusion matrix elements are calculated for each prediction.

    8. Get estimated confusion matrix elements for the whole set of predictions, e.g. for True Positives:

    .. math::
        {TP}_i = \sum_{j=1}^{n} {TP}_{i,j}

    9. Calculate estimated true positive rate and false positive rate:

    .. math::
        {TPR}_i = \frac{{TP}_i}{{TP}_i + {FN}_i}
    .. math::
        {FPR}_i = \frac{{FP}_i}{{FP}_i + {TN}_i}

    10. Repeat steps 1-9 to get :math:`TPR` and :math:`FPR` for all thresholds :math:`\mathbf{t}` (i.e. for
        :math:`i` from 1 to :math:`m`). As a result, get vectors of decreasing true positive rates and true
        negative rates, e.g.:

    .. math::
        \mathbf{TPR} = ({TPR}_1, {TPR}_2, ..., {TPR}_m)

    11. Calculate ROC AUC.


Multiclass Classification
=========================

Multiclass classification model outputs prediction label (predicted class) and
probability for each class. It means that if there are three classes, for example A, B and C, model output
should contain four pieces of information - predicted class (e.g. A) and three probabilities, one for each class.
Assuming these probabilities are well calibrated, they can be used to estimate performance metrics. As an example,
let's describe the process for macro-averaged precision. Let's use :math:`c` to denote total number of classes and
:math:`k` to indicate a particular class. We can stick to previously introduced notation keeping in mind that
:math:`y` and :math:`\hat{y}`
are not binary anymore and take one of :math:`c` values.

    The algorithm runs as follows:

    1. Estimate precision for each class separately, just like in binary classification. Transform vector of
       multiclass predictions :math:`\mathbf{\hat{y}}` to binary vector relevant for the class :math:`k` i.e.
       :math:`\mathbf{\hat{y}_k}` and take corresponding predicted probabilities :math:`\mathbf{\hat{p}_k}`:

        .. math::
            {precision}_k = precision(\mathbf{\hat{y}_k}, \mathbf{\hat{p}_k})

        where:

        .. math::
            \hat{y}_{k,j} = \begin{cases} 1, \qquad  \hat{y}_{j}=k \\ 0, \qquad \hat{y}_{j} \neq
            k\end{cases}

    2. Calculate macro-averaged precision:

        .. math::
            {precision} = \frac{1}{c} \sum_{k=1}^{c} {precision}_{k}

Recall, f1, specificity and *one-vs-rest* ROC AUC are estimated in the exact same way. Multiclass accuracy is
just estimated as the mean of predicted probabilities corresponding to the predicted classes.


.. _performance-estimation-deep-dive-limitations:

---------------------------
Assumptions and Limitations
---------------------------

CBPE is unbiased estimator of performance assuming:

**The monitored model returns well-calibrated probabilities.**
    Well-calibrated probabilities allow to accurately estimate confusion matrix elements and thus estimate any metric
    based on them. A model that returns perfectly calibrated probabilities
    is an ideal probabilistic model (Bayes Classifier). One may ask if there's anything to estimate if the model is perfect?
    Performance of ideal model is usually far from being equal to the maximum possible value for a given metric - it is
    lower because of the irreducible error coming from the fact that classes are not perfectly separable with the data available. In
    reality, many models are very close to Bayes Classifier - close enough for CBPE to work. Usually *good models*
    (e.g. ROC AUC>0.9) return well-calibrated probabilities, or scores that can be accurately
    :ref:`calibrated in postprocessing<cbpe_probability_calibration>`. There are also models considered as *poor*
    (with performance just better than random) that still return well-calibrated probabilities - this happens when
    most of the error is an irreducible error i.e. when there is not enough signal in the features to predict the target.
    Performance of all models change in time as a result of changes in distributions of inputs (X).
    Very often it happens right after switching to production.
    The good news is that **CBPE will remain accurate under data drift i.e. when distribution of inputs P(X) changes but
    probability of target given inputs P(Y|X) stays the same** (or in other words - if probabilities remain
    well-calibrated). An example might be a situation when one segment of population starts to dominate in
    the data - in medical applications we might have training data which is balanced with respect to patients' age
    but in production mainly older patients are analyzed. Performance of the monitored model will probably change in such
    case and this change will be noticed by CBPE.

**There is no data drift to previously unseen regions in the input space.**
    The algorithm will most likely not work if
    the drift happens to subregions previously unseen in the input
    space. In such case the monitored  model was not able to learn P(Y|X). Using
    the same example, this will happen when the model was trained on youths only but then it is applied to middle-aged people. If true relation
    between age and the target is nonlinear, most models will not estimate probability correctly on previously unseen data.
    This also depends on the type of the algorithm trained and its ability to extrapolate estimation of probabilities. For example Random Forest
    model estimated probability will remain constant and equal to the one in the closest input space region covered by training
    data - in our case this will be the probability for *the oldest patients of youths*. On the other hand, Logistic
    Regression will learn a parameter (coefficient) between age and the target and extrapolate linearly. Provided that true
    underlying relationship is also linear, Logistic Regression model will estimate probability correctly even for unseen
    *ages*.


**There is no concept drift**.
    While dealing well with data drift, CBPE will not work under concept drift i.e. when
    P(Y|X) changes. Except
    from very specific cases, there is no way to identify concept drift without any ground truth data.

**Sample of data used for estimation is large enough.**
    CBPE calculates expected values of confusion matrix elements. It means it will get more accurate with increasing
    sample size. If you toss a coin only a few times, you are likely to get more heads than tails or the other way
    around, even though the expectation of 50/50 is correct. With small sample not only CBPE won't work well - even
    if there is an access to the ground truth the calculated metric will not be reliable as well. Let's imagine a very
    good model that is evaluated on a sample of 5 observations only. For each observation model returns a positive class
    with 90% probability. It still may happen that they all turn out being negative and accuracy of your model on such sample is equal to 0!
    Read more about it :ref:`here<chunk-data-minimum-chunk>`.


.. _cbpe_probability_calibration:

---------------------------------
Appendix: Probability calibration
---------------------------------

In order to accurately estimate the performance from the model scores, they need to be well calibrated. If a classifier assigns a probability of 0.9 for a set of observations and 90% of these observations belong to the positive class, we consider that classifier to be well calibrated with respect to that subset. Most predictive models focus on performance rather than on probability estimation, therefore their scores are rarely calibrated.
Examples of different models and their calibration curves are shown below [1]_:

.. image:: ../_static/deep_dive_performance_estimation_calibration_curves.png

Probabilities can be calibrated in post-processing. NannyML uses isotonic regression to
calibrate model scores [1]_ [2]_. Since some of the models
are probabilistic and their probabilities are calibrated by design, first NannyML checks whether calibration is required
. It is done according to the following logic:

1. Stratified shuffle split [3]_ (controlled for the positive class) of reference data into 3 folds.
2. For each fold, a calibrator is fitted on train and *predicts* new probabilities for test.
3. Quality of calibration is evaluated by comparing the Expected Calibration Error (ECE) [4]_ for the raw and calibrated
   (predicted) probabilities on the test splits:


    - If in any of the folds the ECE score is higher after post processing (i.e. calibration curve is worse), the
      calibration will not be performed.

    - If in each fold post processing improves the quality of calibration, the calibrator is fitted on the whole
      reference set and probabilities are calibrated on the set that is subject to analysis.

Calibrating probabilities is yet another reason why NannyML requires reference data that is not a training set of the monitored model.
Fitting a calibrator on model training data would introduce bias [1]_.

**References**

.. [1] https://scikit-learn.org/stable/modules/calibration.html
.. [2] https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
.. [3] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
.. [4] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht: "Obtaining well calibrated probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

