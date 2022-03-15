.. _performance-estimation-deep-dive:

=======================================
Confidence-based Performance Estimation
=======================================

Introduction
============

A binary classification model typically returns two outputs for each prediction - a class (binary) and a class
probability predictions (sometimes referred to as score). The score provides information about the confidence of the
prediction. A rule of thumb is that, the closer a score is to its lower or upper limit (usually 0 and 1), the higher
the probability that a classifier’s prediction is correct. When this score is an actual probability, it can be
directly used to calculate the probability of making an error. For instance, imagine a high-performing model which,
for a large set of observations, returns a prediction of 1 (positive class) with probability of 0.9. It means that
for approximately 90% of these observations, the model is correct while for the other 10% the model is wrong.
Assuming properly calibrated probabilities, NannyML reconstructs the whole confusion matrix and calculates ROC AUC
for a set of :math:`n` predictions according to following algorithm:

    Let’s denote:

    :math:`\hat{p} = Pr(y=1)` - monitored model probability estimate,

    :math:`y` - target label, :math:`y\in{\{0,1\}}`,

    :math:`\hat{y}` - predicted label, :math:`\hat{y}\in{\{0,1\}}`.

    To calculate ROC AUC one needs values of confusion matrix elements (True Positives, False Positives, True Negatives, False Negatives)
    for a set of all thresholds :math:`t`. This set is obtained by selecting subset of :math:`m`
    unique values from the set of probability predictions
    :math:`\mathbf{\hat{p}}` and sorting them increasingly.
    Therefore :math:`\mathbf{t}=\{\hat{p_1}, \hat{p_2}, ..., \hat{p_m}\}` and
    :math:`\hat{p_1} < \hat{p_2} < ... < \hat{p_m}` (notice that in some cases :math:`m=n`).

    The algorithm runs as follows:

    1. Get :math:`i`-*th* threshold from :math:`\mathbf{t}`,  denote :math:`t_i`.
    2. Get :math:`j`-*th* prediction from :math:`\mathbf{\hat{p}}`, denote :math:`p_j`.
    3. Get binary prediction by thresholding probability estimate:

    .. math::
        \hat{p}_{i,j}=\begin{cases}1,\qquad  \hat{p}_j \geq t_i \\ 0,\qquad  \hat{p}_j < t_i \end{cases}

    4. Calculate the estimated probability that the prediction is false:

    .. math::
        P(\hat{y} \neq y)_{i,j} = |\hat{y}_{i,j} -  \hat{p}_{i,j}|

    5. Calculate the estimated probability that the prediction is correct:

    .. math::
        P(\hat{y} = y)_{i,j}=1-P(\hat{y} \neq y)_{i,j}

    6. Calculate the confusion matrix elements probability:

    .. math::
        TP_{i,j}=\begin{cases}P(\hat{y} = y)_{i,j},\qquad  y_{i,j}=1  \\  0,\qquad \qquad \qquad \thinspace  y_{i,j}=0 \end{cases}

    .. math::
        FP_{i,j}=\begin{cases}P(\hat{y} \neq y)_{i,j},\qquad  y_{i,j}=1  \\  0,\qquad \qquad \qquad \thinspace  y_{i,j}=0
        \end{cases}

    .. math::
        TN_{i,j}=\begin{cases} 0,\qquad \qquad \qquad \thinspace  y_{i,j}=1 \\ P(\hat{y} = y)_{i,j},\qquad y_{i,j}=0\end{cases}

    .. math::
        FN_{i,j}=\begin{cases} 0,\qquad \qquad \qquad \thinspace  y_{i,j}=1 \\ P(\hat{y} \neq y)_{i,j},\qquad y_{i,j}=0\end{cases}

    7. Calculate steps 2-6 for all predictions in :math:`\hat{\mathbf{p}}`
       (i.e. for all :math:`j` from 1 to :math:`n`) so
       that confusion matrix elements are calculated for each prediction.

    8. Get estimated confusion matrix elements for the whole set of predictions, e.g. for True Positives:

    .. math::
        {TP}_i = \sum_{j}^{n} {TP}_{i,j}

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


Probability calibration
=======================
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
