.. _how-it-works-univariate-drift-detection:

Univariate Drift Detection
==========================

Univariate Drift Detection looks at each feature individually and checks whether its
distribution has changed compared to reference data. There are many ways to compare two samples of data and measure
their *similarity*. NannyML provides several drift detection methods so that the users can choose the one that suits
their data best or the one they are familiar with. Additionally more than one method can be used together to
gain different perspectives on how the distribution of your data is changing.

This page explains which aspects of a distribution change each drift detection method is able to capture,
what are the important implementation details and in which situations a specific method can be a good choice.

We are grouping the drift detection methods presented according to whether they apply to categorical (discrete) or
continuous features. When a method is used for both it is mentioned in both places because of implementation differences
between the two types of features.

Lastly let's note that we are always performing two sample tests or comparisons. Probability density functions (PDF) and
cumulative density functions (CDF) are always estimated from the data samples that are being compared.

.. _univariate-drift-detection-continuous-methods:

Methods for Continuous Features
--------------------------------

.. _univ_cont_method_ks:

Kolmogorov-Smirnov Test
.......................

The `Kolmogorov-Smirnov Test`_ is a two-sample, non-parametric statistical test. It is used to test for the equality of
one-dimensional continuous distributions. The test outputs the test statistic, called D-statistic, and an associated p-value.
The test statistic is the maximum distance of the cumulative distribution functions (CDF) of the two samples.

The D-statistic is robust to small changes in the data, easy to interpret and falls into  0-1 range.
This makes the Kolmogorov-Smirnov test a popular choice for many data distribution monitoring
practitioners. You can see on the image below how the value of D-statistic changes with the change of data
distribution to build some intuition on it's behavior.

.. image:: ../_static/how-it-works-ks.svg
    :width: 1400pt


.. _univariate-drift-detection-cont-jensen-shannon:

Jensen-Shannon Distance
........................

Jensen-Shannon Distance is a metric that tells us how different two probability distributions are.
It is based on `Kullback-Leibler divergence`_ but is created in such a way that it is symmetric and ranges between 0 and 1.

Between two distributions :math:`P,Q` of a continuous feature `Kullback-Leibler divergence`_  is defined as:

.. math::
    D_{KL} \left(P || Q \right) = \int_{-\infty}^{\infty}p(x)\ln \left( \frac{p(x)}{q(x)} \right) dx


where :math:`p(x)` and :math:`q(x)` are the probability density functions of the distributions :math:`P,Q` respectively.
And `Jensen-Shannon Divergence`_ is defined as:

.. math::
    D_{JS} \left(P || Q \right) = \frac{1}{2} \left[ D_{KL} \left(P \Bigg|\Bigg| \frac{1}{2}(P+Q) \right) + D_{KL} \left(Q \Bigg|\Bigg| \frac{1}{2}(P+Q) \right)\right]

and is a method of measuring the similarity between two probability distributions.

Jensen-Shannon Distance is then defined as the squared root of Jensen-Shannon divergence and is a proper distance metric.
Unlike KS D-static that looks at maximum difference between two empirical CDFs, JS distance looks at the total difference between empirical Probability Density Functions
(PDF). This makes it
more sensitive to changes that may be ignored by KS. This effect can be observed in the plot below to get the intuition:

.. image:: ../_static/how-it-works-js.svg
    :width: 1400pt

In the two rows we see two different changes been induced to the reference dataset.
We can see from the cumulative density functions on the right that the resulting KS distance is the same.
On the left we see the probability density functions of the samples and the resulting Jensen-Shannon Divergence
at each point. Integrating over it and taking the square root gives the Jensen-Shannon distance showed. We can
see that the resulting Jensen-Shannon distance is able to differentiate the two changes.

In order to calculate Jensen-Shannon Distance NannyML splits a continuous feature into bins, calculates the relative
frequency for each bin from reference and analyzed data and calculates the
resulting Jensen-Shannon Distance. The binning is done using `Doane's formula`_ from numpy.
If a continuous feature has relatively low amount of unique values, meaning that
unique values are less then 10% of the reference dataset size up to a maximum of 50, each value becomes a bin.

.. _univariate-drift-detection-cont-wasserstein:

Wasserstein Distance
........................

The `Wasserstein Distance`_, also known as earth mover's distance and the Kantorovich-Rubinstein metric,
is a measure of the difference between two probability distributions. Wasserstein distance
can be thought of as the minimum amount of work needed to transform one distribution into the other. Informally, if
the PDF of each distribution is imagined as a pile of dirt, the Wasserstein distance is the amount of work it would
take to transform one pile of dirt into the other (which is why it is also called the earth mover's distance).

While finding the Wasserstein distance can be framed as an optimal transport problem, when each distribution is
one-dimensional, the CDFs of the two distributions can be used instead. When defined in this way, the Wasserstein
distance is the integral of the absolute value of the difference between the two CDFs, or more simply, the area between the CDFS. The figure below illustrates this.

.. image:: ../_static/how-it-works-emd.svg
    :width: 1400pt

Mathematically we can express this as follows: For the :math:`i^\text{th}` feature of a dataset :math:`X=(X_1,...,X_i,...,X_n)`, let :math:`\hat{F}_{ref}` and :math:`\hat{F}_{ana}` represent the
empirical CDFs of the reference and analysis samples respectively. Further, let :math:`X_i^{ref}` and :math:`X_i^{ana}` represent the reference and analysis samples. Then the
Wasserstein distance between the two distributions is given by:

.. math::
    W_1\left(X_i^{ref},X_i^{ana}\right) = \int_\mathbb{R}\left|\hat{F}_{ref}(x)-\hat{F}_{ana}(x)\right|dx

.. _univariate-drift-detection-cont-hellinger:

Hellinger Distance
........................

The `Hellinger Distance`_, is a distance metric used to quantify the similarity between two probability distributions. It measures the overlap between the probabilities assigned 
to the same event by both reference and analysis samples. It ranges from 0 to 1 where a value of 1 is only achieved when reference assigns zero probability to each event to which 
the analysis sample assigns some positive probability and vice versa. 
The formula is given by: 

.. math::
    H\left(X_i^{ref},X_i^{ana}\right) = \frac{1}{\sqrt{2}}\left[\int_{}\left(\sqrt{{F}_{ref}(x)}-\sqrt{{F}_{ana}(x)}\right)^2dx\right]^{1/2}

In order to Calculate Hellinger Distance NannyML splits a continuous feature into bins based on the reference data. The relative frequency
for each bin from reference and the samples of analysis data is calculated to generate the
resulting Hellinger Distance. If there's new data in the analysis sample that does not fall into the range of the bin edges that were calculated based on reference, another bin 
is created that fits all that data. An additional bin is also created in reference and its probability/relative frequency is set to 0. The binning is done using `Doane's formula`_ from numpy.
If a continuous feature has relatively low amount of unique values, meaning that
unique values are less then 10% of the reference dataset size up to a maximum of 50, each value becomes a bin.

This distance is very closely related to the Bhattacharya Coefficient. However we choose the former because it follows the triangle inequality and is 
a proper distance metric. Moreover the division by the squared root of 2 ensures that the distance is always between 0 and
1, which is not the case with the Bhattacharya Coefficient. The relationship between the two can be depicted as follows:

.. math::
    H^2\left(X_i^{ref},X_i^{ana}\right) = 2(1-BC\left(X_i^{ref},X_i^{ana}\right))

where 

.. math::
    BC\left(X_i^{ref},X_i^{ana}\right) =  \int_{}\sqrt{{F}_{ref}(x){F}_{ana}(x)}dx

.. _univariate-drift-detection-categorical-methods:

Methods for Categorical Variables
---------------------------------

.. _univ_cat_method_chi2:

Chi-squared Test
................

The `Chi-squared test`_ is a statistical hypothesis test of independence for categorical data.
The test outputs the test statistic, sometimes called chi2 statistic, and an associated p-value.

We can understand the Chi-squared test in the following way. We create a `contingency table`_ from the
categories present in the data and the two samples we are comparing. The expected frequencies,
denoted :math:`m_i`, are calculated from the marginal sums of the contingency table.
The observed frequencies, denoted :math:`x_i`, are calculated from the actual
frequency entries of the contingency table. The test statistic is then given by the formula:

.. math::
    \chi^2 = \sum_{i=1}^k \frac{(x_i - m_i)^2}{m_i}

where we sum over all entries in the contingency table.

This makes the chi-squared statistic sensitive to all changes in the distribution,
especially to the ones in low-frequency categories, as the expected frequency is in the denominator.
It is therefore not recommended for categorical features with many low-frequency categories or high cardinality
features, unless the sample size is really large.
Otherwise, in both cases false-positive alarms are expected.
Additionally, the statistic is non-negative and not limited which sometimes makes it difficult to interpret.
Despite that, the Chi-squared test is a common choice amongst practitioners as it provides p-value together with the
statistic that helps to better evaluate its result.

On the image below there is a visualization of the chi-squared statistic for a categorical variable with two
categories, a and b. You can see the expected values are calculated from both the reference and analysis data.
The red bars represent the difference between the observed and expected frequencies.
As mentioned above, in the chi-squared statistic formula,
the difference is squared and divided by the expected frequency and the resulting value is then summed over all categories
for both samples.

.. image:: ../_static/how-it-works-chi2.svg
    :width: 1400pt

.. _univ_cat_method_js:

Jensen-Shannon Distance
........................

Jensen-Shannon Distance is a metric that tells us how different two probability distributions are.
It is based on `Kullback-Leibler divergence`_ but is created in such a way that it is symmetric and ranges between 0 and 1.

Between two distributions :math:`P,Q` of a categorical feature `Kullback-Leibler divergence`_  is defined as:

.. math::
    D_{KL} \left(P || Q \right) = \sum_{x \in X} P(x)\ln \left( \frac{P(x)}{Q(x)} \right)


where :math:`p(x)` and :math:`q(x)` are the probability mass functions of the distributions :math:`P,Q` respectively.
And `Jensen-Shannon Divergence`_ is defined as:

.. math::
    D_{JS} \left(P || Q \right) = \frac{1}{2} \left[ D_{KL} \left(P \Bigg|\Bigg| \frac{1}{2}(P+Q) \right) + D_{KL} \left(Q \Bigg|\Bigg| \frac{1}{2}(P+Q) \right)\right]

and is a method of measuring the similarity between two probability distributions.

Jensen-Shannon Distance is then defined as the squared root of Jensen-Shannon divergence and is a proper distance metric.
As we see for
categorical data, JS distance is calculated based on the relative frequencies of each category in reference and
analysis data. The intuition is that it measures an *average* of all changes in relative frequencies of categories.
Frequencies are compared by dividing one by another, therefore JS distance, just like Chi-squared statistic,
is sensitive to changes in less frequent classes. This means that an absolute change of 1 percentage point for less
frequent class will have stronger
contribution to the final JS distance value than the same change in more frequent class. For this reason it
may not be the best choice for categorical variables with many low-frequency classes or high cardinality.

To help our intuition we can look at the image below:

.. image:: ../_static/how-it-works-cat_js.svg
    :width: 1400pt


We see how the relative frequencies of three categories have changed between reference and analysis data.
We also see that the JS Divergence contribution of each change and the resulting JS distance.

.. _univ_cat_method_hellinger:

Hellinger Distance
........................

The `Hellinger Distance`_, is a distance metric used to quantify the similarity between two probability distributions. It measures the overlap between the probabilities assigned 
to the same event by both reference and analysis samples. It ranges from 0 to 1 where a value of 1 is only achieved when reference assigns zero probability to each event to which 
the analysis sample assigns some positive probability and vice versa. 

For a categorical feature Hellinger Distance is defined as:

.. math::
 H\left(X_i^{ref},X_i^{ana}\right) = \frac{1}{\sqrt{2}}\left[\sum_{x \in X}\left(\sqrt{{F}_{ref}(x)}-\sqrt{{F}_{ana}(x)}\right)^2\right]^{1/2}

where :math:`{F}_{ref}` and :math:`{F}_{ana}` refer to the Probability Mass Functions of the reference and analysis samples respectively. 

In order to Calculate Hellinger Distance for categorical data, NannyML splits a categorical feature into bins where each bin corresponds to a unique label in the reference data. The relative frequency
for each bin from reference and the samples of analysis data is calculated to generate the
resulting Hellinger Distance. If there's any unseen category that does not already exist in the calculated bins, another bin is created 
that fits all the new data. An additional bin is also created in reference and its probability/relative frequency is set to 0. 

.. _univ_cat_method_l8:

L-Infinity Distance
...................

We are using L-Infinity to measure the similarity of categorical features. L-Infinity, for categorical features, is defined as
the maximum of the absolute difference between the relative frequencies of each category in the reference and analysis data.
You can find more about `L-Infinity at Wikipedia`_. It falls into the range of 0-1 and is easy to interpret as
is the greatest change in relative frequency among all categories. This behavior is different compared to Chi Squared test
where even small changes in low frequency labels can heavily influence the resulting test statistic.

To help our intuition we can look at the image below:

.. image:: ../_static/how-it-works-linf.svg
    :width: 1400pt

We see how the relative frequencies of three categories have changed between reference and analysis data.
We also see that the resulting L-Infinity distance is the relative frequency change in category c.

Comparison of Methods for Continuous Variables
----------------------------------------------
The drift measures discussed below quantify the difference between two data sets - in our case, a reference data set and an analysis data set. The reference data establishes what we
expect the data to look like, and the drift measure tells us how different the analysis data set is from the reference data set.

Shifting the Mean of the Analysis Data Set
.............................
In this experiment, we show how each method responds as the mean of the analysis data set moves further away from the mean of the reference data set.
To demonstrate this, the reference data set was sampled from :math:`\mathcal{N}(0,1)`, and the analysis data set was sampled from :math:`\mathcal{N}(M,1)`
where :math:`M = \{0,0.1,0.2,...,7\}`. It is worth noting that some of the larger shifts demonstrated here represent extreme cases unlikely to come up in the real world.
However, we chose to evaluate the methods in these cases to give a complete picture of their behaviour.

We show the confidence intervals for empirical experiments like this one to demonstrate the stability of each method in comparison to the others. The confidence intervals depend
on both the sample size of the reference and analysis data sets and the number of trials we average over. However, since the sample sizes and the number of repetitions are the same
for each drift detection method, the results are comparable.

In this experiment, the sample size of both the reference and analysis datasets was 1000 points, and the number of repetitions for each value of the mean of the analysis data set was 20.

.. image:: ../_static/univariate-comparison/shifting_mean.svg
    :width: 1400pt
The results illustrate that Wasserstein distance changes proportionally to the distance that the mean has moved. Jensen-Shannon Distance and the Kolmogorov-Smirnov Statistic are both relatively
more sensitive to smaller shifts compared to bigger shifts. This means that a shift in the mean of the analysis data set from 0 to 0.1 will cause a bigger change than a change from 5.0 to 5.1. 
Hellinger Distance, on the other hand, displays behaviour resembling a sigmoid function; Hellinger Distance is not as sensitive to small and large changes to the mean of the analysis data set 
compared to shifts of medium size.

Shifting the Standard Deviation of the Analysis Data Set
...........................................
In this experiment, we show how each method responds as the standard deviation of the analysis set increases. The reference data set was sampled from :math:`\mathcal{N}(0, 1)` and the analysis data set
was sampled from :math:`\mathcal{N}(0, \Sigma)` where :math:`\Sigma = \{1, 1.1, 1.2,...,10\}`. The size of both the reference and analysis data sets was 1000 points, and the number of repetitions for each value
of the independent variable (the standard deviation of the analysis data set) was 20.

.. image:: ../_static/univariate-comparison/shifting_std.svg
    :width: 1400pt

In this case, Wasserstein distance again changes proportionally to the change in standard deviation. Jensen-Shannon distance, the Kolmogorov-Smirnov statistic, and the Hellinger distance exhibit high sensitivity, even 
to small changes. However, the Hellinger distance has a slightly softer start than the Jensen-Shannon distance and the Kolmogorov-Smirnov statistic. In this experiment, the main difference between the Jensen-Shannon distance, 
the Kolmogorov-Smirnov statistic, and Hellinger distance is that the stability of the measures (illustrated by the confidence intervals) differs, with Jensen-Shannon distance exhibiting the highest stability of the three. So, if 
you are looking for a measure more sensitive to small changes in standard deviation than Wasserstein Distance, we suggest Jensen-Shannon distance because of its stability.

Tradeoffs of The Kolmogorov-Smirnov Statistic
.............................................
The Kolmogorov-Smirnov Statistic differs from the other three methods because the value is the maximum distance between the empirical cumulative density functions (ECDFs). This can lead to cases where drift occurring in one region
of the analysis distribution hides drift occurring in other areas. In the visualization below, we show one extreme case of this happening to help build intuition. 

In this visualization, the reference distribution is a combination of two normal distributions and thus is bimodal. On the top row, labeled Analysis 1, only the right mode of the analysis distribution shifts. On the bottom row, labeled Analysis 2, 
both the left mode and the right mode of the analysis distribution shift.

.. image:: ../_static/univariate-comparison/fool_ks.svg
    :width: 1400pt

Looking at column 1, which visualizes the Jensen-Shannon distance, we see that the value increases from 0.707 to 0.999 because of the divergence between the distributions increases. In the second column, which visualizes Hellinger distance, the value 
increases from 0.707 to 0.997 because the overlap between the reference and analysis distribution decreases. **In the third column, which visualizes the Kolmogorov-Smirnov statistic, we see that the largest difference between the analysis ECDF and the 
reference ECDF remains the same in the first row and the second row, even though the drift increases.** In the fourth column, we see that Wasserstein distance increases from 4 to 6.999 due to the increase in the area between the reference ECDF and analysis ECDF.

Tradeoffs of Jensen-Shannon Distance and Hellinger Distance
...........................................................
Experiment 1
************
Both Jensen-Shannon Distance and Hellinger Distance are related to the amount of overlap between probability distributions. 
This means that in cases where the amount of overlap stays the same but drift increases, neither the Jensen-Shannon distance nor 
the Hellinger distance will detect the change. Cases where the amount of overlap stays the same, but the drift increases are very 
rare in practice, but they can occur when two distributions are disjoint to begin with and then move further away from one another. 
An example of this is shown below:

.. image:: ../_static/univariate-comparison/fool_js_ks_hellinger.svg
    :width: 1400pt

In this example, the reference distribution is a combination of two normal distributions and is thus bimodal. In the first case, the right 
mode of the analysis distribution shifts to the right, and in the second case, both modes shift to the right. In the second case, this could 
mean that either the left mode shifted over to the right of what was initially the right mode of the analysis or both the left mode and the 
right mode of analysis shifted to the right. In either case, this subjectively seems like more drift, and neither Jensen-Shannon distance nor 
Hellinger distance catches this, but Wasserstein distance does. This is because Wasserstein distance is not based on the overlap between the 
two distributions but on the amount of work required to transform one distribution into the other. In this context, “work” can be thought of 
as the amount of probability density multiplied by the distance it has to travel.

Experiment 2
************
Since Jensen-Shannon distance and Hellinger distance are related to the overlap between distributions, if the distributions are completely disjoint, 
then both measures will be maxed out at 1. So, if the distributions begin disjoint and get even further apart, Jensen-Shannon distance and Hellinger will not increase. 
On the other hand, since Wasserstein Distance quantifies the distance between distributions, the measure will increase. 

.. image:: ../_static/univariate-comparison/disjoint_only_emd.svg
    :width: 1400pt

In this experiment, we double the distance between the reference and analysis, and we see that Jensen-Shannon distance, the Kolmogorov-Smirnov statistic, 
and Hellinger distance remain at 1 (their max value), while Wasserstein distance increases proportionally to the distance that the distribution has moved. 
This example is more of an edge case, but disjoint distributions can arise in real-world scenarios. For example, when training generative adversarial networks, 
this issue can arise, and a common remedy is using a loss function based on Wasserstein Distance!

Tradeoffs of Wasserstein Distance
.................................
Experiment 1
************
In this experiment, we demonstrate a case where Wasserstein distance behaves differently from the Jensen-Shannon distance and Hellinger distance because it is based on 
the amount of work that would be required to transform one distribution into the other as opposed to the amount of overlap between the distributions.

In this example, the reference and analysis distribution are both mixtures of normal distributions with two modes, as can be seen below:

.. image:: ../_static/univariate-comparison/fool_emd.svg
    :width: 1400pt

In this case, the overall distance between the reference and analysis remains the same in both the shift demonstrated on the first row and the second row, but the amount of 
overlap changes. The distance stays the same because, in the case visualized on the first row, one mode of the analysis distribution moves a significant distance. In the 
second case, which is visualized in row 2, each mode moves half of that distance. The Kolmogorov-Smirnov statistic and the Wasserstein distance treat both cases the same, 
even though in one case, one mode of the analysis distribution moves and in the other, both move. 

Experiment 2
************
Since Wasserstein distance measures the work (defined as the amount of density times the distance it must travel) that it would take to transform one distribution into the other, 
the presence of extreme data points can greatly increase the value. If two distributions are mostly identical, but one has an outlier, then the work it takes to transport that 
small bit of probability density to the other distribution is still significant (small density multiplied by a large distance).

.. image:: ../_static/univariate-comparison/outlier.svg
    :width: 1400pt

In this experiment, we move one data point to increasingly extreme values, and the result is that Wasserstein Distance increase in proportion to the size of that extreme value while the 
other methods remain the same. In cases where the overall shape of the distribution is more important than detecting a few outliers, we advise against using Wasserstein distance. 
However, if outliers are important in your case, then Wasserstein distance might be the right choice.

Experiment 3
************
In this experiment, we further exemplify the sensitivity of Wasserstein Distance to extreme values. To do so, we compare a normal distribution to a 
Cauchy distribution. The Cauchy distribution has no analytically derivable moments, and generating samples from a random variable distributed 
according to the Cauchy distribution will result in a data set with much of its density in a small range but with fat tails. The probability 
density function (PDF) in the range :math:`[-10,10]` is visualized below.

.. image:: ../_static/univariate-comparison/outlier.svg
    :width: 1400pt

Notably, the general shape of the Cauchy distribution resembles the normal distribution, but there is much more density in the tails. 
When increasing the scale parameter, the Cauchy distribution spreads out, and the tails become even denser. The behaviour of Wasserstein 
distance, Jensen-Shannon distance, Hellinger distance, and the Kolmogorov-Smirnov statistic when the reference sample is drawn from 
:math:`\mathcal{N}(0,1)` and the analysis is drawn from :math:`\text{Cauchy}(0,\Gamma)` where :math:`\Gamma = \{0.5, 0.6,...,3\}` is shown below:

.. image:: ../_static/univariate-comparison/cauchy_empirical.svg
    :width: 1400pt
    
Since Wasserstein distance is sensitive to extreme values, the variance of the measure is high for all parameters and increases as the scale parameter does. 
Jensen-Shannon distance, the Kolmogorov-Smirnov statistic, and the Hellinger distance show far less variance. It is important to note that the lack of 
stability observed in the Wasserstein distance is the result of high variance in the tails of the Cauchy distribution. 

Comparison of Methods for Categorical Variables
-----------------------------------------------
Sensitivity to Sample Size of Different Drift Measures
......................................................
In many cases, we would like methods that return the same value for the same magnitude of drift, regardless of the sample size of either the reference or 
analysis set. Jensen-Shannon distance, Hellinger distance, and L-Infinity distance all exhibit this property, while the Chi-Squared statistic does not. In 
cases where the chunks in your analysis may be different sizes, as can be the case when using period-based chunking, we suggest considering this behaviour 
before you use the chi-squared statistic.

In this experiment, the proportions of each category were held constant in both the reference and analysis data sets. In the reference data set, the relative 
frequency of category “a” was always 0.5, the relative frequency of category “b” was also 0.5, and the data set size was held constant at 2000 points. 
In the analysis data set, the relative frequency of category “a” was always 0.8, the relative frequency of category “b” was always 0.2, and 
the data size increased from 100 points to 1000 points, as shown below.

.. image:: ../_static/univariate-comparison/chi2_sample_size.svg
    :width: 1400pt

Behaviour When a Category Slowly Disappears
............................................
In this experiment, we show how each method behaves as a category shrinks and eventually disappears. 
The analysis distribution starts identically distributed, and slowly, the category “b” shrinks, and “c” grows.

.. image:: ../_static/univariate-comparison/cat_disappears.svg
    :width: 1400pt

We see that L-Infinity has linear behaviour in relation to the proportion of the categories changing. 
In contrast, the Hellinger distance and chi-squared statistic increase slowly at first but more quickly when 
the “b” category is about to disappear.

Behaviour When Observations from a New Category Occur
......................................................
In this experiment, we show how each method reacts to the slow entry of a new category. To begin with, the 
analysis distribution is distributed identically to the reference distribution.

.. image:: ../_static/univariate-comparison/cat_enters.svg
    :width: 1400pt

The main two things to note in this experiment are that both Jensen-Shannon distance and Hellinger distance 
are extremely sensitive to the new category and the behaviour of the methods as a new category appears is not 
necessarily symmetric to their behaviour as a new category appears. The chi-squared statistic and L-infinity 
distance show more uniform behaviour that scales roughly linearly with the proportion of the new category. If the 
appearance of a new category is significant in your case, then the Hellinger distance or Jensen-Shannon distance may be a good fit.

Effect of Sample Size on Different Drift Measures
..................................................
In this experiment, we demonstrate the stability of each method while changing the size of the analysis sample. To demonstrate this, 
we first drew a sample of 5,000 points from  :math:`\text{Binom}(10,0.5)` to serve as the reference data set. The probability 
mass function (PMF) of this distribution looks like this:

.. image:: ../_static/univariate-comparison/binomial_pmf.svg
    :width: 1400pt

Then, to demonstrate the effect of sample size, we drew samples of sizes :math:`\{100, 200, 300,..., 3000\}` , again 
from :math:`\text{Binom}(10,0.5)`, to serve as our analysis data sets. We know that there is no distribution shift 
between the reference sample and any of the analysis samples because they were all drawn from the same distribution, namely :math:`\text{Binom}(10,0.5)`. 
In this way, we can see the impact that sample size has on each of the drift measures. 

Below are visualizations of measuring drift between the reference data set and the analysis data sets of varying size:

.. image:: ../_static/univariate-comparison/binomial_and_sample_size.svg
    :width: 1400pt

For Jensen-Shannon distance, Hellinger distance, and L-infinity distance, the observed drift decreases as the analysis sample 
increases in size and thus better represents the distribution. The chi-squared statistic, however, does not decrease as the analysis 
sample better approximates the population. So, in cases where analysis chunks have varying sizes and false positives in smaller 
chunks are a concern, the chi-squared statistic may be the right choice.

Effect of the Number of Categories on Different Drift Measures
..............................................................
In this experiment, we show how the number of categories affects each method. The setup of 
this experiment is as follows: First, we defined a set :math:`M = \{2,3,4,...,60\}`, and for each :math:`m` in :math:`M`, we 
drew a sample from :math:`\text{Binom}(m, 0.5)` of 5000 points to serve as the reference data set. We then 
drew a sample of 1000 points again from :math:`\text{Binom}(m, 0.5)` to serve as the analysis data set. We then calculated 
the drift between the reference data set and analysis data set with the Jensen-Shannon distance, Hellinger distance, 
L-infinity distance, and the chi-squared statistic. The results are shown below:

.. image:: ../_static/univariate-comparison/binom_and_num_cats.svg
    :width: 1400pt

We see an increase in the Jensen-Shannon distance, Hellinger distance, and the chi-squared statistic as the number of categories 
increases because the small differences in the frequencies in each category due to sampling effects are summed up. Thus, the more 
terms in the sum, the higher the value. On the other hand, L-infinity distance does not increase because it only looks at the largest 
change in frequency of all the categories. For intuition, a visualization of the Hellinger distance and the L-infinity distance is shown 
below when the number of categories is 61 (i.e., :math:`\text{Binom(60, 0.5}`)).

.. image:: ../_static/univariate-comparison/hellinger_vs_linf.svg
    :width: 1400pt

Again, since Hellinger Distance (as well as Jensen-Shannon distance and the chi-squared statistic) sums a transformation of the differences 
between the frequencies of each category, the values will increase with the number of categories. L-infinity distance only looks at the maximum 
difference, in this case, a difference of 0.03 in category 29, and thus is unaffected by the number of categories. So, when dealing with 
data sets with many categories, we suggest using the L-infinity distance.

Comparison of Drift Methods on Data Sets with Many Categories
.............................................................

In cases with many categories, it can be difficult to detect important drift if it only occurs in a few categories. This is because some methods 
(like Jensen-Shannon distance, Hellinger distance, and the chi-squared statistic) sum a transformation of the difference between 
the relative frequency of each category. Sampling effects can cause small differences in the frequency of each category, but when summed 
together, these small differences can hide important drifts that occur in only a few categories. L-infinity distance only looks at the 
largest change in relative frequency among all the categories. It thus doesn't sum up all of the small, negligible differences caused by sampling error.

Here we show an experiment that highlights this behaviour. There are three important samples in this experiment, namely the reference sample, an analysis 
sample with no real drift (i.e. the sample is drawn from the same distribution), and an analysis set with severe drift in only one category. The 
reference and analysis set without drift were drawn from the uniform distribution with 200 categories. The analysis set with severe drift was 
constructed by drawing a sample from the uniform distribution with 200 categories, then adding more occurrences of the 100th category. The sample 
size of each of the three sets was 7000 points. A visualization of the empirical probability mass function can be seen below.

.. image:: ../_static/univariate-comparison/uniform.svg
    :width: 1400pt

We see that each of the three distributions looks similar, aside from a major drift in category 100 in the analysis sample with severe drift. We can 
compare the values that each method returns for the difference between the reference sample and the analysis sample without drift, and the reference 
sample and the analysis sample with severe drift in one category, as seen below:

.. image:: ../_static/univariate-comparison/horizontal_bar.svg
    :width: 1400pt

We see that the sampling effects (the small differences in the frequencies of each category) hide the important drift when using Jensen-Shannon distance, 
Hellinger distance, and the chi-squared statistic because they sum the difference in frequency for each category. On the other hand, L-infinity shows a 
significant difference between the two.

TL;DR
-----
Methods for Continuous Variables
................................
**We suggest Jensen-Shannon distance or Wasserstein distance for continuous features.**
While there is no one-size-fits-all method, both of these methods perform well in many cases, and generally, if drift occurs, these methods will catch it. 

There are three main differences between these two measures. First, Jensen-Shannon distance will always be in the range :math:`[0, 1]`, whereas Wasserstein distance 
has a range of :math:`[0, \infty)`. Second, Jensen-Shannon distance tends to be more sensitive to small drifts, meaning that it will likely raise more false alarms 
than Wasserstein distance, but it might be more successful in catching meaningful low-magnitude drifts. And third, Wasserstein distance tends to be more 
sensitive to outliers than Jensen-Shannon distance.

Methods For Categorical Variables
.................................
**For categorical features, we recommend Jensen-Shannon distance or L-Infinity distance if you have many categories.** 
Both methods perform well in most cases, exhibit few downsides, and are bounded in the range :math:`[0,1]`. In cases 
where there are many categories, and you care about changes to even one category, we suggest L-Infinity distance.


.. _`Chi-squared test`: https://en.wikipedia.org/wiki/Chi-squared_test
.. _`Kolmogorov-Smirnov Test`: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
.. _`Jensen-Shannon Divergence`: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
.. _`Hellinger Distance`: https://en.wikipedia.org/wiki/Hellinger_distance
.. _`L-Infinity at Wikipedia`: https://en.wikipedia.org/wiki/L-infinity
.. _`Kullback-Leibler divergence`: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
.. _`Doane's formula`: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
.. _`Wasserstein Distance`: https://en.wikipedia.org/wiki/Wasserstein_metric
.. _`contingency table`: https://en.wikipedia.org/wiki/Contingency_table
