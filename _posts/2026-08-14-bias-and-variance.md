---
layout: post
title: "Bias and Variance: Understanding Underfitting and Overfitting"
date: 2026-08-14 09:00:00 +1000
description: How bias, variance, and Bayes error explain a model's generalization error—and what training and validation results can tell us.
tags: [machine-learning, fundamentals, bias-variance, tutorial]
categories: ["ML Basics and Concepts"]
toc:
  beginning: true
related_posts: true
math: true
---

_This is the first post in the [ML Basics and Concepts series]({% post_url 2026-08-14-ml-basics-and-concepts-series %})._

A machine learning model must do more than fit its training data. It must learn a
pattern that continues to work on examples it has never seen. Bias and variance
describe two different ways this can go wrong:

- **High bias** means the model is too restricted to capture the relevant pattern.
  It **underfits**.
- **High variance** means the model is too sensitive to the particular training
  sample. It **overfits**.

The best model is not necessarily the most complex one or the one with the lowest
training error. It is the model that generalizes best to the data we will encounter
after training.

> Here, **bias** means statistical bias in a learning procedure. It is different
> from social or demographic bias in data and model predictions.

## The learning setup

Suppose examples are drawn from an unknown distribution

$$
(X,Y) \sim P(X,Y),
$$

and a training set $D$ is sampled from this distribution. A learning algorithm
uses $D$ to produce a predictor $\hat f_D$. If we drew a different training set,
we would generally obtain a different predictor.

This gives us two questions:

1. On average, how far is the learned prediction from the true relationship?
2. How much does the learned prediction change when the training sample changes?

The first question is about **bias**. The second is about **variance**.

## Bias: error from restrictive assumptions

Bias measures the systematic error introduced by the assumptions of the model and
learning algorithm. A high-bias model cannot represent the important structure in
the data, even when it is trained correctly.

Imagine fitting a straight line to a relationship that is strongly curved. More
data will locate the best straight line more precisely, but it will not make that
line curved. The model class itself is too restrictive.

Typical signs of high bias are:

- training error is high;
- validation error is also high;
- the gap between training and validation error is relatively small;
- increasing the amount of training data produces little improvement.

High bias is therefore associated with **underfitting**.

Possible responses include using a richer model, constructing more informative
features, reducing excessive regularization, or fixing an optimization problem that
prevents the model from fitting even the training data.

## Variance: error from sensitivity to the sample

Variance measures how much the learned predictor would change if it were trained on
a different dataset sampled from the same population. A high-variance model learns
not only the underlying pattern but also accidental details and noise in its training
sample.

A very flexible curve may pass through every training point while behaving
unpredictably between those points. Its training error can be extremely low, yet its
error on new data can be much higher.

Typical signs of high variance are:

- training error is low;
- validation error is substantially higher;
- performance changes noticeably across folds, random seeds, or resampled datasets;
- adding representative training data improves validation performance.

High variance is therefore associated with **overfitting**.

Possible responses include collecting more representative data, strengthening
regularization, simplifying the model, using data augmentation, stopping training
earlier, or averaging several models through an ensemble.

## The bias--variance trade-off

For squared-error regression, the expected prediction error at an input $x$ has
the familiar decomposition

$$
\mathbb{E}_D\!\left[(Y-\hat f_D(x))^2\mid X=x\right]
=
\operatorname{Bias}(x)^2
+ \operatorname{Variance}(x)
+ \sigma^2(x),
$$

where

$$
\operatorname{Bias}(x)
=
\mathbb{E}_D[\hat f_D(x)]-f(x),
$$

and

$$
\operatorname{Variance}(x)
=
\mathbb{E}_D\!\left[
\left(\hat f_D(x)-\mathbb{E}_D[\hat f_D(x)]\right)^2
\right].
$$

and $\sigma^2(x)$ is irreducible noise in $Y$ given $X=x$.

Increasing model flexibility often reduces bias because the model can represent more
complex patterns. The same flexibility can increase variance because the fitted model
has more ways to respond to fluctuations in a finite training set. Regularization
moves in the opposite direction: it usually increases bias while reducing variance.

The trade-off is not a law that one side must always worsen when the other improves.
Better features, a more appropriate inductive bias, or more representative data can
improve both. It is nevertheless a useful way to diagnose errors.

For classification and other losses, the clean squared-error equation does not carry
over unchanged. Bias and variance remain useful concepts, but their exact
decomposition depends on the loss and the definition being used.

## A practical diagnosis

Training and validation errors provide a useful first check:

| Observation                                         | Likely problem                            | Useful next step                                                                 |
| --------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
| High training error and high validation error       | High bias / underfitting                  | Increase suitable capacity, improve features, or reduce excessive regularization |
| Low training error and much higher validation error | High variance / overfitting               | Add data or augmentation, regularize, simplify, or ensemble                      |
| Low training and validation error                   | Good fit on the current distribution      | Test robustness, calibration, and distribution shift                             |
| High training error despite a powerful model        | Optimization or data problem may dominate | Check labels, preprocessing, loss, learning rate, and convergence                |

These are diagnostics, not proofs. A validation set drawn from the wrong distribution
can hide both underfitting and overfitting. Label leakage can make validation error
look excellent while real deployment performance is poor.

## Bayes error: the ideal classifier's limit

Even with unlimited data and a sufficiently capable model, classification error may
not be zero. The input can be ambiguous, relevant information can be missing, labels
can be noisy, or the outcome itself can be stochastic.

Assume we know the true conditional distribution $P(Y\mid X)$. Under 0--1 loss,
the **Bayes classifier** predicts the most probable class:

$$
f^*(x)=\arg\max_y P(Y=y\mid X=x).
$$

Its expected error is the **Bayes error** or **Bayes risk**:

$$
R^*
=
\mathbb{E}_X\left[1-\max_y P(Y=y\mid X)\right].
$$

No classifier using the same input $X$, evaluated on the same distribution with the
same 0--1 loss, can achieve a lower expected error. Bayes error is therefore the
theoretical **lower bound** for that learning problem.

The qualification “using the same input” matters. Suppose two classes look identical
in an image but can be distinguished with an additional sensor. The image-only task
can have positive Bayes error even though the richer task does not. What appears to be
irreducible error may be uncertainty created by an incomplete representation of the
world.

## Why Bayes error can only be estimated

In real problems, we do not know the true $P(Y\mid X)$. We only observe a finite
sample from the world, so we generally cannot compute $R^*$ exactly. We can only
construct proxies.

### Fit a calibrated probabilistic model

A direct approach is to train a probabilistic classifier—such as logistic regression
or a neural network with a softmax output—to estimate

$$
\widehat P(Y\mid X)=P(Y\mid X,\theta),
$$

where $\theta$ is learned from data. Given a large independent test set
$\{x_i\}_{i=1}^N$, we can form the plug-in estimate

$$
\widehat R_{\text{plugin}}
=
\frac{1}{N}\sum_{i=1}^N
\left(1-\max_y \widehat P(Y=y\mid x_i)\right).
$$

This is intuitive for deep learning because the model already outputs an estimated
class distribution. It is reliable only when those probabilities are close to the
true conditional probabilities. Good calibration is important: a model that reports
90% confidence should be correct roughly 90% of the time on comparable examples.
Calibration alone is not sufficient, however, because aggregate calibration can hide
large local errors in $\widehat P(Y\mid X=x)$.

The test set must also be independent and representative of the deployment
distribution. Otherwise, the estimate describes the wrong prediction problem.

### Estimate class-conditional densities

For a simpler low-dimensional problem, we can estimate each class-conditional density
$p(x\mid Y=k)$ and the class prior $P(Y=k)$. Bayes' rule then gives

$$
P(Y=k\mid x)
=
\frac{p(x\mid Y=k)P(Y=k)}
{\sum_j p(x\mid Y=j)P(Y=j)}.
$$

Once this posterior has been estimated, it can be inserted into the Bayes-risk
formula. Gaussian density models, kernel density estimation, and nearest-neighbor
probability estimates are possible choices.

This approach becomes difficult in high-dimensional spaces. Density estimation in
image or language input spaces usually requires an enormous amount of data and strong
modeling assumptions, so the resulting Bayes-error estimate can be dominated by
density-estimation error.

### Repeated labels and human-level performance

Collecting several independent labels for the same input gives more information than
forcing a single “ground-truth” label. If 90 out of 100 reliable annotators label an
example as class A and 10 label it as class B, we might estimate

$$
P(A\mid x)\approx 0.9,
\qquad
P(B\mid x)\approx 0.1.
$$

The estimated local Bayes error for that example is then

$$
1-\max\{0.9,0.1\}=0.1.
$$

This interpretation assumes that the variation in labels reflects genuine
conditional uncertainty. Annotators can instead share systematic mistakes, use
different labeling rules, or lack relevant context. Clear instructions, reliable
annotators, independent judgments, and adjudication are therefore important.

Human error is not the formal Bayes lower bound: people can make systematic mistakes,
and a classifier may outperform them. It is better understood as a practical measure
of task difficulty. Repeated disagreement can reveal ambiguity, missing information,
or label noise, while expert consensus can expose errors in the dataset labels.

### Use strong models as a practical proxy

In modern high-dimensional problems, exact estimation is usually impossible. If
several powerful model families trained on large datasets converge to roughly the
same test error, their performance provides a useful practical reference. For any
particular classifier $h$,

$$
R^*\leq R(h).
$$

The same applies to the strongest model we have trained:

$$
R^*\leq R(h_{\text{best}}).
$$

If the best network obtains 3% test error, then—after accounting for finite-test-set
uncertainty—we have evidence that $R^*$ is no greater than approximately 3%. We cannot
conclude that the Bayes error is exactly 3%.

However, this achieved error is normally an **upper bound** on Bayes error, not proof
of the lower bound: the model can still have approximation error, estimation error,
optimization error, or distribution mismatch. A larger model is a better proxy only
when its predictions are supported by sufficient data and evaluated properly.

None of these approaches recovers the true world model from limited data. They give
increasingly useful approximations to the best achievable performance under a clearly
defined input, target, distribution, and loss.

## Reducible and irreducible error

A helpful conceptual summary is

$$
\text{generalization error}
\approx
\text{Bayes error}
+ \text{approximation error}
+ \text{estimation error}
+ \text{optimization error}.
$$

This is a conceptual accounting for classification, not a universal exact additive
identity.

- **Bayes error** is irreducible without changing the information, target, loss, or
  data-generating process.
- **Approximation error** arises when the model family cannot represent a sufficiently
  good decision rule; it is closely related to high bias.
- **Estimation error** arises because the learner sees only finite data; it is closely
  related to high variance.
- **Optimization error** arises when training fails to find the best available model
  in the chosen family.

This distinction prevents a common mistake: blaming every stubborn error on model
capacity. Sometimes a model underfits. Sometimes it overfits. Sometimes training is
poor. And sometimes the input simply does not contain enough information to determine
the label reliably.

## Takeaways

- High bias usually appears as underfitting: both training and validation errors stay
  high.
- High variance usually appears as overfitting: training error is low but validation
  error is much higher.
- Model capacity, regularization, data quantity, and inductive bias determine where a
  learner sits between these two failure modes.
- Bayes error is the theoretical lower bound for a fixed prediction problem, but the
  true $P(Y\mid X)$ is unknown in practice.
- Strong models, reliable human annotators, and repeated labels can provide useful
  proxies for task difficulty, but their observed errors are not themselves proofs of
  the Bayes lower bound.
