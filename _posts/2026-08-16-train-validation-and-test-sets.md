---
layout: post
title: "Train, Validation, and Test Sets: Measuring Generalization Correctly"
date: 2026-08-16 09:00:00 +1000
description: How to split data, prevent leakage, choose models, and obtain an honest estimate of performance on unseen examples.
tags: [machine-learning, fundamentals, generalization, evaluation, tutorial]
categories: ["ML Basics and Concepts"]
toc:
  beginning: true
related_posts: true
math: true
---

_This is the second post in the [ML Basics and Concepts series]({% post_url 2026-08-14-ml-basics-and-concepts-series %}). It builds on [Bias and Variance]({% post_url 2026-08-14-bias-and-variance %})._

A model can achieve excellent performance on examples it has already seen and still
fail on new ones. The central purpose of dataset splitting is to separate **learning**
from **evaluation** so that we can answer the question that actually matters:

> How well will this model perform on future data drawn from the target population?

The familiar train, validation, and test split is not merely an organizational
convention. Each subset has a different statistical role. When those roles are mixed,
information leaks into the learning process and the reported result becomes
optimistic.

## The three roles

Suppose we begin with a dataset $D$ sampled from some population. We partition it into
three disjoint subsets:

$$
D = D_{\text{train}} \cup D_{\text{val}} \cup D_{\text{test}}.
$$

| Subset             | Primary role                  | What may depend on it?                                                              |
| ------------------ | ----------------------------- | ----------------------------------------------------------------------------------- |
| **Training set**   | Fit model parameters          | Weights, coefficients, learned representations, and fitted preprocessing            |
| **Validation set** | Make development decisions    | Architecture, hyperparameters, thresholds, feature choices, and stopping time       |
| **Test set**       | Estimate final generalization | Ideally nothing; it is used only after the model and evaluation procedure are fixed |

The key distinction is between **model parameters** and **development choices**.
Gradient descent may fit millions of weights using the training set, while the
validation set determines choices such as depth, learning rate, regularization
strength, data augmentation, or the epoch at which training stops. Both are forms of
learning from data.

The test set is different. It simulates data that remained unavailable throughout
development.

## Why training performance is not enough

A sufficiently flexible model can memorize its training examples. Its empirical
training error,

$$
\widehat R_{\text{train}}(h)
=
\frac{1}{|D_{\text{train}}|}
\sum_{(x_i,y_i)\in D_{\text{train}}}
\ell\!\left(h(x_i),y_i\right),
$$

measures fit to the sample used for learning. It does not provide an independent
estimate of the population risk

$$
R(h)=\mathbb{E}_{(X,Y)\sim P}\left[\ell(h(X),Y)\right].
$$

The gap between these quantities is the generalization gap. A validation or test set
approximates population risk only when it has not already influenced the predictor in
the way it is being evaluated.

## The validation set is part of training

Although no gradient may be computed directly from the validation set, repeated
validation feedback guides development. Consider a workflow that tries 100 model
configurations and keeps the one with the highest validation accuracy. The selected
model benefits from information in that validation set: configurations that happened
to match its noise were more likely to win.

This is **overfitting to the validation set**. It becomes more severe when:

- many configurations are compared;
- the validation set is small or noisy;
- decisions are repeatedly made after inspecting individual errors;
- researchers manually adapt features or rules to validation examples;
- only the best result across many random seeds is reported.

Validation data are therefore development data. They should not be treated as an
untouched estimate of final performance.

## Why the test set should be used once

The test set is intended to answer one final question after the model, preprocessing,
decision threshold, and metric have been fixed. If its results influence another
round of development, it has begun to act as a validation set.

This does not mean a test set can literally be evaluated only once. The principle is
that it should not provide an adaptive feedback loop. If we inspect the test result,
change the system, and inspect it again many times, the final system can overfit the
test set even without directly training on its examples.

The same problem occurs at the community level. A public benchmark can become a
shared validation set after years of model selection, paper writing, and leaderboard
tuning. Honest progress then requires fresh test data or a hidden evaluation server.

## Choosing the split proportions

Rules such as 70/15/15 or 80/10/10 are starting points, not laws. The right split is
determined by the number of examples required for two competing goals:

1. enough training data to learn a useful model;
2. enough validation and test data to compare and estimate performance precisely.

With millions of independent examples, even 1% may produce a large evaluation set.
With a small dataset, fixed holdout sets may be too noisy, making cross-validation
more useful.

For a fixed classifier evaluated with **0--1 loss**, the empirical test error $\hat p$
is the proportion of misclassified examples:

$$
\hat p
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}\!\left[h(x_i)\neq y_i\right].
$$

Each test outcome is therefore treated as a Bernoulli variable: 1 when the prediction
is wrong and 0 when it is correct. On $N$ approximately independent test examples,
the estimated standard error of this proportion is

$$
\operatorname{SE}(\hat p)
\approx
\sqrt{\frac{\hat p(1-\hat p)}{N}}.
$$

This standard error measures **finite-test-set sampling uncertainty**. It describes
how much the measured error would typically vary if we evaluated the same fixed
classifier on another independent test set of size $N$ from the same distribution. A
rough 95% confidence interval is $\hat p\pm1.96\operatorname{SE}(\hat p)$.

It does not measure variation caused by retraining on different training sets,
uncertainty in an individual prediction, or distribution shift. When examples are
grouped or correlated, the effective sample size can be far smaller than $N$, so a
group-aware confidence interval or bootstrap procedure is more appropriate.

## Choose the split unit before the split method

The most important decision is often not the percentage but the **unit that must be
independent**. Splitting individual rows is wrong when several rows come from the same
underlying entity.

Examples include:

- multiple images of the same patient;
- several frames from the same video;
- repeated transactions from one customer;
- patches cropped from the same source image;
- augmented versions of one original example;
- messages from the same conversation;
- measurements from the same machine or location.

If related examples appear in both training and test sets, the model can exploit
entity-specific information. The measured performance may then describe recognition
of known entities rather than generalization to new ones.

The safe rule is: **split groups first, then construct examples or augmentations
within each split**.

## Common splitting strategies

### Random split

A random split is appropriate when examples are approximately independent and
identically distributed, and future data will come from the same stable population.
It is simple and often effective, but it does not protect against hidden groups,
duplicates, or temporal dependence.

### Stratified split

A stratified split approximately preserves selected proportions—most commonly class
frequencies—across subsets. It is useful when a random split might place too few
minority-class examples in validation or test data.

Stratification should not be confused with balancing. Preserving a naturally rare
class keeps the evaluation representative; oversampling that class changes the
evaluation distribution unless metrics are reweighted appropriately.

### Grouped split

A grouped split keeps all examples from an entity in one subset. For medical imaging,
the split is usually by patient rather than image. For document chunks, it may be by
source document. For federated learning, it may be by client.

This strategy evaluates the harder and often more relevant question: can the model
generalize to unseen groups?

### Time-based split

When a model predicts the future from the past, the split should respect time:

$$
t_{\text{train}} < t_{\text{val}} < t_{\text{test}}.
$$

Randomly moving future observations into the training set leaks information that
would not have existed at deployment time. Time-based evaluation also exposes
temporal drift in user behavior, prevalence, policy, sensors, or market conditions.

### Spatial or domain-based split

If deployment involves new hospitals, countries, cameras, environments, or data
providers, holding out entire domains can be more informative than randomly holding
out examples within every domain. The correct split should resemble the intended
generalization boundary.

## Data leakage

**Data leakage** occurs when information unavailable at prediction time influences
training or model selection. Leakage can make a weak model appear remarkably strong.

### Target leakage

A feature directly or indirectly contains information about the target that would not
be available when the prediction is made. For example, using a treatment prescribed
after diagnosis to predict that diagnosis leaks the outcome timeline.

### Preprocessing leakage

Any preprocessing step that learns from data must be fitted using the training set
only. This includes:

- normalization means and variances;
- missing-value imputation;
- vocabulary construction;
- feature selection;
- dimensionality reduction;
- learned tokenization or representations;
- class-balancing procedures such as synthetic oversampling.

The correct order is:

1. split the raw examples;
2. fit preprocessing on the training split;
3. apply the fitted transformation to validation and test splits.

A pipeline abstraction is valuable because it binds preprocessing and model fitting
into one procedure and reduces accidental leakage during cross-validation.

### Duplicate and augmentation leakage

Exact duplicates, near-duplicates, or augmented versions of the same example must not
cross split boundaries. Deduplication should consider semantics, not only identical
file hashes: resized images, copied text with minor edits, and overlapping time windows
can still reveal almost the same example.

### Selection leakage

Selecting features, thresholds, prompts, or checkpoints after observing test results
leaks test information into the final system. Even an apparently harmless manual
inspection can become selection leakage when it changes what is submitted or
reported.

## Cross-validation

When data are limited, one validation split may give an unstable comparison. In
$K$-fold cross-validation, the development data are divided into $K$ folds. Each fold
serves as validation data once while the remaining $K-1$ folds are used for training:

$$
\widehat R_{\text{CV}}
=
\frac{1}{K}\sum_{k=1}^{K}\widehat R^{(k)}_{\text{val}}.
$$

The average gives a more stable estimate for model selection, and variation across
folds reveals sensitivity to the sampled data. The fold construction must still obey
the correct unit: use stratified folds for class proportions, group folds for entities,
and forward-chaining splits for time series.

Cross-validation does not eliminate the need for an independent test set when a final
unbiased estimate is required. The usual workflow is:

1. reserve the test set;
2. use cross-validation within the remaining development data;
3. select the model and hyperparameters;
4. optionally refit the fixed procedure on all development data;
5. evaluate once on the reserved test set.

When both model selection and performance estimation must use resampling, **nested
cross-validation** places model selection in an inner loop and evaluation in an outer
loop. It is more expensive but avoids evaluating a hyperparameter search on the same
folds that selected it.

## Distribution mismatch

Independence is not enough. Validation and test data must also represent the
distribution we care about.

Suppose a model is trained on studio photographs but deployed on mobile-phone images.
A random split of the studio dataset can produce an accurate estimate of performance
on more studio photographs while saying little about deployment. The estimate is
statistically valid for the wrong population.

Useful questions include:

- Do class frequencies match deployment?
- Are devices, locations, demographics, and time periods represented?
- Does data collection reproduce the information available at prediction time?
- Are rare but important operating conditions present?
- Is the label definition consistent across splits and deployment?

Sometimes training data are intentionally broader or rebalanced. That can be useful,
but validation and test sets should still reflect the target population or be
reweighted to a clearly defined target distribution.

If multiple deployment domains matter, report results separately rather than hiding
failures inside a single average.

## Connecting splits to bias and variance

The [previous article]({% post_url 2026-08-14-bias-and-variance %}) used training and
validation errors as practical diagnostics. With a trustworthy split, their pattern
provides an initial signal:

| Training result | Validation result   | Likely interpretation                                                          |
| --------------- | ------------------- | ------------------------------------------------------------------------------ |
| Poor            | Poor and similar    | High bias, weak features, optimization failure, or noisy labels                |
| Strong          | Much poorer         | High variance, leakage in training evaluation, or distribution mismatch        |
| Strong          | Strong              | Good fit on the validation distribution; final test evaluation is still needed |
| Poor            | Surprisingly strong | Small-sample noise, inconsistent pipelines, or an implementation error         |

Learning curves make the diagnosis more informative. Plot training and validation
performance as the training-set size increases. A persistent high error in both
curves suggests bias. A large gap that narrows with more data suggests variance.

These patterns are clues rather than proofs. Before changing model capacity, verify
the split, metric, labels, and preprocessing pipeline.

## A practical workflow

1. **Define deployment first.** Specify the population, prediction time, available
   inputs, target, and metric.
2. **Identify dependence.** Decide whether the split unit is an example, person,
   document, device, location, or time period.
3. **Reserve the test set.** Keep it inaccessible during routine development.
4. **Split before preprocessing or augmentation.** Fit every learned transformation
   using training data only.
5. **Develop with validation data.** Use a holdout set or an appropriate form of
   cross-validation.
6. **Track all adaptive decisions.** Architecture searches, prompt changes, threshold
   selection, and manual error analysis all consume validation information.
7. **Freeze the procedure.** Fix preprocessing, hyperparameters, checkpoint selection,
   threshold, and metric.
8. **Evaluate the test set once for the final claim.** Report uncertainty and relevant
   subgroup or domain results.
9. **Create fresh test data after further iteration.** Once test feedback changes the
   system, that test set is no longer fully untouched.

## Takeaways

- Training data fit parameters; validation data guide development; test data estimate
  final generalization.
- Repeatedly adapting to validation or test results causes evaluation overfitting.
- Split by the unit that must generalize: patient, document, device, location, or time,
  not automatically by row.
- Perform learned preprocessing and augmentation only after the split boundaries are
  established.
- Cross-validation reduces dependence on one validation split but does not excuse
  leakage or an inappropriate split strategy.
- An independent test set is useful only when it represents the deployment population.
- Report uncertainty: test performance is an estimate based on a finite sample, not a
  permanent property of the model.
