---
layout: post
title: The ML Basics and Concepts Series
date: 2026-08-14 08:00:00 +1000
description: A guided series on the core ideas that shape how machine learning models learn, generalize, and fail.
tags: [machine-learning, fundamentals, tutorial]
categories: ["ML Basics and Concepts"]
series_index: true
related_posts: true
---

This series collects my notes on the foundations of machine learning. The goal is
to connect familiar terms to the reasoning behind them: what each concept means,
how it appears in experiments, and what we can do about it in practice.

The guiding question is:

> What does a model's error tell us about the data, the learning algorithm, and
> the task itself?

## Reading order

1. [Bias and Variance]({% post_url 2026-08-14-bias-and-variance %})

   Why high bias leads to underfitting, why high variance leads to overfitting,
   and where Bayes error sets the irreducible limit.

2. [Train, Validation, and Test Sets]({% post_url 2026-08-16-train-validation-and-test-sets %})

   How to measure generalization without leaking information, overfitting the
   evaluation, or testing on data that does not represent deployment.

## Topics to expand later

Future notes will build on this foundation with topics such as:

- regularization and model capacity;
- loss functions and evaluation metrics;
- maximum likelihood and maximum a posteriori estimation;
- calibration, uncertainty, and distribution shift.

The series will grow as I add concise explanations, mathematical intuition, and
practical diagnostics for each concept.
